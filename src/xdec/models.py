"""Decoder-architecture pilot: E1-E7 (+ R00 = CoordMLP reference retrained under the same protocol).

Every decoder: forward(feats, g) -> Z (R, 576, 768) on the 24x24 C0 crop grid, in the RAW final-LN DINOv2 feature
space (same space as the teacher and as the Bicubic tensor B). Input = feats[12] only (frozen F336 final layer); the
teacher never enters (only the loss sees it). Geometry g = collate_geoms batch + roi_frame (src.xdec.geometry).

Common pieces (hidden d = 256, 8 heads, FFN ratio 4, pre-LN, dropout 0, 2 blocks), all taken from ONE canonical
initialisation (torch seed 0, `canonical_state()`), so every module that two conditions share starts from the SAME
weights (not just the same seed):
  mem_in  : LN(768) -> Linear(768, 256)          memory tokens (full-image tokens or ROI tokens)
  q_in    : LN(768) -> Linear(768, 256)          Bicubic query projection (E1, E4, E6)
  mask_tok: (256,) learned token, trunc-normal 0.02 (E2, E3, E5, E7)
  pe      : PosEnc, one module for queries AND memory (same ROI-relative frame): Fourier(sym-log(u - 1/2)),
            Fourier(sym-log(v - 1/2)) (2^k pi, k = -2..4), log cell size / 4 -> Linear -> 256. sym-log(x) = sign(x)
            log(1 + |x| / 0.5): monotone, identity-like inside the ROI (|x| <= 0.5 -> |s| <= log 2), compresses the large
            outside-ROI values of small ROIs WITHOUT clamping them.
  blk[i]  : ln_q, ln_kv, ca (cross-attn), ln_s, sa (self-attn), ln_f, ffn  (a module is used only where the
            condition's definition has it)
  head_ln : LN(256);  head: Linear(256, 768)  (direct = default init;  skip = weight & bias ZERO)
  output  : direct  Z = mu12 + sd12 * head(head_ln(h))        (mu12 / sd12 = fixed teacher channel stats, = R00 mean_T/std_T)
            skip    Z = B    + sd12 * head(head_ln(h))        (zero-init -> Z == B exactly at step 0; addition in fp32,
                                                               no normalisation after the addition)

  E1  query q_in(B) + pe(q)         blocks [SA, FFN] x2 (no cross-attn)        skip
  E2  query mask_tok + pe(q)        blocks [CA(ROI tokens), FFN] x2            direct
  E3  query mask_tok + pe(q)        blocks [CA(full 576), FFN] x2              direct
  E4  query q_in(B) + pe(q)         blocks [CA(full 576), FFN] x2              direct
  E5  query mask_tok + pe(q)        blocks [CA(full 576), FFN] x2              skip
  E6  query q_in(B) + pe(q)         blocks [CA(full 576), FFN] x2              skip
  E7  query mask_tok + pe(q)        blocks [CA(full 576), SA, FFN] x2          direct
Memory kv = mem_in(token) + pe(token position), LN'd per block (ln_kv). Masks: E2 cross-attn key mask = ROI-token
selection & ~batch padding (variable length, raw tokens, never resampled); E3-E7 = all 576 tokens of the ROI's image.
Self-attention key mask (E1, E7) = C0 valid-coverage (valid_w > 0): invalid (letterbox padding) queries are never
keys/values for anyone; they still get an output (loss weight 0, R0/R2 ignore them).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.xdec import geometry as XG

D, DH, HEADS, FFN_R, NBLK = 768, 256, 8, 4, 2
KMIN, KMAX = -2, 4
COND_SPEC = {
    "E1": dict(query="bic", mem=None, sa=True, ca=False, out="skip"),
    "E2": dict(query="mask", mem="roi", sa=False, ca=True, out="direct"),
    "E3": dict(query="mask", mem="full", sa=False, ca=True, out="direct"),
    "E4": dict(query="bic", mem="full", sa=False, ca=True, out="direct"),
    "E5": dict(query="mask", mem="full", sa=False, ca=True, out="skip"),
    "E6": dict(query="bic", mem="full", sa=False, ca=True, out="skip"),
    "E7": dict(query="mask", mem="full", sa=True, ca=True, out="direct"),
}


class Attn(nn.Module):
    def __init__(self, d=DH, heads=HEADS):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, kv, allow=None):
        R, Lq, d = x.shape
        Lk = kv.shape[1]
        q = self.q(x).view(R, Lq, self.h, -1).transpose(1, 2)
        k = self.k(kv).view(R, Lk, self.h, -1).transpose(1, 2)
        v = self.v(kv).view(R, Lk, self.h, -1).transpose(1, 2)
        m = None if allow is None else allow[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(R, Lq, d))


def ffn(d=DH, r=FFN_R):
    return nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d))


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs() / 0.5)


class PosEnc(nn.Module):
    def __init__(self, d=DH):
        super().__init__()
        self.register_buffer("f", (2.0 ** torch.arange(KMIN, KMAX + 1, dtype=torch.float32)) * math.pi)
        self.nf = KMAX - KMIN + 1
        self.lin = nn.Linear(4 * self.nf + 2, d)

    def features(self, uv, size):
        s = symlog(uv - 0.5)                                                 # (R, N, 2)
        x = s[..., None] * self.f                                            # (R, N, 2, nf)
        four = torch.cat([x.sin(), x.cos()], -1).flatten(-2)                 # (R, N, 4 nf)
        ls = (size.clamp_min(1e-6).log() / 4)[:, None].expand(-1, uv.shape[1], -1)
        return torch.cat([four, ls], -1)

    def forward(self, uv, size):
        return self.lin(self.features(uv, size))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_q, self.ln_kv, self.ca = nn.LayerNorm(DH), nn.LayerNorm(DH), Attn()
        self.ln_s, self.sa = nn.LayerNorm(DH), Attn()
        self.ln_f, self.ffn = nn.LayerNorm(DH), ffn()


class Canonical(nn.Module):
    """Every module any condition can use, built once with torch seed 0 -> the shared initial values."""

    def __init__(self):
        super().__init__()
        self.mem_in = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DH))
        self.q_in = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DH))
        self.mask_tok = nn.Parameter(torch.zeros(DH))
        nn.init.trunc_normal_(self.mask_tok, std=0.02)
        self.pe = PosEnc()
        self.blk = nn.ModuleList([Block() for _ in range(NBLK)])
        self.head_ln = nn.LayerNorm(DH)
        self.head = nn.Linear(DH, D)


_CANON = {}


def canonical_state():
    if "sd" not in _CANON:
        rng = torch.random.get_rng_state()
        torch.manual_seed(0)
        _CANON["sd"] = {k: v.clone() for k, v in Canonical().state_dict().items()}
        torch.random.set_rng_state(rng)
    return _CANON["sd"]


class XDec(nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.cond, self.spec = cond, COND_SPEC[cond]
        sp = self.spec
        self.arch = f"xdec_{cond}"
        if sp["mem"] is not None:
            self.mem_in = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DH))
        if sp["query"] == "bic":
            self.q_in = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DH))
        else:
            self.mask_tok = nn.Parameter(torch.zeros(DH))
        self.pe = PosEnc()
        self.blk = nn.ModuleList()
        for _ in range(NBLK):
            b = nn.Module()
            if sp["ca"]:
                b.ln_q, b.ln_kv, b.ca = nn.LayerNorm(DH), nn.LayerNorm(DH), Attn()
            if sp["sa"]:
                b.ln_s, b.sa = nn.LayerNorm(DH), Attn()
            b.ln_f, b.ffn = nn.LayerNorm(DH), ffn()
            self.blk.append(b)
        self.head_ln = nn.LayerNorm(DH)
        self.head = nn.Linear(DH, D)
        self.register_buffer("mu12", torch.zeros(D))
        self.register_buffer("sd12", torch.ones(D))
        # ---- shared initial values from the canonical state (subset; every parameter of this model must be covered)
        can = canonical_state()
        own = self.state_dict()
        missing = [k for k in own if k not in can and k not in ("mu12", "sd12")]
        assert not missing, missing
        self.load_state_dict({k: (can[k] if k in can else own[k]) for k in own})
        if sp["out"] == "skip":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def set_stats(self, mu, sd):
        self.mu12.copy_(mu)
        self.sd12.copy_(sd)

    # ------------------------------------------------------------------ pieces
    def queries(self, B, g):
        uv, size = XG.query_uv(g)
        pe = self.pe(uv, size)
        if self.spec["query"] == "bic":
            return self.q_in(B) + pe
        return self.mask_tok.expand(pe.shape[0], pe.shape[1], -1) + pe

    def memory(self, F12, g):
        """-> kv (R, M, 256), allow (R, M) bool or None."""
        if self.spec["mem"] == "full":
            kv = self.mem_in(F12)[g["img_index"]]                         # (R, 576, 256), projected once per image
            uv, size = XG.full_uv(g)
            return kv + self.pe(uv, size), None
        # ROI memory: the raw ROI tokens of the ROI(+ring) gather, ring and batch padding masked out (variable length)
        tok = MS_gather(F12, g)                                            # (R, n_ctx, 768), zeros at padded slots
        uv_all, size = XG.full_uv(g)
        uv = torch.gather(uv_all, 1, g["tok"][..., None].expand(-1, -1, 2))
        return self.mem_in(tok) + self.pe(uv, size), XG.roi_token_mask(g)

    def trunk(self, x, kv, allow, qvalid):
        for b in self.blk:
            if self.spec["ca"]:
                x = x + b.ca(b.ln_q(x), b.ln_kv(kv), allow)
            if self.spec["sa"]:
                h = b.ln_s(x)
                x = x + b.sa(h, h, qvalid)
            x = x + b.ffn(b.ln_f(x))
        return x

    def forward(self, feats, g, return_parts=False):
        F12 = feats[12]
        B = XG.bicubic_roi(F12, g) if (self.spec["query"] == "bic" or self.spec["out"] == "skip") else None
        x = self.queries(B, g)
        kv, allow = self.memory(F12, g) if self.spec["mem"] is not None else (None, None)
        qvalid = XG.valid_query_mask(g) if self.spec["sa"] else None
        h = self.trunk(x, kv, allow, qvalid)
        y = self.head(self.head_ln(h)).float()
        Z = (B if self.spec["out"] == "skip" else self.mu12) + y * self.sd12
        if return_parts:
            return Z, dict(B=B, head=y)
        return Z

    def groups(self):
        """Parameter groups for gradient / update logging."""
        gr = {"query_gen": [], "memory_in": [], "pos_enc": [], "trunk": [], "head": []}
        for n, p in self.named_parameters():
            k = ("query_gen" if n.startswith(("q_in", "mask_tok")) else "memory_in" if n.startswith("mem_in")
                 else "pos_enc" if n.startswith("pe.") else "head" if n.startswith(("head.", "head_ln")) else "trunk")
            gr[k].append(p)
        gr["head_last_linear"] = list(self.head.parameters())
        gr["all_before_last_linear"] = [p for n, p in self.named_parameters() if not n.startswith("head.")]
        return gr


def MS_gather(F12, g):
    from models.roi_reader import gather_roi
    return gather_roi(F12, g)


class R00Dec(nn.Module):
    """R00 = multilevel_relation_decoder MLRDecoder('final'): F = F_final + Adapter(F_final) on the ROI(+ring) tokens
    (zero-init adapter) -> CoordMLP body (k = 4 nearest ROI(+ring) tokens + Fourier offsets, ROI summary + box vector,
    query positions) -> out * std_T + mean_T (mean_T / std_T = mu12 / sd12, the same constants as E-series heads).
    Unchanged code (models_mlr). Fresh init (torch seed 0); the old gen_decoders mlp_s0 init is not on this server."""

    arch = "xdec_R00"

    def __init__(self):
        super().__init__()
        import src.msd.common as MSC
        self.cond = "R00"
        rng = torch.random.get_rng_state()
        torch.manual_seed(0)
        self.adapter = MSC.MM.Adapter()
        self.body = MSC.MM.CoordMLP()
        torch.random.set_rng_state(rng)
        self.register_buffer("P", MSC.query_coords_t())

    def set_stats(self, mu, sd):
        self.body.set_stats(mu, sd)

    def forward(self, feats, g, return_parts=False):
        Fr = MS_gather(feats[12], g)
        Fin = Fr + self.adapter(Fr).masked_fill(g["pad"][..., None], 0.0)
        Z = self.body(Fin, g, self.P)
        return (Z, dict(B=None, head=None)) if return_parts else Z

    def groups(self):
        return {"adapter": list(self.adapter.parameters()), "body": list(self.body.parameters()),
                "head_last_linear": list(self.body.mlp[-1].parameters()),
                "all_before_last_linear": [p for n, p in self.named_parameters() if not n.startswith("body.mlp.3")]}


def build(cond):
    if cond == "R00":
        return R00Dec()
    return XDec(cond)


def build_from_arch(arch):
    assert arch.startswith("xdec_"), arch
    return build(arch[len("xdec_"):])


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))
