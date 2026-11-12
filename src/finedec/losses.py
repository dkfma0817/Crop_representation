"""Losses of FA-A and FA-B. All loss statistics in true FP32 (autocast off, TF32 off; previous TF32 setting restored).

FA-A  mean / residual KD on the post-norm features. Per parent p of a ROI (children = the parent's VALID children c, letterbox-invalid and
      padding excluded): mu = mean_c Y, r_c = Y_c - mu. Per ROI: L_mean = mean over parents with >= 1 valid child of ||mu_S - mu_T||^2 / D,
      L_res = mean over valid children of ||r_S - r_T||^2 / D; L_A = L_mean / (v_mean + eps) + 1.0 * L_res / (v_res + eps), averaged over
      ROIs. v_mean, v_res: fixed constants from COCO train teacher features (src.finedec.prep), same parent / residual definitions.
FA-B  head-free patch CE + SIGReg, ported from the professor's reference kernel (dinov3_token_expansion_final.py, 2026-11-11):
      patch CE = -sum_a p_T,a log p_S,a with p = softmax(normalize(.) @ normalize(Q)^T / 0.07), same immutable queue snapshot for teacher and
      student, mean over the valid child tokens of an image, then mean over images (reference reduction). KL = CE - H(p_T) logged.
      SIGReg on the raw decoder child latent Z (before the correction heads / frozen norm): image-balanced, parent-diverse sampling without
      replacement (budget 1,024), 256 unit random directions, t in [0, 3] with 17 trapezoid points, Gaussian window, x 2 (symmetry) x M
      (sample count), mean over directions. New directions every optimizer step from a dedicated generator (seed BASE + step).
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

EPS = 1e-8


@contextmanager
def fp32():
    prev, prec = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        torch.set_float32_matmul_precision(prec)


def parent_stats(Y, lp, valid):
    """Y (R, N, D); lp (R, N) local parent index; valid (R, N) bool -> mu per child (R, N, D), parent count per child (R, N), and per-parent
    (mu (R, P, D), cnt (R, P))."""
    R, N, D = Y.shape
    P = int(lp.max()) + 1
    w = valid.float()[..., None]
    s = torch.zeros(R, P, D, device=Y.device, dtype=Y.dtype).scatter_add_(1, lp[..., None].expand(-1, -1, D), Y * w)
    c = torch.zeros(R, P, 1, device=Y.device, dtype=Y.dtype).scatter_add_(1, lp[..., None], w)
    mu = s / c.clamp_min(1)
    return mu.gather(1, lp[..., None].expand(-1, -1, D)), mu, c[..., 0]


def mean_residual(Ys, Yt, lp, valid, v_mean, v_res, w_res=1.0):
    with fp32():
        Ys, Yt = Ys.float(), Yt.float()
        mus_c, mus, cnt = parent_stats(Ys, lp, valid)
        mut_c, mut, _ = parent_stats(Yt, lp, valid)
        pv = (cnt > 0).float()
        lm = ((mus - mut).pow(2).mean(-1) * pv).sum(1) / pv.sum(1).clamp_min(1)                 # per ROI
        vf = valid.float()
        lr = (((Ys - mus_c) - (Yt - mut_c)).pow(2).mean(-1) * vf).sum(1) / vf.sum(1).clamp_min(1)
        ok = vf.sum(1) > 0
        L_mean, L_res = lm[ok].mean(), lr[ok].mean()
        loss = L_mean / (v_mean + EPS) + w_res * L_res / (v_res + EPS)
        with torch.no_grad():
            child = ((Ys - Yt).pow(2).mean(-1) * vf).sum(1) / vf.sum(1).clamp_min(1)
            rS, rT = (Ys - mus_c), (Yt - mut_c)
            stats = dict(L_mean=float(L_mean), L_res=float(L_res), roi_child_mse=float(child[ok].mean()),
                         res_energy_student=float((rS.pow(2).mean(-1) * vf).sum() / vf.sum()), res_energy_teacher=float((rT.pow(2).mean(-1) * vf).sum() / vf.sum()))
    return loss, L_mean, L_res, stats


def patch_ce(Ys, Yt, valid, rimg, n_img, Q, tau=0.07, chunk=8192):
    """Per image: mean over its valid child tokens (all ROIs of the image), then mean over images. Q (A, D) immutable unit rows."""
    with fp32():
        img = rimg[:, None].expand_as(valid)[valid]
        s, t = F.normalize(Ys.float()[valid], dim=-1), F.normalize(Yt.float()[valid], dim=-1)
        ces, ents = [], []
        for i in range(0, len(s), chunk):
            ls = F.log_softmax(s[i:i + chunk] @ Q.T / tau, -1)
            with torch.no_grad():
                lt = F.log_softmax(t[i:i + chunk] @ Q.T / tau, -1)
                pt = lt.exp()
                ents.append(-(pt * lt).sum(-1))
            ces.append(-(pt * ls).sum(-1))
        ce_tok, ent_tok = torch.cat(ces), torch.cat(ents)
        cnt = torch.zeros(n_img, device=s.device).index_add_(0, img, torch.ones_like(ce_tok))
        ok = cnt > 0
        ce_img = torch.zeros(n_img, device=s.device).index_add_(0, img, ce_tok)[ok] / cnt[ok]
        ent_img = torch.zeros(n_img, device=s.device).index_add_(0, img, ent_tok.detach())[ok] / cnt[ok]
        ce, ent = ce_img.mean(), ent_img.mean()
    return ce, dict(patch_ce=float(ce), teacher_entropy=float(ent), patch_kl=float(ce) - float(ent))


def sample_latents(Z, valid, rimg, child_id, budget=1024, seed=0):
    """Image-balanced, without replacement, parents first (reference sample_child_latents adapted to ROI-major batches).
    Z (R, N, d); valid (R, N); rimg (R,); child_id (R, N) = cy * 48 + cx. Per image: valid children deduplicated over overlapping ROIs
    (first occurrence kept), parents in random order, children of a parent in random order, taken round-robin (one child of every
    parent, then a second child of every parent, ...); the image takes budget // n_img (+1 for the first budget % n_img images)."""
    g = torch.Generator(device=Z.device).manual_seed(int(seed))
    zs, img, cid = Z[valid], rimg[:, None].expand_as(valid)[valid], child_id[valid]
    imgs = torch.unique(img).tolist()
    parts = []
    for k, b in enumerate(imgs):
        m = torch.nonzero(img == b).flatten()
        o = torch.sort(cid[m], stable=True).indices
        c_sorted = cid[m][o]
        keep = torch.ones_like(c_sorted, dtype=torch.bool)
        keep[1:] = c_sorted[1:] != c_sorted[:-1]
        m = m[o[keep]]
        c = cid[m]
        par = (c // 48 // 2) * 24 + (c % 48) // 2
        up, pinv = torch.unique(par, return_inverse=True)
        prank = torch.randperm(len(up), generator=g, device=Z.device)[pinv]
        jitter = torch.rand(len(m), generator=g, device=Z.device)
        o2 = torch.argsort(pinv.double() * 2 + jitter.double())
        sp = pinv[o2]
        start = torch.ones_like(sp, dtype=torch.bool)
        start[1:] = sp[1:] != sp[:-1]
        pos = torch.arange(len(sp), device=Z.device)
        gstart = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
        within = torch.empty_like(pos)
        within[o2] = pos - gstart
        order = torch.argsort(within * (len(up) + 1) + prank)
        take = min(len(m), budget // len(imgs) + int(k < budget % len(imgs)))
        parts.append(zs[m[order[:take]]])
    return torch.cat(parts)


def sigreg(z, projections=256, points=17, t_max=3.0, seed=0):
    """Reference kernel (dinov3_token_expansion_final.sigreg), unchanged except the seed argument is a new value every step."""
    with fp32():
        z = z.float()
        gen = torch.Generator(device=z.device).manual_seed(int(seed))
        directions = F.normalize(torch.randn(z.shape[1], projections, device=z.device, generator=gen), dim=0)
        t = torch.linspace(0., t_max, points, device=z.device)
        gaussian_cf = torch.exp(-.5 * t * t)
        phase = (z @ directions).unsqueeze(-1) * t
        real, imag = phase.cos().mean(0), phase.sin().mean(0)
        err = ((real - gaussian_cf) ** 2 + imag ** 2) * gaussian_cf
        return 2 * z.shape[0] * torch.trapezoid(err, t, dim=-1).mean()


class TeacherQueue:
    """Ring FIFO of L2-normalised real teacher features (reference TeacherQueue): no parameters, immutable snapshot per step."""

    def __init__(self, features):
        self.features = F.normalize(features.float(), dim=-1).clone()
        self.pointer = 0

    def snapshot(self):
        return self.features.detach().clone()

    @torch.no_grad()
    def enqueue(self, x):
        x = F.normalize(x.detach().float(), dim=-1)
        assert torch.isfinite(x).all()
        n, cap = len(x), len(self.features)
        if n >= cap:
            self.features.copy_(x[-cap:])
            self.pointer = 0
            return
        p = self.pointer
        first = min(n, cap - p)
        self.features[p:p + first] = x[:first]
        if first < n:
            self.features[:n - first] = x[first:]
        self.pointer = (p + n) % cap


@torch.no_grad()
def enqueue_samples(Yt, valid, rimg, child_id, per_image=32, seed=0):
    """32 teacher child features per image, diverse parents and child offsets (same ordering rule as sample_latents)."""
    n = len(torch.unique(rimg[valid.any(1)])) * per_image
    return sample_latents(Yt, valid, rimg, child_id, budget=n, seed=seed)
