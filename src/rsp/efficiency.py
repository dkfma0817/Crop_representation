"""Inference cost of RSP-Masked / noPix vs re-encoding (GPU 1, bf16 autocast, batch = 1 image, CUDA-event latency median of 50 after 10
warm-up runs; FLOPs from torch.utils.flop_counter = 2 x MACs of matmul / conv / SDPA).

Components:
  per image : frozen DINOv3 384 (581 tokens) | predictor context stream (L0)
  per ROI   : evidence conv (pix) | L1 stream (65 tokens pix / 1 noPix) | queries (n fine cells; 3x3 parent ROI = 144, 6x6 = 576, 12x12 = 2,304)
  references: F768 full re-encoding (2,309 tokens / image) | C0 crop re-encoding (one 384 forward per ROI) | dense teacher used directly
              (coarse once + the ROI's fine stream)
Scenarios: 1 image + 1 ROI and 1 image + 8 ROIs (each 3x3 / 6x6 parents).

  python -m src.rsp.efficiency --pix <model.pt> --nopix <model.pt> --out <json>
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.rsp.common import PT, fine_rect_index
from src.rsp.encoder import load_rsp
from src.rsp.teacher import DenseTeacher
from src.tsplit.features import frozen_vit


def timeit(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e9


def measure(fn):
    return dict(ms=round(timeit(fn), 3), gflops=round(flops(fn), 3))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pix", required=True)
    ap.add_argument("--nopix", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    vit = frozen_vit(dev)
    teach = DenseTeacher(vit).to(dev)
    x = torch.randn(1, 3, 384, 384, device=dev)
    x768 = torch.randn(1, 3, 768, 768, device=dev)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    out = dict(setup="H200 GPU 1, bf16 autocast, batch 1 image, median of 50 runs after 10 warm-up; FLOPs = torch flop counter (2 x MAC)")
    with ac():
        final = vit.forward_features(x).float()

    def dino384():
        with ac():
            vit.forward_features(x)

    def dino768():
        with ac():
            vit.forward_features(x768)
    out["DINOv3_384_per_image"] = measure(dino384)
    out["F768_reencode_per_image"] = measure(dino768)
    out["C0_crop_reencode_per_ROI"] = out["DINOv3_384_per_image"]
    for name, path in (("RSP-Masked", a.pix), ("RSP-Masked-noPix", a.nopix)):
        m, mu, sd, ck = load_rsp(path, dev)
        res = dict(trainable_params=sum(p.numel() for p in m.parameters()))

        def ctx():
            with ac():
                return m.context(final)
        res["context_per_image"] = measure(ctx)
        cache = ctx()
        for nroi in (1, 8):
            for side in (3, 6, 12):
                box = torch.tensor([[64., 64., 64. + 16 * side, 64. + 16 * side]], device=dev).repeat(nroi, 1)
                ev = torch.randn(nroi, 3, 64, 64, device=dev) if m.pix else None
                rimg = torch.zeros(nroi, dtype=torch.long, device=dev)
                fr = torch.tensor([[16, 16, 4 * side, 4 * side]], device=dev).repeat(nroi, 1)
                _, uy, ux, _ = fine_rect_index(fr)
                cen = torch.stack([(uy.float() + .5) * PT, (ux.float() + .5) * PT], -1)
                key = f"{nroi}roi_{side}x{side}parents_{(4 * side) ** 2}queries"

                def l1():
                    with ac():
                        return m.level1(cache, rimg, box, ev)
                l1o, z = l1()

                def qs():
                    with ac():
                        m.queries(cache, l1o, rimg, z, cen)

                def total():
                    with ac():
                        f = vit.forward_features(x).float()
                        c = m.context(f)
                        l, zz = m.level1(c, rimg, box, ev)
                        m.queries(c, l, rimg, zz, cen)
                r = dict(L1_stream=measure(l1), queries=measure(qs), total_incl_dino384=measure(total))
                if m.pix and nroi == 1 and side == 3:
                    def evc():
                        with ac():
                            m.ev_embed(ev)
                    res["evidence_conv_per_ROI"] = measure(evc)
                res[key] = r
        out[name] = res
    for side in (3, 6, 12):
        fr = torch.tensor([[16, 16, 4 * side, 4 * side]], device=dev)
        content = torch.tensor([[0., 0., 384., 384.]], device=dev)

        def dt():
            with ac():
                fl, kv = teach.coarse(x)
                teach.fine(teach.fine_embed(x), kv, fr, torch.zeros(1, dtype=torch.long, device=dev), content)
        out[f"dense_teacher_direct_1roi_{side}x{side}parents"] = measure(dt)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
