"""M1 / M2 inference cost, same convention as src.siglip_pm.efficiency (2 FLOPs/MAC). Detail MACs are counted from the actual modules with forward hooks at a 640 x 640 input."""
import argparse
import json
from pathlib import Path
import statistics
import time

from PIL import Image
import pandas as pd
import torch
from torch import nn

from src.siglip_cue.data import LB, native_canvas, native_geo
from src.siglip_cue.model import load_m2
from src.siglip_pm.efficiency import costs
from src.siglip_pm.model import load_hcf, n_params


def detail_macs(detail, dev):
    total, hooks = [0], []

    def hook(module, _, output):
        k = module.kernel_size[0] * module.kernel_size[1]
        total[0] += k * module.in_channels * module.out_channels * output.shape[2] * output.shape[3] // module.groups

    for m in detail.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    geo = torch.tensor([[640, 480, 384, 288, 0, 48]], dtype=torch.float32, device=dev)
    with torch.no_grad():
        detail(torch.zeros(1, 3, 640, 640, device=dev), geo)
    for h in hooks:
        h.remove()
    return total[0]


def latency(fn, n=60):
    times = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            if i >= 10:
                times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    run = Path(a.run)
    dev = torch.device('cuda')
    row = pd.read_parquet('output/2026-11-07/paco_oracle_attr/data/manifest_val.parquet').iloc[0]
    img = Image.open(row.path).convert('RGB')
    import numpy as np
    x = LB(*img.size, 384, 16).canvas(img)[None].to(dev)
    native = native_canvas(np.asarray(img))[None].to(dev)
    geo = native_geo(*img.size)[None].to(dev)
    out = []
    for c in ('M1', 'M2'):
        path = run / c / 'checkpoints/ema_last.pt'
        model = load_m2(path, dev)[0] if c == 'M2' else load_hcf(path, dev)[0]
        torch.cuda.reset_peak_memory_stats()
        fn = (lambda: model.forward_all(x, native=native, geo=geo)) if c == 'M2' else (lambda: model.forward_all(x))
        ms = latency(fn)
        base = costs('P')
        r = dict(model=c, parameters=n_params(model), stored_tokens_coarse=576, stored_tokens_fine=2304, stored_scalars=24 * 24 * 768 + 48 * 48 * 384, feature_MiB_fp32=(24 * 24 * 768 + 48 * 48 * 384) * 4 / 2 ** 20, feature_MiB_fp16=(24 * 24 * 768 + 48 * 48 * 384) * 2 / 2 ** 20, median_encode_ms=ms, peak_inference_vram_gib_1image=torch.cuda.max_memory_allocated() / 2 ** 30, encoder_GFLOPs_p8_part=base['encoder_GFLOPs'], ROI_fusion_GFLOPs_per_ROI=base['ROI_fusion_GFLOPs_per_ROI'], flop_convention=base['flop_convention'])
        if c == 'M2':
            macs = detail_macs(model.detail, dev)
            r.update(detail_parameters=n_params(model.detail), detail_GFLOPs_640=2 * macs / 1e9, detail_extra_stored_tokens=0, detail_transient_map='80x80x128')
        r['total_GFLOPs'] = r['encoder_GFLOPs_p8_part'] + r.get('detail_GFLOPs_640', 0.)
        done = json.loads((run / c / 'done.json').read_text())
        r.update(train_peak_vram_gib=done['peak_gib'], train_time_s=done['time_s'])
        out.append(r)
        del model
        torch.cuda.empty_cache()
    (run / 'results').mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(run / 'results/efficiency.csv', index=False)
    (run / 'efficiency.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
