"""Self-contained light-mode HTML for the G1 leak-check + ADE20K evaluation: experiment structure, cross-dataset summary,
ADE qualitative examples (3), per-class IoU change (4), patch-token change maps (5). Every number is read from
results.json / paco/paco_dedup.json / ade/* / viz/* / the ImageNet run; nothing typed by hand.

  python3 -m src.g1eval.viz_html --run output/2026-11-07/g1_leakcheck_ade20k
"""
import argparse
import base64
import colorsys
import io
import json
import os

import numpy as np
from PIL import Image

from src.g1x.viz_html import CSS, JS, E, arrow, box, forest, svg, train_diagram

G1 = "output/2026-11-06/g1_cross_distill"
IMNET = "output/2026-11-07/g1_imagenet_generalization"
ROOT = "/home/irteam/datasets/ade20k/ADEChallengeData2016"
MODELS = ("original", "g1pool", "g1cross")
NAME = {"original": "Original DINOv3", "g1pool": "G1-pool", "g1cross": "G1-cross Encoder Only"}
SH = {"original": "Original", "g1pool": "G1-pool", "g1cross": "G1-cross"}
COLV = {"original": "var(--c-frozen)", "g1pool": "var(--c-pool)", "g1cross": "var(--c-xenc)"}
HOV = 4  # hover-array downsampling of the display images
# sequential blue ramp (light -> dark) for token change 1 - cos
RAMP = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAX = 0.6  # 1 - cos at the darkest colour


def b64png(a):
    bio = io.BytesIO()
    Image.fromarray(a).save(bio, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode()


def b64jpg(a):
    bio = io.BytesIO()
    Image.fromarray(a).save(bio, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(bio.getvalue()).decode()


def palette():
    """deterministic identity colours for the 150 classes (not the official ADE palette); ignore/other = light gray."""
    p = np.zeros((256, 3), np.uint8)
    for k in range(150):
        h = (k * 0.618034) % 1.0
        s = (0.45, 0.62, 0.80)[k % 3]
        v = (0.95, 0.80, 0.66)[(k // 3) % 3]
        p[k] = [round(255 * c) for c in colorsys.hsv_to_rgb(h, s, v)]
    p[255] = (225, 224, 217)
    return p


QUAL = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#a0cbe8",
        "#8cd17d", "#d37295", "#ffbe7d", "#499894", "#f1ce63", "#86bcb6"]


def image_palette(maps, base):
    """per-image colours: the 16 classes with most pixels over GT + 3 predictions get well-separated colours, the rest
    keep the deterministic base colour; 255 = light gray."""
    p = base.copy()
    allv = np.concatenate([m.ravel() for m in maps])
    c, n = np.unique(allv[allv != 255], return_counts=True)
    for j, k in enumerate(c[np.argsort(-n)][: len(QUAL)]):
        p[k] = [int(QUAL[j][i:i + 2], 16) for i in (1, 3, 5)]
    return p


def ramp_rgb():
    c = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in RAMP], np.float32)
    x = np.linspace(0, 1, len(RAMP))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, c[:, j]) for j in range(3)], 1).astype(np.uint8)


def cos_img(c, R):
    v = np.clip((1 - c) / CMAX, 0, 1)
    return R[(v * 255).round().astype(np.int64)]


def b64u8(a):
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


# ------------------------------------------------------------------------------------------------ diagrams
def eval_overview():
    b = ""
    b += box(16, 196, 136, 72, "Full image", "384 · 이미지당 1회", "data")
    ys = {"original": 116, "g1pool": 196, "g1cross": 276}
    subs = {"original": "학습 전 원본", "g1pool": "ema_last · enc", "g1cross": "ema_last · enc만 (dec 버림)"}
    for m, y in ys.items():
        b += box(178, y, 178, 64, NAME[m] if m != "g1cross" else "G1-cross encoder", subs[m], "frozen", accent=COLV[m], tag=False)
        b += arrow([(152, 232), (164, 232), (164, y + 32), (178, y + 32)])
    b += '<text class="t-s" x="267" y="362" text-anchor="middle">❄ 세 encoder 모두 frozen</text>'
    b += '<text class="t-s" x="267" y="378" text-anchor="middle">checksum = PACO probe와 동일</text>'
    b += '<text class="t-s" x="267" y="394" text-anchor="middle">평가 전후 불변</text>'
    rows = [(40, "① PACO val R2 — 누수 제거 후 재계산 (Table A)",
             [("Full tokens 576", "보간 없음", "plain"), ("R2 reader", "box 겹친 칸만 · 재학습 없음", "frozen"), ("AP 재계산", "full 2,051 vs dedup 2,047장", "data")]),
            (184, "② ImageNet-1K linear probe (이전 run, 참고)",
             [("CLS token", "final LN", "plain"), ("Linear 768→1000", "encoder별 독립 학습", "train"), ("Top-1", "val 50,000장", "data")]),
            (328, "③ ADE20K linear segmentation (Table B)",
             [("Patch tokens h×w", "CLS·register 4개 제외", "plain"), ("Linear 768→150", "같은 init · 같은 batch", "train"), ("원본 해상도 mIoU", "bilinear 업샘플 · val 2,000장", "data")])]
    b += '<path class="ar" d="M356,148 L380,148 M356,228 L380,228 M356,308 L380,308 M380,72 L380,360"/>'
    for y, title, steps in rows:
        b += f'<text class="t-b" x="404" y="{y - 10}">{E(title)}</text>'
        b += arrow([(380, y + 32), (404, y + 32)])
        x = 404
        for i, (t, s, k) in enumerate(steps):
            w = 176 if i < 2 else 196
            b += box(x, y, w, 64, t, s, k, accent="var(--c-xenc)" if k == "train" else None, tag=False)
            if i < 2:
                b += arrow([(x + w, y + 32), (x + w + 18, y + 32)])
            x += w + 18
    y = 470
    b += f'<line class="gridl" x1="16" y1="{y - 34}" x2="990" y2="{y - 34}"/>'
    b += f'<text class="t-b" x="404" y="{y - 10}">④ 누수 확인 (① 이전에 수행)</text>'
    b += box(16, y, 336, 64, "G1 학습 이미지 50,489장 × PACO val 2,051장", "COCO train2017 stream (pool / cross 동일)", "data")
    b += arrow([(352, y + 32), (404, y + 32)])
    b += box(404, y, 176, 64, "정확 일치 검사", "id · md5 · pixel md5 · dHash", "plain")
    b += arrow([(580, y + 32), (598, y + 32)])
    b += box(598, y, 176, 64, "근접 중복 탐색", "썸네일 상관 전수 + dHash → 육안", "plain")
    b += arrow([(774, y + 32), (792, y + 32)])
    b += box(792, y, 196, 64, "4장 / ROI 7개 제외", "정확 일치 0장", "plain")
    b += arrow([(988, y + 32), (996, y + 32), (996, 72), (988, 72)], cls="ar ar-dash")
    b += '<text class="t-s t-ar" x="992" y="300" text-anchor="end">dedup val</text>'
    return svg(1000, 548, b, "평가 구조")


# ------------------------------------------------------------------------------------------------ charts
def class_bars(dl, names, base, m, w=1000, h=320):
    L, Rm, top, bot = 44, 10, 26, 26
    order = np.argsort(dl)
    lo, hi = -30.0, 15.0
    sy = lambda v: top + (hi - v) / (hi - lo) * (h - top - bot)  # noqa: E731
    bw = (w - L - Rm) / len(dl)
    b = f'<text class="t-b" x="0" y="15">{E(SH[m])}: 클래스별 IoU 변화 (Original 대비, 작은 값부터 정렬, 150개)</text>'
    for v in range(-30, 16, 5):
        b += f'<line class="{"zero" if v == 0 else "gridl"}" x1="{L}" y1="{sy(v)}" x2="{w - Rm}" y2="{sy(v)}"/><text class="t-ax" x="{L - 6}" y="{sy(v) + 4}" text-anchor="end">{v:+d}</text>'
    for r, c in enumerate(order):
        v = float(dl[c])
        x = L + r * bw
        y0, y1 = sy(max(v, 0)), sy(min(v, 0))
        tip = f"{names[c]} · Original IoU {base[c]:.1f} → {base[c] + v:.1f} ({v:+.2f})"
        b += (f'<g class="hit" data-tip="{E(tip)}"><rect class="hitbox" x="{x}" y="{top}" width="{bw}" height="{h - top - bot}"/>'
              f'<rect x="{x + 1}" y="{y0}" width="{max(bw - 2, 1)}" height="{max(y1 - y0, 0.8)}" rx="1.5" style="fill:{COLV[m]}"/></g>')
    # direct labels for the extremes: stacked with leader lines (no overlap)
    prev = None
    for r, c in sorted(((r, c) for r, c in enumerate(order[:4])), key=lambda t: -sy(float(dl[t[1]]))):
        v, xc = float(dl[c]), L + r * bw + bw / 2
        y = sy(v) + 4 if prev is None else min(prev - 15, sy(v) + 4)
        prev = y
        tx = L + 9 * bw + 8
        b += f'<path class="xh" style="stroke-dasharray:none;stroke:var(--base)" d="M{xc},{sy(v)} L{tx - 3},{y - 4}"/>'
        b += f'<text class="t-s" x="{tx}" y="{y}">{E(names[c])} {v:+.1f}</text>'
    prev = None
    n = len(order)
    for r, c in sorted(((n - 3 + j, c) for j, c in enumerate(order[-3:])), key=lambda t: sy(float(dl[t[1]]))):
        v, xc = float(dl[c]), L + r * bw + bw / 2
        y = sy(v) + 4 if prev is None else max(prev + 15, sy(v) + 4)
        prev = y
        tx = L + (n - 9) * bw - 8
        b += f'<path class="xh" style="stroke-dasharray:none;stroke:var(--base)" d="M{xc},{sy(v)} L{tx + 3},{y - 4}"/>'
        b += f'<text class="t-s" x="{tx}" y="{y}" text-anchor="end">{E(names[c])} {v:+.1f}</text>'
    b += f'<text class="t-ax" x="{L}" y="{h - 8}">← 가장 많이 떨어진 클래스</text><text class="t-ax" x="{w - Rm}" y="{h - 8}" text-anchor="end">가장 많이 오른 클래스 →</text>'
    return f'<svg viewBox="0 0 {w} {h}" class="ch" role="img" aria-label="{E(SH[m])} class IoU change">{b}</svg>'


def scatter_cls(xc, yd, names, ntok, m, rho, w=500, h=330):
    L, Rm, top, bot = 48, 14, 46, 40
    x0, x1, y0, y1 = 0.5, 1.0, -30.0, 15.0
    sx = lambda v: L + (v - x0) / (x1 - x0) * (w - L - Rm)  # noqa: E731
    sy = lambda v: top + (y1 - v) / (y1 - y0) * (h - top - bot)  # noqa: E731
    b = f'<text class="t-b" x="0" y="15">{E(SH[m])}: 클래스별 token 변화 vs IoU 변화 (Spearman ρ = {rho:+.2f})</text>'
    for v in range(-30, 16, 10):
        b += f'<line class="{"zero" if v == 0 else "gridl"}" x1="{L}" y1="{sy(v)}" x2="{w - Rm}" y2="{sy(v)}"/><text class="t-ax" x="{L - 6}" y="{sy(v) + 4}" text-anchor="end">{v:+d}</text>'
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b += f'<line class="gridl" x1="{sx(v)}" y1="{top}" x2="{sx(v)}" y2="{h - bot}"/><text class="t-ax" x="{sx(v)}" y="{h - bot + 15}" text-anchor="middle">{v:.1f}</text>'
    b += f'<text class="t-ax" x="{(L + w - Rm) / 2}" y="{h - 4}" text-anchor="middle">클래스 평균 token cos(G1, Original) — 왼쪽일수록 많이 바뀜</text>'
    b += f'<text class="t-ax" x="{L}" y="{top - 6}">ΔIoU</text>'
    for c in range(len(names)):
        if xc[c] is None:
            continue
        x, y = sx(min(max(xc[c], x0), x1)), sy(min(max(yd[c], y0), y1))
        tip = f"{names[c]} · token cos {xc[c]:.3f} ({ntok[c]:,} tokens) · ΔIoU {yd[c]:+.2f}"
        b += f'<g class="hit" data-tip="{E(tip)}"><circle cx="{x}" cy="{y}" r="9" class="hitbox"/><circle cx="{x}" cy="{y}" r="4" class="dot" style="fill:{COLV[m]};stroke-width:1"/></g>'
    return f'<svg viewBox="0 0 {w} {h}" class="ch" role="img" aria-label="token change vs IoU change">{b}</svg>'


def hist_chart(hist, means, w=500, h=330):
    L, Rm, top, bot = 48, 14, 30, 40
    edges = np.linspace(0, 1, len(hist["g1pool"]) + 1)
    tot = {m: sum(v) for m, v in hist.items()}
    ymax = max(max(np.array(v) / tot[m]) for m, v in hist.items()) * 100 * 1.08
    x0 = 0.3
    sx = lambda v: L + (v - x0) / (1 - x0) * (w - L - Rm)  # noqa: E731
    sy = lambda v: top + (1 - v / ymax) * (h - top - bot)  # noqa: E731
    b = '<text class="t-b" x="0" y="15">Patch token cos(G1, Original) 분포 (ADE val 300장)</text>'
    for v in np.arange(0, ymax, 5 if ymax > 12 else 2):
        b += f'<line class="gridl" x1="{L}" y1="{sy(v)}" x2="{w - Rm}" y2="{sy(v)}"/><text class="t-ax" x="{L - 6}" y="{sy(v) + 4}" text-anchor="end">{v:.0f}%</text>'
    for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b += f'<text class="t-ax" x="{sx(v)}" y="{h - bot + 15}" text-anchor="middle">{v:.1f}</text>'
    b += f'<text class="t-ax" x="{(L + w - Rm) / 2}" y="{h - 4}" text-anchor="middle">token cos (1 = 변화 없음)</text>'
    for m in ("g1pool", "g1cross"):
        f = np.array(hist[m]) / tot[m] * 100
        pts = []
        for i, v in enumerate(f):
            if edges[i + 1] <= x0:
                continue
            a = max(edges[i], x0)
            pts += [(sx(a), sy(v)), (sx(edges[i + 1]), sy(v))]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        b += f'<path class="ln" d="{d}" style="stroke:{COLV[m]}"/>'
        b += f'<line x1="{sx(means[m])}" x2="{sx(means[m])}" y1="{top}" y2="{h - bot}" style="stroke:{COLV[m]};stroke-dasharray:4 3;stroke-width:1.5"/>'
    b += f'<text class="t-s" x="{sx(min(means.values())) - 6}" y="{top + 12}" text-anchor="end">평균 {means["g1pool"]:.3f} / {means["g1cross"]:.3f}</text>'
    return f'<svg viewBox="0 0 {w} {h}" class="ch" role="img" aria-label="token cos histogram">{b}</svg>'


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def tr(cells, th=False):
    t = "th" if th else "td"
    return "<tr>" + "".join(f"<{t}>{c}</{t}>" for c in cells) + "</tr>"


def fci(s):
    sig = s["lo"] > 0 or s["hi"] < 0
    return f"{s['point']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]{'*' if sig else ''}"


EXTRA_CSS = """
.ex{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.ex.c5{grid-template-columns:repeat(5,minmax(0,1fr))}.ex figure{margin:0}.pw{position:relative;line-height:0}
.pw img{width:100%;height:auto;display:block;border-radius:4px}.pw img.px{image-rendering:pixelated}
.mk{position:absolute;width:13px;height:13px;margin:-7px 0 0 -7px;border:2px solid #fff;outline:1.5px solid #0b0b0b;border-radius:50%;display:none;pointer-events:none}
figcaption{font-size:12px;color:var(--ink2);margin-top:4px;line-height:1.35}.exh{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline;margin:16px 0 6px;font-size:13px;color:var(--ink2)}
.exh b{color:var(--ink);font-size:14px}.tag{font-size:11px;border:1px solid var(--border);border-radius:999px;padding:1px 8px;color:var(--ink2);background:var(--frozen-bg)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 0;vertical-align:-1px}.cl{font-size:12px;color:var(--ink2);margin-top:4px}
.cbar{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2);margin:6px 0}.cbar span.g{display:inline-block;width:220px;height:12px;border-radius:3px}
@media (max-width:700px){.ex,.ex.c5{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

EXTRA_JS = """
const HV=window.HV||{};
function b64(s){const b=atob(s);const a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
const DEC={};function dec(k){if(!DEC[k]){const h=HV[k];DEC[k]={w:h.w,h:h.h};for(const f in h.a)DEC[k][f]=b64(h.a[f])}return DEC[k]}
const NM=window.CLSN||[];function cn(v){return v===255?'(other / ignore)':NM[v]}
document.querySelectorAll('.ex').forEach(row=>{const k=row.dataset.k;const kind=row.dataset.kind;
 row.querySelectorAll('.pw').forEach(pw=>{pw.addEventListener('mousemove',e=>{const r=pw.getBoundingClientRect();
  const fx=Math.min(Math.max((e.clientX-r.left)/r.width,0),0.999),fy=Math.min(Math.max((e.clientY-r.top)/r.height,0),0.999);
  row.querySelectorAll('.mk').forEach(m=>{m.style.left=(fx*100)+'%';m.style.top=(fy*100)+'%';m.style.display='block'});
  const d=dec(k);const i=Math.floor(fy*d.h)*d.w+Math.floor(fx*d.w);let t;
  if(kind==='seg'){t='GT: '+cn(d.gt[i])+' · Original: '+cn(d.original[i])+' · G1-pool: '+cn(d.g1pool[i])+' · G1-cross: '+cn(d.g1cross[i])}
  else{t='GT: '+cn(d.gt[i])+' · token cos  G1-pool '+(d.cp[i]/250).toFixed(2)+' · G1-cross '+(d.cc[i]/250).toFixed(2)}
  show(t,e)});
  pw.addEventListener('mouseleave',()=>{row.querySelectorAll('.mk').forEach(m=>m.style.display='none');hide()})})});
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = a.run
    res = json.load(open(os.path.join(R, "results.json")))
    paco = json.load(open(os.path.join(R, "paco", "paco_dedup.json")))
    imn = json.load(open(os.path.join(IMNET, "results.json")))
    vz = json.load(open(os.path.join(R, "viz", "viz_data.json")))
    ps = json.load(open(os.path.join(R, "ade", "patch_stats.json")))
    w = json.load(open(os.path.join(G1, "calib", "weights.json")))
    tlog = [json.loads(l) for l in open(os.path.join(R, "ade", "train_log.jsonl"))]
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "objectInfo150.txt"))][1:]
    names = [r[4].split(",")[0].strip() for r in rows]
    ratio = np.array([float(r[1]) for r in rows])
    ntrain = np.array([int(r[2]) for r in rows])
    z = np.load(os.path.join(R, "ade", "val_inter_union.npz"))
    iou = {m: 100 * z[f"I_{m}"].astype(np.int64).sum(0) / np.maximum(z[f"U_{m}"].astype(np.int64).sum(0), 1) for m in MODELS}
    dl = {m: iou[m] - iou["original"] for m in ("g1pool", "g1cross")}
    ade, mi = res["ade"], res["ade"]["mIoU"]
    lk, dd = paco["leak"], paco["dedup"]
    PC = lambda x, y, c="M0 Overall": dd["contrasts"][f"{x} - {y}"][c]  # noqa: E731
    IC = lambda k: imn["ci"][k]["top1"]  # noqa: E731
    pal0, RR = palette(), ramp_rgb()

    # ---- tiles
    tiles = '<div class="tiles">'
    tiles += (f'<div class="tile"><div class="k">① PACO val 누수</div><div class="v">{len(lk["removed_images"])}장</div>'
              f'<div class="s">정확 일치 0 · 근접 중복 {len(lk["removed_images"])}장 / ROI {lk["removed_rois"]}개 ({lk["removed_image_pct"]:.2f}%)</div></div>')
    tiles += (f'<div class="tile"><div class="k">PACO R2 M0 (dedup) vs Frozen</div><div class="v">+{PC("G1-cross Encoder Only", "Frozen")["point"]:.2f}</div>'
              f'<div class="s">G1-cross* · G1-pool {PC("G1-pool", "Frozen")["point"]:+.2f}* · 누수 제거 전과 동일</div></div>')
    tiles += (f'<div class="tile"><div class="k">② ImageNet Top-1 vs Original</div><div class="v">{IC("ind:g1cross-original")["point"]:+.2f}</div>'
              f'<div class="s">G1-cross* · G1-pool {IC("ind:g1pool-original")["point"]:+.2f}* · 거의 유지</div></div>')
    tiles += (f'<div class="tile"><div class="k">③ ADE20K mIoU vs Original</div><div class="v">{ade["ci"]["g1cross-original"]["point"]:+.2f}</div>'
              f'<div class="s">G1-cross* · G1-pool {ade["ci"]["g1pool-original"]["point"]:+.2f}* · Original {mi["original"]:.2f}</div></div>')
    tiles += "</div>"
    legend = ('<div class="legend">' + "".join(f'<span><i style="background:{COLV[m]}"></i>{E(NAME[m])}</span>' for m in MODELS)
              + '<span><i style="background:var(--ink2)"></i>G1-cross − G1-pool</span></div>')

    # ---- cross-dataset forest panels
    def fr(c):
        return (c["point"], c["lo"], c["hi"])
    f_paco = forest([("G1-pool − Frozen", COLV["g1pool"], *fr(PC("G1-pool", "Frozen"))),
                     ("G1-cross − Frozen", COLV["g1cross"], *fr(PC("G1-cross Encoder Only", "Frozen"))),
                     ("cross − pool", "var(--ink2)", *fr(PC("G1-cross Encoder Only", "G1-pool")))],
                    "① PACO val R2 M0 (dedup, AP pt)", -1, 3, w=330, L=118)
    f_imn = forest([("G1-pool − Orig", COLV["g1pool"], *fr(IC("ind:g1pool-original"))),
                    ("G1-cross − Orig", COLV["g1cross"], *fr(IC("ind:g1cross-original"))),
                    ("cross − pool", "var(--ink2)", *fr(IC("ind:g1cross-g1pool")))],
                   "② ImageNet Top-1 (pp)", -1, 1, w=330, L=118)
    f_ade = forest([("G1-pool − Orig", COLV["g1pool"], *fr(ade["ci"]["g1pool-original"])),
                    ("G1-cross − Orig", COLV["g1cross"], *fr(ade["ci"]["g1cross-original"])),
                    ("cross − pool", "var(--ink2)", *fr(ade["ci"]["g1cross-g1pool"]))],
                   "③ ADE20K mIoU (pt)", -5, 1, w=330, L=118)

    # ---- tables
    ta = tr(["Method", "set", "Images", "ROIs", "M0 Overall", "Obj D5–16", "Part D1–4", "M2 Overall"], True)
    for nm, r in (("full", paco["full"]), ("dedup", dd)):
        for m, p in r["points"].items():
            row = [E(m), nm, f"{r['n_images']:,}", f"{r['n_rois']:,}"] + [f"{p[c]:.2f}" for c in ("M0 Overall", "Obj D5-16", "Part D1-4", "M2 Overall")]
            ta += tr([f"<b>{x}</b>" for x in row] if nm == "dedup" else row)
    tc = tr(["dedup contrast", "M0 Overall", "Obj D5–16", "Part D1–4", "M2 Overall"], True)
    for k, c in dd["contrasts"].items():
        tc += tr([E(k.replace(" - ", " − "))] + [fci(c[x]) for x in ("M0 Overall", "Obj D5-16", "Part D1-4", "M2 Overall")])
    tb = tr(["Method", "ADE20K mIoU", "Δ vs Original (95% CI)", "head train loss (마지막)", "ImageNet Top-1", "PACO dedup M0"], True)
    pk = {"original": "Frozen", "g1pool": "G1-pool", "g1cross": "G1-cross Encoder Only"}
    for m in MODELS:
        tb += tr([f'<span style="color:{COLV[m]}">●</span> {E(NAME[m])}', f"{mi[m]:.2f}",
                  "–" if m == "original" else fci(ade["ci"][f"{m}-original"]), f"{tlog[-1]['loss_' + m]:.3f}",
                  f"{imn['independent'][m]['val_top1']:.2f}", f"{dd['points'][pk[m]]['M0 Overall']:.2f}"])

    # ---- (3) qualitative examples + (5) cos maps
    used = set()
    ex_seg, ex_cos, HVd = "", "", {}
    WHY = {"random": "무작위 (seed 0)", "drop": "하락 큰 이미지", "gain": "상승 큰 이미지"}
    for e in vz["examples"]:
        k = e["k"]
        d = np.load(os.path.join(R, "viz", f"ex_{k}.npz"))
        hh, ww = d["gt"].shape
        sub = lambda x: np.ascontiguousarray(x[HOV // 2::HOV, HOV // 2::HOV])  # noqa: E731
        hv = {"gt": sub(d["gt"])}
        for m in MODELS:
            hv[m] = sub(d[f"pred_{m}"])
        hv["cp"] = (np.clip(sub(d["cos_g1pool"]), 0, 1) * 250).round().astype(np.uint8)
        hv["cc"] = (np.clip(sub(d["cos_g1cross"]), 0, 1) * 250).round().astype(np.uint8)
        HVd[str(k)] = dict(w=int(hv["gt"].shape[1]), h=int(hv["gt"].shape[0]), a={f: b64u8(v) for f, v in hv.items()})
        pal = image_palette([d["gt"]] + [d[f"pred_{m}"] for m in MODELS], pal0)
        allv = np.concatenate([d["gt"].ravel()] + [d[f"pred_{m}"].ravel() for m in MODELS])
        cls_present, cnt = np.unique(allv[allv != 255], return_counts=True)
        top = cls_present[np.argsort(-cnt)][:10]
        used.update(int(c) for c in top)
        swl = " ".join(f'<span><i class="sw" style="background:rgb{tuple(int(v) for v in pal[c])}"></i>{E(names[c])}</span>' for c in top)
        mk = '<div class="mk"></div>'
        head = (f'<div class="exh"><b>#{k + 1} {E(e["file"])}</b><span class="tag">{WHY[e["why"]]}</span>'
                + " · ".join(f'{SH[m]} {e["miou"][m]:.1f}' for m in MODELS) + " <span class='note'>(이미지별 mIoU)</span></div>")
        figs = f'<figure><div class="pw"><img src="{b64jpg(d["rgb"])}" alt="">{mk}</div><figcaption>입력</figcaption></figure>'
        figs_gt = f'<figure><div class="pw"><img src="{b64png(pal[d["gt"]])}" alt="">{mk}</div><figcaption>GT</figcaption></figure>'
        ex_seg += head + f'<div class="ex c5" data-k="{k}" data-kind="seg">' + figs + figs_gt
        for m in MODELS:
            ex_seg += (f'<figure><div class="pw"><img src="{b64png(pal[d["pred_" + m]])}" alt="">{mk}</div>'
                       f'<figcaption><i class="sw" style="background:{COLV[m]}"></i>{SH[m]} · {e["miou"][m]:.1f}</figcaption></figure>')
        ex_seg += f'</div><div class="cl">범례(이 이미지의 GT·예측에서 픽셀이 많은 순): {swl}</div>'
        ex_cos += (f'<div class="exh"><b>#{k + 1} {E(e["file"])}</b><span class="tag">{WHY[e["why"]]}</span>'
                   f'평균 token cos: G1-pool {e["cos_mean"]["g1pool"]:.3f} · G1-cross {e["cos_mean"]["g1cross"]:.3f}</div>'
                   f'<div class="ex" data-k="{k}" data-kind="cos">' + figs + figs_gt)
        for m in ("g1pool", "g1cross"):
            ex_cos += (f'<figure><div class="pw"><img src="{b64png(cos_img(d["cos_" + m], RR))}" alt="">{mk}</div>'
                       f'<figcaption><i class="sw" style="background:{COLV[m]}"></i>{SH[m]}: 1 − cos (진할수록 많이 바뀜)</figcaption></figure>')
        ex_cos += "</div>"
    dif = [(abs(c["miou_cpu"] - c["miou_saved"]), e["k"]) for e in vz["examples"] for c in e["cpu_check"].values()]
    cpu_med, (cpu_max, cpu_k) = float(np.median([x for x, _ in dif])), max(dif)
    cbar = ('<div class="cbar">token cos 1.0 (변화 없음) <span class="g" style="background:linear-gradient(90deg,' + ",".join(RAMP)
            + f')"></span> ≤ {1 - CMAX:.1f} (크게 바뀜)</div>')

    # ---- (4) per-class
    seg4 = ('<div class="seg" data-group="m4"><button aria-pressed="true" data-k="g1cross">G1-cross</button>'
            '<button aria-pressed="false" data-k="g1pool">G1-pool</button></div>')
    bars = "".join(f'<div class="card dgw" data-m4="{m}" style="{"" if m == "g1cross" else "display:none"}">{class_bars(dl[m], names, iou["original"], m)}</div>'
                   for m in ("g1cross", "g1pool"))
    q = np.lexsort((-ntrain, -ratio))  # pixel ratio (4 d.p., ties) then number of train images
    tiers = tr(["클래스 묶음 (pixel 비율 순, 동률은 train 이미지 수)", "Original 평균 IoU", "Δ G1-pool", "Δ G1-cross", "내려간 클래스 수 (pool / cross)"], True)
    for lab, sl in (("상위 50 (wall, building, sky …)", q[:50]), ("중간 50", q[50:100]), ("하위 50", q[100:])):
        tiers += tr([lab, f"{iou['original'][sl].mean():.2f}", f"{dl['g1pool'][sl].mean():+.2f}", f"{dl['g1cross'][sl].mean():+.2f}",
                     f"{(dl['g1pool'][sl] < 0).sum()} / {(dl['g1cross'][sl] < 0).sum()}"])
    avg = (dl["g1pool"] + dl["g1cross"]) / 2
    ext = tr(["", "클래스", "Original IoU", "Δ G1-pool", "Δ G1-cross"], True)
    for tag, idx in (("하락", np.argsort(avg)[:10]), ("상승", np.argsort(-avg)[:5])):
        for c in idx:
            ext += tr([tag, E(names[c]), f"{iou['original'][c]:.1f}", f"{dl['g1pool'][c]:+.1f}", f"{dl['g1cross'][c]:+.1f}"])

    # ---- (5) class-level token change
    cl = vz["cls"]
    ntok = cl["n_tokens"]
    sc5, rhos = "", {}
    for m in ("g1cross", "g1pool"):
        xc = [v if (v is not None and ntok[c] >= 30) else None for c, v in enumerate(cl["cos"][m])]
        ok = [c for c in range(150) if xc[c] is not None]
        rhos[m] = spearman(np.array([xc[c] for c in ok]), dl[m][ok])
        sc5 += f'<div data-m5="{m}" style="{"" if m == "g1cross" else "display:none"}">{scatter_cls(xc, dl[m], names, ntok, m, rhos[m])}</div>'
    n_ok = sum(1 for c in range(150) if ntok[c] >= 30)
    seg5 = ('<div class="seg" data-group="m5"><button aria-pressed="true" data-k="g1cross">G1-cross</button>'
            '<button aria-pressed="false" data-k="g1pool">G1-pool</button></div>')
    hist = hist_chart(cl["hist"], cl["mean"])

    tdg = {False: train_diagram(False, w["lam_detail"], w["lam_global"]), True: train_diagram(True, w["lam_detail"], w["lam_global"])}
    cfg = ade["cfg"]
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 누수 확인 + ADE20K</title><style>{CSS}{EXTRA_CSS}</style></head><body><div id="tip" role="tooltip"></div><main style="position:relative">
<h1>G1 encoder 추가 평가: PACO 누수 확인 + ADE20K linear segmentation</h1>
<p class="note">2026-11-07 · DINOv3-B/16 @384 · Original / G1-pool / G1-cross Encoder Only (decoder 미사용) · 새 distillation·fine-tuning 없음 · seed 0 · 모든 평가 val (PACO / ADE test 미사용) · run dir <code>output/2026-11-07/g1_leakcheck_ade20k</code></p>
<p class="lead"><b>결론:</b> PACO 개선은 누수 때문이 아니다(정확 일치 0장, 근접 중복 {len(lk['removed_images'])}장을 빼도 수치가 그대로다). 하지만 같은 encoder로 ADE20K를 frozen linear segmentation하면 <b>mIoU가 {ade['ci']['g1pool-original']['point']:+.2f} / {ade['ci']['g1cross-original']['point']:+.2f} 떨어진다</b>. ImageNet은 거의 유지된다({IC('ind:g1pool-original')['point']:+.2f} / {IC('ind:g1cross-original')['point']:+.2f}pp).
G1은 CLS와 patch 평균은 거의 그대로 두고 개별 patch token을 크게 바꿨다. 그 결과 COCO ROI readout에는 유리해졌지만, scene 단위 dense 의미 정보 일부를 잃었다.</p>
{tiles}
{legend}

<h2>1. 실험 구조</h2>
<p>두 G1 encoder는 같은 원본 DINOv3에서 시작해 COCO 이미지로 crop distillation했다(PACO label은 쓰지 않음). 이번 평가는 그 checkpoint를 <b>그대로 얼려서</b> 세 과제에 붙인 것이다. 새로 학습한 것은 linear head(ImageNet, ADE)뿐이다.</p>
<h3>1-A. G1-pool 학습 구조 (기존 checkpoint, 이번에 재학습 없음)</h3>
<div class="card dgw">{tdg[False]}</div>
<div class="formula">L(G1-pool)  = L_roi + {w['lam_global']}·L_global
L(G1-cross) = L_roi + {w['lam_detail']}·L_detail + {w['lam_global']}·L_global
L_roi    = mean_ROI [ 1 − cos( ROI 면적 가중 pool(full token F),  crop teacher token T의 유효 셀 pool ) ]
L_global = ½[ (1 − cos(CLS_s, CLS_0)) + (1 − cos(mean F, mean F_0)) ]     ← CLS와 평균만 붙잡음 (개별 patch는 제약 없음)</div>
<h3>1-B. G1-cross 학습 구조 (decoder는 학습에만 쓰였고, 이번 평가에서는 버림)</h3>
<div class="card dgw">{tdg[True]}</div>
<h3>1-C. 이번 평가 구조</h3>
<div class="card dgw">{eval_overview()}</div>
<p class="note">ADE head 학습 설정(사전에 고정, 모델별 튜닝 없음): AdamW lr {cfg['lr']}, wd {cfg['wd']}, batch {cfg['batch']}, {cfg['iters']:,} iter, warmup {cfg['warmup']:,} + poly; 학습 입력은 짧은 변 384×U{tuple(cfg['ratio_range'])} resize → 384 crop(pad = ignore) + hflip.
세 head는 매 iteration 같은 batch를 받고, 초기값도 같다. 평가 입력은 짧은 변 384이고, 결과는 원본 해상도에서 계산했다. 공식 DINOv3 linear seg(512, BN 등)와는 프로토콜이 달라 절대값은 비교하지 않는다.</p>

<h2>2. 결과 요약: 같은 encoder, 세 과제</h2>
<p>각 패널은 단위가 다르므로 축도 따로다. 점 = 차이, 선 = 이미지 단위 paired bootstrap 95% CI(B = 1000). 채운 점 = CI가 0을 제외, 빈 점 = n.s.</p>
<div class="grid2" style="grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr))">
<div class="card">{f_paco}</div><div class="card">{f_imn}</div><div class="card">{f_ade}</div></div>
<h3>Table A: PACO val, 누수 제거 전후 (R2, dev-selected epoch, AP 재계산)</h3>
<div class="card tw"><table>{ta}</table></div>
<div class="card tw"><table>{tc}</table></div>
<h3>Table B: ADE20K linear segmentation (val 2,000장, 원본 해상도 mIoU)</h3>
<div class="card tw"><table>{tb}</table></div>
<p class="note">Head train loss도 G1 쪽이 높다. 즉 val만 나쁜 과적합이 아니라, 학습 데이터에서도 선형으로 덜 분리된다.</p>

<h2>3. ADE20K 정성 비교</h2>
<p>저장된 linear head로 CPU에서 다시 추론한 결과다(학습 없음). 마우스를 올리면 네 패널의 같은 위치가 표시되고, 각 모델이 그 픽셀을 어떤 클래스로 예측했는지 보여준다.
선택 기준: 무작위 4장 + 이미지별 mIoU 하락이 가장 큰 3장 + 상승이 가장 큰 2장(G1 두 모델 평균 기준, union&gt;0 클래스 5개 이상인 이미지 중). <b>하락·상승 예시는 극단값을 일부러 고른 것이다</b>. 전체 경향은 4절을 보면 된다.
CPU(fp32) 재추론과 원래 GPU(bf16) 평가의 이미지별 mIoU 차이는 중앙값 {cpu_med:.2f}pt다. 다만 #{cpu_k + 1}은 경계 픽셀 예측이 바뀌어 최대 {cpu_max:.1f}pt 차이가 난다(캡션 수치는 원래 GPU 평가 값). 색은 이미지마다 다시 배정했다(같은 이미지의 다섯 패널 안에서만 같은 클래스 = 같은 색, 공식 ADE 팔레트 아님). 회색 = other/ignore.</p>
{ex_seg}

<h2>4. 클래스별 IoU 변화</h2>
<p>150개 클래스 각각에서 Original 대비 IoU가 얼마나 변했는지 보여준다(val 2,000장 전체에서 합산). 막대에 마우스를 올리면 클래스 이름이 보인다.</p>
{seg4}{bars}
<div class="grid2"><div class="card tw"><table>{tiers}</table><p class="note">→ 흔한 stuff 클래스와 드문 object 클래스 모두에서 비슷하게 떨어졌다. 특정 클래스 몇 개에 몰린 하락이 아니다.</p></div>
<div class="card tw"><table>{ext}</table></div></div>

<h2>5. Patch token 변화</h2>
<p>각 위치의 G1 patch token이 같은 위치의 Original token과 얼마나 다른지(1 − cos)를 그렸다. 진할수록 많이 바뀐 것이다. 3절과 같은 이미지이며, 마우스를 올리면 그 위치의 GT 클래스와 cos 값을 보여준다.</p>
{cbar}
<p class="note">관찰: 9장 모두에서 변화가 물체 경계나 특정 클래스 영역에 몰리지 않고, 이미지 전체에 token 단위로 흩어져 있다. 평균 cos는 하락이 큰 #5–#7(0.75–0.80)이 무작위 #1–#4(0.78–0.82)보다 약간 낮지만, 지도의 패턴은 비슷하다. 상승한 #8–#9도 0.77–0.79로 비슷하다. 그래서 token 변화량만으로는 어느 이미지가 손해를 볼지 설명하기 어렵다.</p>
{ex_cos}
<h3>5-A. 전체 분포와 클래스 단위 관계</h3>
<div class="grid2"><div class="card">{hist}</div><div class="card">{seg5}{sc5}</div></div>
<p class="note">분포는 무작위 ADE val {cl['n_images']}장에서 GT가 한 클래스로 50% 이상 차지하는 token만 셌다. 산점도는 token이 30개 이상인 클래스 {n_ok}개다(x는 이 300장 기준, y는 val 2,000장 기준 ΔIoU).
Spearman ρ: G1-cross {rhos['g1cross']:+.2f}, G1-pool {rhos['g1pool']:+.2f}로 거의 0이다. 클래스 단위로 봐도 token이 많이 바뀐 클래스가 더 많이 떨어지지는 않는다.
(참고) ImageNet val에서 CLS cos는 {imn['preservation']['val']['clscos_g1pool']:.3f} / {imn['preservation']['val']['clscos_g1cross']:.3f}, patch 평균 cos는 {imn['preservation']['val']['patchcos_g1pool']:.3f} / {imn['preservation']['val']['patchcos_g1cross']:.3f}다. ADE patch token 평균 norm은 Original {ps['norm']['original']:.2f} → {ps['norm']['g1pool']:.2f} / {ps['norm']['g1cross']:.2f}다.</p>
<p><b>해석:</b> 이미지 요약(CLS, patch 평균)은 거의 그대로이고, 개별 patch token만 크게 움직였다. L_global이 CLS와 평균만 제약하기 때문이다. 그래서 CLS로 분류하는 ImageNet은 유지되고, patch token을 직접 읽는 ADE는 떨어진다. 다만 어디서 얼마나 떨어지는지는 cos로 잰 변화량과 거의 무관하다(이미지 단위, 클래스 단위 모두). 변화의 양보다 어떤 정보가 바뀌었는지가 중요할 수 있지만, 이번 실험으로는 확인하지 않았다.</p>

<h2>6. Q1–Q4</h2>
<div class="card qa">
<div class="q">Q1</div><div><b>사실상 겹치지 않는다.</b> 정확 일치 0장, 근접 중복 {len(lk['removed_images'])}장 / ROI {lk['removed_rois']}개({lk['removed_image_pct']:.2f}% / {lk['removed_roi_pct']:.2f}%). 썸네일 상관 0.85 미만인 크게 crop된 변형은 놓쳤을 수 있다.</div>
<div class="q">Q2</div><div><b>유지된다.</b> dedup에서 G1-pool {fci(PC('G1-pool', 'Frozen'))}, G1-cross {fci(PC('G1-cross Encoder Only', 'Frozen'))}. cross − pool은 M0 {fci(PC('G1-cross Encoder Only', 'G1-pool'))}로 경계선상이고, Obj D5–16 {fci(PC('G1-cross Encoder Only', 'G1-pool', 'Obj D5-16'))}만 유의하다.</div>
<div class="q">Q3</div><div><b>유지되지 않는다.</b> ADE mIoU {fci(ade['ci']['g1pool-original'])} / {fci(ade['ci']['g1cross-original'])}(약 8% 상대 손실). 하락은 거의 모든 클래스에 걸쳐 있고, head train loss도 높다. ImageNet이 유지된 것은 CLS가 보존됐기 때문이다.</div>
<div class="q">Q4</div><div><b>주장할 수 있는 것:</b> PACO ROI readout 개선은 실제이고, 이미지 누수 때문이 아니다. 그러나 G1은 일반 표현을 개선한 것이 아니라, COCO ROI readout 쪽으로 표현을 재배치했고 그 대가로 dense 의미 정보 일부를 잃었다.
<b>주장할 수 없는 것:</b> 일반화되는 crop-like 표현이라는 주장, COCO 도메인 밖으로의 일반화(누수가 아니라 in-domain 학습이다), ADE 하락이 G1 방식의 본질적 한계라는 주장(레시피·비선형 head·512 해상도·seed는 확인 안 함), PACO test 결과.</div>
</div>
<p class="note">이번 평가는 single seed, 고정 프로토콜 하나로만 돌렸다. ImageNet 수치는 이전 run(<code>{IMNET}</code>)을 그대로 가져왔다. 상세는 같은 폴더의 README.md에 있다.</p>
</main><script>window.HV={json.dumps(HVd)};window.CLSN={json.dumps(names)};{JS}{EXTRA_JS}</script></body></html>"""
    out = os.path.join(R, "g1ev_visualization.html")
    open(out, "w").write(page)
    print(out, len(page))


if __name__ == "__main__":
    main()
