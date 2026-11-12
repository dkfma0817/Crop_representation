"""Self-contained HTML visualisation of the G1-pool / G1-cross pilot: architecture + loss diagrams per method and result
charts (all numbers read from results.json / diag.jsonl / train_log.jsonl / calib/weights.json; nothing typed by hand).

  python3 -m src.g1x.viz_html --run output/2026-11-06/g1_cross_distill
"""
import argparse
import html
import json
import os

E = html.escape

# method -> (css colour var, short label)
METH = [("Frozen Full", "frozen", "Frozen Full"), ("G1-pool", "pool", "G1-pool"),
        ("G1-cross Encoder Only", "xenc", "G1-cross Enc"), ("G1-cross + Decoder", "xdec", "G1-cross + Dec"),
        ("C0 Crop Oracle", "oracle", "C0 Oracle")]
COL = {m: f"var(--c-{k})" for m, k, _ in METH}
SHORT = {m: s for m, _, s in METH}


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ------------------------------------------------------------------------------------------------ svg primitives
def box(x, y, w, h, title, sub="", kind="plain", accent=None, sub2="", tag=True):
    cls = {"train": "bx bx-train", "frozen": "bx bx-frozen", "plain": "bx", "data": "bx bx-data", "ghost": "bx bx-ghost",
           "dec": "bx bx-dec"}[kind]
    s = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
    if accent:
        s += f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2" style="fill:{accent}"/>'
    ty = y + (h / 2 - (8 if sub else 0) - (7 if sub2 else 0)) + 5
    s += f'<text class="t-b" x="{x + w / 2}" y="{ty}" text-anchor="middle">{E(title)}</text>'
    if sub:
        s += f'<text class="t-s" x="{x + w / 2}" y="{ty + 17}" text-anchor="middle">{E(sub)}</text>'
    if sub2:
        s += f'<text class="t-s" x="{x + w / 2}" y="{ty + 32}" text-anchor="middle">{E(sub2)}</text>'
    if kind == "frozen" and tag:
        s += f'<text class="t-tag" x="{x + w - 8}" y="{y + 14}" text-anchor="end">❄ frozen</text>'
    if kind in ("train", "dec") and tag:
        s += f'<text class="t-tag t-tag-tr" x="{x + w - 8}" y="{y + 14}" text-anchor="end">▲ trainable</text>'
    return s


def pill(x, y, w, h, title, sub="", cls="pl"):
    s = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}"/>'
    s += f'<text class="t-b t-loss" x="{x + w / 2}" y="{y + h / 2 + (-2 if sub else 5)}" text-anchor="middle">{E(title)}</text>'
    if sub:
        s += f'<text class="t-s" x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle">{E(sub)}</text>'
    return s


def arrow(pts, label="", lx=None, ly=None, cls="ar", anchor="middle"):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    s = f'<path class="{cls}" d="{d}" marker-end="url(#ah)"/>'
    if label:
        x, y = (lx, ly) if lx is not None else ((pts[-2][0] + pts[-1][0]) / 2, (pts[-2][1] + pts[-1][1]) / 2 - 6)
        s += f'<text class="t-s t-ar" x="{x}" y="{y}" text-anchor="{anchor}">{E(label)}</text>'
    return s


def svg(w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}" class="dg">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>'
            '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            '<line x1="0" y1="0" x2="0" y2="6" class="hatch"/></pattern></defs>' + body + "</svg>")


# ------------------------------------------------------------------------------------------------ training diagrams
def train_diagram(cross, lam_d, lam_g):
    acc = COL["G1-cross Encoder Only"] if cross else COL["G1-pool"]
    b = ""
    # row 1: crop teacher (top)
    b += box(20, 40, 140, 64, "GT crop", "원본 픽셀 · 384", "data")
    b += box(190, 40, 175, 64, "Original DINOv3-B/16", "teacher E₀", "frozen")
    b += box(395, 40, 150, 64, "Crop tokens T", "24×24×768 (C0 grid)", "plain")
    b += box(575, 40, 175, 64, "Crop 유효 셀 pool", "c = letterbox coverage", "plain")
    b += arrow([(160, 72), (190, 72)]) + arrow([(365, 72), (395, 72)]) + arrow([(545, 72), (575, 72)])
    # row 2: student (middle)
    b += box(20, 180, 140, 64, "Full image", "384 · 이미지당 1회", "data")
    b += box(190, 180, 175, 64, "Student DINOv3-B/16", "E_s · 같은 초기 가중치", "train", accent=acc)
    b += box(395, 180, 150, 64, "Full tokens F", "24×24×768 (576)", "plain")
    b += box(575, 180, 175, 64, "ROI 면적 가중 pool", "w_k = |box ∩ cell_k|", "plain")
    b += arrow([(160, 212), (190, 212)]) + arrow([(365, 212), (395, 212)]) + arrow([(545, 212), (575, 212)])
    # L_roi pill between rows 1 and 2
    b += pill(790, 112, 190, 60, "L_roi  (×1)", "1 − cos(pool F, pool T)", "pl pl-roi")
    b += arrow([(750, 72), (770, 72), (800, 112)], "pool T", 792, 88, anchor="start")
    b += arrow([(750, 212), (770, 212), (800, 172)], "pool F (encoder에 직접)", 762, 236, anchor="start")
    # row 3: decoder (cross) or placeholder
    if cross:
        b += box(395, 300, 150, 64, "ROI box + query grid", "24×24 위치 · Fourier PE", "data")
        b += box(575, 300, 175, 64, "ROI Cross-Attn Dec.", "2 blocks · 768-d · 8 heads", "dec", accent=COL["G1-cross + Decoder"])
        b += arrow([(470, 244), (470, 272), (610, 272), (610, 300)], "K/V = F 576개 전부 (ROI mask 없음)", 476, 266, anchor="start")
        b += arrow([(545, 332), (575, 332)])
        b += pill(790, 302, 190, 60, f"L_detail  (×{lam_d})", "token cos + pooled cos", "pl pl-det")
        b += arrow([(750, 332), (790, 332)], "Z", 770, 324)
        # T -> L_detail routed around the right edge
        b += arrow([(470, 40), (470, 18), (995, 18), (995, 332), (980, 332)], "T 위치별 (q ↔ q, stop-grad)", 700, 13, cls="ar ar-dash")
    else:
        b += f'<rect class="bx bx-ghost" x="395" y="300" width="585" height="64" rx="8"/>'
        b += '<text class="t-s" x="687" y="337" text-anchor="middle">Decoder 없음 — L_detail 없음 (G1-pool)</text>'
    # row 4: global preservation (bottom)
    b += box(20, 420, 140, 64, "Full image", "같은 입력", "data")
    b += box(190, 420, 175, 64, "Original DINOv3-B/16", "E₀ (EMA 아님)", "frozen")
    b += box(395, 420, 150, 64, "CLS₀, mean token₀", "원본 global 요약", "plain")
    b += arrow([(160, 452), (190, 452)]) + arrow([(365, 452), (395, 452)])
    b += pill(790, 422, 190, 60, f"L_global  (×{lam_g})", "½[(1−cos CLS)+(1−cos mean)]", "pl pl-glob")
    b += arrow([(545, 452), (790, 452)])
    b += arrow([(277, 244), (277, 398), (885, 398), (885, 422)], "CLS_s, mean F (student)", 560, 392, cls="ar ar-dash")
    title = "G1-cross 학습 구조" if cross else "G1-pool 학습 구조"
    return svg(1000, 500, b, title)


def decoder_diagram():
    b = ""
    # left: image with ROI + query grid
    b += '<rect class="img" x="20" y="40" width="200" height="200" rx="4"/>'
    for i in range(1, 8):
        b += f'<line class="grid" x1="{20 + i * 25}" y1="40" x2="{20 + i * 25}" y2="240"/><line class="grid" x1="20" y1="{40 + i * 25}" x2="220" y2="{40 + i * 25}"/>'
    b += '<rect class="roi" x="95" y="85" width="80" height="62"/>'
    for i in range(6):
        for j in range(6):
            b += f'<circle class="qdot" cx="{101 + j * 13.6}" cy="{90 + i * 10.4}" r="2.2"/>'
    b += '<text class="t-s" x="120" y="262" text-anchor="middle">full image 24×24 token 셀(일부만 표시)</text>'
    b += '<text class="t-s" x="120" y="278" text-anchor="middle">● = ROI 안 24×24 query 위치(일부)</text>'
    b += '<text class="t-s" x="120" y="30" text-anchor="middle">ROI box (full-image 정규화 좌표)</text>'
    # query path
    b += box(258, 40, 262, 92, "Query 576개 (ROI마다 다름)", "Fourier(ROI 상대 xy) ⊕ Fourier(절대 xy)", "plain",
             sub2="⊕ log w·h, 종횡비 → MLP_q + E_grid")
    b += arrow([(220, 100), (258, 86)])
    # kv path
    b += box(258, 160, 262, 92, "Key / Value 576개", "LN(F_k + MLP_k(geo_k)) · F = full token", "plain",
             sub2="geo: 절대 xy · ROI 상대 xy · 겹침 · inside")
    b += arrow([(220, 200), (258, 206)])
    # blocks
    b += box(556, 70, 196, 150, "Cross-Attn Block × 2", "q ← q + MHA(LN q, LN kv)", "dec",
             accent=COL["G1-cross + Decoder"], sub2="q ← q + FFN×4(LN q)")
    b += '<text class="t-s" x="654" y="196" text-anchor="middle">SDPA · ROI 64개씩 chunk</text>'
    b += arrow([(520, 86), (556, 120)]) + arrow([(520, 206), (556, 170)])
    b += box(786, 95, 200, 100, "Z = Linear(LN q)", "576×768 = ROI 24×24", "plain", sub2="Z(i,j) ↔ teacher T(i,j)")
    b += arrow([(752, 145), (786, 145)])
    b += '<text class="t-s" x="640" y="266" text-anchor="middle">ROI hard mask 없음 — ROI 바깥 token도 attend (letterbox padding 셀만 mask)</text>'
    b += '<text class="t-s" x="640" y="284" text-anchor="middle">feature bilinear 보간 없음 — 576개 query가 attention으로 full token을 직접 읽음</text>'
    return svg(1000, 300, b, "ROI cross-attention decoder 상세")


def eval_row(y, name, steps, note):
    c = COL[name]
    b = f'<rect x="10" y="{y}" width="6" height="56" rx="3" style="fill:{c}"/>'
    b += f'<text class="t-b" x="26" y="{y + 24}">{E(name)}</text><text class="t-s" x="26" y="{y + 42}">{E(note)}</text>'
    x = 200
    for i, (t, s, k) in enumerate(steps):
        w = 150 if i < len(steps) - 1 else 200
        b += box(x, y, w, 56, t, s, k, accent=(c if k in ("train", "dec") else None), tag=False)
        if i < len(steps) - 1:
            b += arrow([(x + w, y + 28), (x + w + 22, y + 28)])
        x += w + 22
    return b


def eval_diagram():
    R2f = ("R2 reader", "box와 겹친 칸만 읽음 (cov>0)", "plain")
    R2c = ("R2 reader", "crop grid 유효 칸 (cov>0)", "plain")
    rows = [
        ("Frozen Full", [("Full image", "384 · 1회 encode", "data"), ("Original DINOv3", "frozen", "frozen"),
                         ("Full tokens 576", "보간 없음", "plain"), R2f], "baseline"),
        ("G1-pool", [("Full image", "384 · 1회 encode", "data"), ("G1-pool encoder", "최종 EMA · frozen", "frozen"),
                     ("Full tokens 576", "보간 없음", "plain"), R2f], "encoder만 distill"),
        ("G1-cross Encoder Only", [("Full image", "384 · 1회 encode", "data"), ("G1-cross encoder", "최종 EMA · frozen", "frozen"),
                                   ("Full tokens 576", "decoder 우회", "plain"), R2f], "같은 ckpt, decoder 제거"),
        ("G1-cross + Decoder", [("Full image", "384 · 1회 encode", "data"), ("G1-cross encoder", "최종 EMA · frozen", "frozen"),
                                ("Decoder → Z 24×24", "ROI box query · frozen", "frozen"), R2c], "같은 ckpt + decoder"),
        ("C0 Crop Oracle", [("GT crop", "ROI마다 재인코딩", "data"), ("Original DINOv3", "frozen", "frozen"),
                            ("Crop tokens 576", "C0 grid", "plain"), R2c], "참조 상한 (ivc_v3)"),
    ]
    b = "".join(eval_row(20 + i * 74, n, s, note) for i, (n, s, note) in enumerate(rows))
    return svg(1000, 20 + len(rows) * 74, b, "PACO 평가 경로 5개")


# ------------------------------------------------------------------------------------------------ charts
def dot_plot(points, key, lo, hi, title, w=470):
    """points: {method: value}; position encoding (non-zero axis allowed for dots)."""
    L, Rm, top, rowh = 120, 50, 28, 34
    h = top + rowh * len(METH) + 30
    sx = lambda v: L + (v - lo) / (hi - lo) * (w - L - Rm)  # noqa: E731
    b = f'<text class="t-b" x="0" y="16">{E(title)}</text>'
    t = lo
    step = 1.0
    while t <= hi + 1e-9:
        X = sx(t)
        b += f'<line class="gridl" x1="{X}" y1="{top}" x2="{X}" y2="{h - 24}"/><text class="t-ax" x="{X}" y="{h - 8}" text-anchor="middle">{t:.0f}</text>'
        t += step
    for i, (m, _, s) in enumerate(METH):
        if m not in points:
            continue
        v = points[m][key]
        y = top + rowh * i + rowh / 2
        b += f'<text class="t-s" x="{L - 10}" y="{y + 4}" text-anchor="end">{E(s)}</text>'
        b += f'<line class="stem" x1="{sx(lo)}" y1="{y}" x2="{sx(v)}" y2="{y}"/>'
        b += (f'<g class="hit" data-tip="{E(m)} · {E(key)} {v:.2f}"><rect x="{L}" y="{y - rowh / 2}" width="{w - L - Rm + 40}" height="{rowh}" class="hitbox"/>'
              f'<circle cx="{sx(v)}" cy="{y}" r="6" class="dot" style="fill:{COL[m]}"/></g>')
        b += f'<text class="t-v" x="{sx(v) + 11}" y="{y + 4}">{v:.2f}</text>'
    return f'<svg viewBox="0 0 {w} {h}" class="ch" role="img" aria-label="{E(title)}">{b}</svg>'


def forest(rows, title, lo, hi, w=470, rowh=30, L=170):
    """rows: [(label, colour, point, lo, hi)]"""
    Rm, top = 40, 26
    h = top + rowh * len(rows) + 28
    sx = lambda v: L + (v - lo) / (hi - lo) * (w - L - Rm)  # noqa: E731
    b = f'<text class="t-b" x="0" y="15">{E(title)}</text>'
    t = lo
    while t <= hi + 1e-9:
        X = sx(t)
        b += f'<line class="{"zero" if abs(t) < 1e-9 else "gridl"}" x1="{X}" y1="{top - 4}" x2="{X}" y2="{h - 22}"/>'
        b += f'<text class="t-ax" x="{X}" y="{h - 7}" text-anchor="middle">{t:+.0f}</text>'
        t += 1 if hi - lo <= 12 else 2
    for i, (lab, c, p, a, z) in enumerate(rows):
        y = top + rowh * i + rowh / 2
        sig = a > 0 or z < 0
        b += f'<text class="t-s" x="{L - 10}" y="{y + 4}" text-anchor="end">{E(lab)}</text>'
        tip = f"{lab}: {p:+.2f} [{a:+.2f}, {z:+.2f}]" + (" · CI가 0 제외" if sig else " · n.s.")
        b += (f'<g class="hit" data-tip="{E(tip)}"><rect class="hitbox" x="{L}" y="{y - rowh / 2}" width="{w - L}" height="{rowh}"/>'
              f'<line class="ci" x1="{sx(max(a, lo))}" y1="{y}" x2="{sx(min(z, hi))}" y2="{y}" style="stroke:{c}"/>'
              f'<circle cx="{sx(p)}" cy="{y}" r="{6 if sig else 5}" class="dot{"" if sig else " dot-open"}" style="{"fill" if sig else "stroke"}:{c}"/></g>')
        b += f'<text class="t-v" x="{w - 2}" y="{y + 4}" text-anchor="end">{p:+.2f}{"*" if sig else ""}</text>'
    return f'<svg viewBox="0 0 {w} {h}" class="ch" role="img" aria-label="{E(title)}">{b}</svg>'


def line_chart(series, title, ylab, w=470, h=220, ylo=None, yhi=None, fmt="{:.3f}", logy=False):
    """series: [(name, colour, [(x, y)])]; crosshair + tooltip via JS (data-series JSON)."""
    L, Rm, top, bot = 54, 16, 50, 30
    xs = [x for _, _, p in series for x, _ in p]
    ys = [y for _, _, p in series for _, y in p]
    x0, x1 = min(xs), max(xs)
    y0 = min(ys) if ylo is None else ylo
    y1 = max(ys) if yhi is None else yhi
    pad = (y1 - y0) * 0.06 or 0.01
    y0, y1 = y0 - (pad if ylo is None else 0), y1 + (pad if yhi is None else 0)
    sx = lambda v: L + (v - x0) / (x1 - x0) * (w - L - Rm)  # noqa: E731
    sy = lambda v: top + (1 - (v - y0) / (y1 - y0)) * (h - top - bot)  # noqa: E731
    b = f'<text class="t-b" x="0" y="15">{E(title)}</text><text class="t-ax" x="0" y="{top - 14}">y: {E(ylab)}</text>'
    lx = w - Rm
    for name, c, _ in reversed(series):
        tw = 7 * len(name) + 8
        lx -= tw + 22
        b += (f'<line x1="{lx}" y1="{top - 18}" x2="{lx + 16}" y2="{top - 18}" style="stroke:{c};stroke-width:2.5"/>'
              f'<text class="t-s" x="{lx + 20}" y="{top - 14}">{E(name)}</text>')
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        b += f'<line class="gridl" x1="{L}" y1="{sy(v)}" x2="{w - Rm}" y2="{sy(v)}"/><text class="t-ax" x="{L - 6}" y="{sy(v) + 4}" text-anchor="end">{fmt.format(v)}</text>'
    for xv in range(0, int(x1) + 1, 3000):
        b += f'<text class="t-ax" x="{sx(xv)}" y="{h - 10}" text-anchor="middle">{xv // 1000}k</text>'
    b += f'<line class="base" x1="{L}" y1="{h - bot}" x2="{w - Rm}" y2="{h - bot}"/>'
    data = []
    for name, c, p in series:
        d = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p)
        b += f'<path class="ln" d="{d}" style="stroke:{c}"/>'
        if len(p) <= 8:
            for x, y in p:
                b += f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" class="dot" style="fill:{c}"/>'
        data.append(dict(name=name, color=c, pts=[[round(sx(x), 1), round(sy(y), 1), x, y] for x, y in p]))
    b += f'<line class="xh" x1="0" y1="{top}" x2="0" y2="{h - bot}" style="display:none"/>'
    b += f'<rect class="hitbox xh-area" x="{L}" y="{top}" width="{w - L - Rm}" height="{h - top - bot}"/>'
    js = E(json.dumps(dict(series=data, fmt=fmt)))
    return f'<svg viewBox="0 0 {w} {h}" class="ch lc" data-lc="{js}" role="img" aria-label="{E(title)}">{b}</svg>'


def scatter(pts, title, w=1000, h=440):
    """pts: [(method, x, y)]; all-pairs form -> direct labels on every point (identity never colour-alone)."""
    L, Rm, top, bot = 56, 40, 58, 44
    x0, x1, y0, y1 = 0.70, 1.02, 43.5, 47.0
    sx = lambda v: L + (v - x0) / (x1 - x0) * (w - L - Rm)  # noqa: E731
    sy = lambda v: top + (1 - (v - y0) / (y1 - y0)) * (h - top - bot)  # noqa: E731
    b = f'<text class="t-b" x="0" y="16">{E(title)}</text>'
    for v in (44, 45, 46, 47):
        b += f'<line class="gridl" x1="{L}" y1="{sy(v)}" x2="{w - Rm}" y2="{sy(v)}"/><text class="t-ax" x="{L - 6}" y="{sy(v) + 4}" text-anchor="end">{v}</text>'
    for v in (0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
        b += f'<line class="gridl" x1="{sx(v)}" y1="{top}" x2="{sx(v)}" y2="{h - bot}"/><text class="t-ax" x="{sx(v)}" y="{h - bot + 16}" text-anchor="middle">{v:.2f}</text>'
    b += f'<text class="t-ax" x="{(L + w - Rm) / 2}" y="{h - 6}" text-anchor="middle">Crop ROI cosine to teacher (COCO dev) →  복원 좋아짐</text>'
    b += f'<text class="t-ax" x="{L - 40}" y="{top - 14}">↑ R2 M0 Overall (PACO val)</text>'
    off = {"Frozen Full": (10, 16, "start"), "G1-pool": (-10, 18, "end"), "G1-cross Encoder Only": (-10, -10, "end"),
           "G1-cross + Decoder": (10, 18, "start"), "C0 Crop Oracle": (-10, -10, "end")}
    # arrow enc -> dec (same encoder)
    e = [p for p in pts if p[0] == "G1-cross Encoder Only"][0]
    d = [p for p in pts if p[0] == "G1-cross + Decoder"][0]
    b += f'<path class="ar ar-dash" d="M{sx(e[1]) + 6},{sy(e[2]) + 3} L{sx(d[1]) - 8},{sy(d[2]) - 4}" marker-end="url(#ah2)"/>'
    b += f'<text class="t-s" x="{(sx(e[1]) + sx(d[1])) / 2 + 8}" y="{(sy(e[2]) + sy(d[2])) / 2 - 2}">decoder 추가: 복원↑ PACO↓</text>'
    for m, x, y in pts:
        dx, dy, an = off[m]
        b += (f'<g class="hit" data-tip="{E(m)} · ROI cos {x:.3f} · M0 {y:.2f}"><circle cx="{sx(x)}" cy="{sy(y)}" r="14" class="hitbox"/>'
              f'<circle cx="{sx(x)}" cy="{sy(y)}" r="7" class="dot" style="fill:{COL[m]}"/></g>')
        b += f'<text class="t-s t-lab" x="{sx(x) + dx}" y="{sy(y) + dy}" text-anchor="{an}">{E(SHORT[m])} ({y:.2f})</text>'
    defs = ('<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker></defs>')
    return f'<svg viewBox="0 0 {w} {h}" class="ch" role="img" aria-label="{E(title)}">{defs}{b}</svg>'


def hbars(groups, title, lo, hi, w=1000):
    """Preservation bars: groups [(metric label, {method: value})]; bars start at `lo` is NOT allowed for magnitude ->
    drawn as dots (position) with a reference line at 1.0 (= original)."""
    L, Rm, top, rowh = 190, 50, 44, 22
    meths = ["G1-pool", "G1-cross Encoder Only"]
    h = top + len(groups) * (rowh * len(meths) + 12) + 28
    sx = lambda v: L + (v - lo) / (hi - lo) * (w - L - Rm)  # noqa: E731
    b = f'<text class="t-b" x="0" y="15">{E(title)}</text>'
    for v in [lo + k * (hi - lo) / 5 for k in range(6)]:
        b += f'<line class="{"zero" if abs(v - 1) < 1e-9 else "gridl"}" x1="{sx(v)}" y1="{top - 4}" x2="{sx(v)}" y2="{h - 22}"/><text class="t-ax" x="{sx(v)}" y="{h - 7}" text-anchor="middle">{v:.2f}</text>'
    b += f'<text class="t-ax" x="{sx(1.0) - 4}" y="{top - 8}" text-anchor="end">원본 = 1.00</text>'
    y = top
    for lab, vals in groups:
        b += f'<text class="t-s" x="{L - 10}" y="{y + rowh + 2}" text-anchor="end">{E(lab)}</text>'
        for m in meths:
            v = vals[m]
            cy = y + rowh / 2
            b += (f'<g class="hit" data-tip="{E(m)} · {E(lab)} {v:.3f}"><rect class="hitbox" x="{L}" y="{y}" width="{w - L}" height="{rowh}"/>'
                  f'<line class="stem" x1="{sx(v)}" y1="{cy}" x2="{sx(1.0)}" y2="{cy}"/>'
                  f'<circle cx="{sx(v)}" cy="{cy}" r="5.5" class="dot" style="fill:{COL[m]}"/></g>'
                  f'<text class="t-v" x="{sx(v) - 9}" y="{cy + 4}" text-anchor="end">{v:.3f}</text>')
            y += rowh
        y += 12
    return f'<svg viewBox="0 0 {w} {h}" class="ch" role="img" aria-label="{E(title)}">{b}</svg>'


# ------------------------------------------------------------------------------------------------ page
CSS = """
:root{color-scheme:only light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--base:#c3c2b7;
--border:rgba(11,11,11,.10);--c-frozen:#898781;--c-pool:#2a78d6;--c-xenc:#eb6834;--c-xdec:#1baf7a;--c-oracle:#52514e;
--train-bg:#fff4ee;--frozen-bg:#f1f0ec;--data-bg:#eef4fc;--loss-roi:#e4effc;--loss-det:#e3f5ee;--loss-glob:#f1f0ec;--good:#006300}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI","Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR","Noto Sans CJK KR",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 6px}h2{font-size:20px;margin:44px 0 10px;padding-top:8px;border-top:1px solid var(--grid)}
h3{font-size:16px;margin:22px 0 8px}p,li{color:var(--ink2)}.lead{font-size:16px;color:var(--ink)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;margin:14px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}.tile .k{font-size:13px;color:var(--ink2)}
.tile .v{font-size:28px;font-weight:650;margin:2px 0}.tile .s{font-size:13px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:14px}
.dgw{overflow-x:auto}.dgw svg.dg,.dgw svg.ch{min-width:760px}
svg{width:100%;height:auto;display:block}
.dg text,.ch text{fill:var(--ink)} .t-b{font-size:14px;font-weight:600}.t-s{font-size:12px;fill:var(--ink2)!important}.t-ax{font-size:11px;fill:var(--muted)!important}
.t-v{font-size:12px;fill:var(--ink)!important;font-variant-numeric:tabular-nums}.t-tag{font-size:10px;fill:var(--muted)!important}.t-tag-tr{fill:var(--ink2)!important;font-weight:600}
.t-loss{font-size:14px}.t-ar{font-size:11px}
.bx{fill:var(--surface);stroke:var(--base);stroke-width:1.2}.bx-train{fill:var(--train-bg);stroke:var(--c-xenc);stroke-width:1.6}
.bx-dec{fill:var(--loss-det);stroke:var(--c-xdec);stroke-width:1.6}.bx-frozen{fill:var(--frozen-bg);stroke:var(--muted);stroke-dasharray:5 3}
.bx-data{fill:var(--data-bg);stroke:var(--base)}.bx-ghost{fill:none;stroke:var(--base);stroke-dasharray:4 4}
.pl{stroke-width:1.6}.pl-roi{fill:var(--loss-roi);stroke:var(--c-pool)}.pl-det{fill:var(--loss-det);stroke:var(--c-xdec)}.pl-glob{fill:var(--loss-glob);stroke:var(--muted)}
.ar{fill:none;stroke:var(--ink2);stroke-width:1.5}.ar-dash{stroke-dasharray:5 4}.ah{fill:var(--ink2)}.hatch{stroke:var(--muted);stroke-width:1}
.img{fill:var(--data-bg);stroke:var(--base)}.grid{stroke:var(--base);stroke-width:.7}.roi{fill:none;stroke:var(--c-xdec);stroke-width:2.2}.qdot{fill:var(--c-xdec)}
.gridl{stroke:var(--grid);stroke-width:1}.zero{stroke:var(--ink2);stroke-width:1.2}.base{stroke:var(--base)}
.stem{stroke:var(--grid);stroke-width:2}.ci{stroke-width:2.2;stroke-linecap:round}.dot{stroke:var(--surface);stroke-width:2}.dot-open{fill:var(--surface)!important;stroke-width:2.2}
.ln{fill:none;stroke-width:2}.xh{stroke:var(--muted);stroke-dasharray:3 3}.hitbox{fill:transparent}.hit{cursor:default}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin:6px 0 4px}.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--frozen-bg);border-radius:8px;padding:10px 12px;color:var(--ink);overflow-x:auto;white-space:pre}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}.tw{overflow-x:auto}
.seg{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}.seg button{font:inherit;font-size:13px;border:1px solid var(--border);background:var(--surface);color:var(--ink2);border-radius:999px;padding:4px 12px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);font-size:12px;padding:6px 9px;border-radius:6px;opacity:0;transition:opacity .08s;z-index:9;max-width:320px}
.qa{display:grid;grid-template-columns:52px 1fr;gap:6px 12px;align-items:start}.qa b{color:var(--ink)}.qa .q{font-weight:700;color:var(--ink)}
.note{font-size:13px;color:var(--muted)}details summary{cursor:pointer;color:var(--ink2);font-size:14px;margin:6px 0}
"""

JS = """
const tip=document.getElementById('tip');
function show(t,e){tip.textContent=t;tip.style.opacity=1;const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px'}
function hide(){tip.style.opacity=0}
document.querySelectorAll('.hit').forEach(g=>{g.addEventListener('mousemove',e=>show(g.dataset.tip,e));g.addEventListener('mouseleave',hide)});
document.querySelectorAll('svg.lc').forEach(s=>{const d=JSON.parse(s.dataset.lc);const xh=s.querySelector('.xh');const area=s.querySelector('.xh-area');
 area.addEventListener('mousemove',e=>{const pt=s.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(s.getScreenCTM().inverse());
  let best=null;d.series.forEach(se=>{se.pts.forEach(q=>{if(best===null||Math.abs(q[0]-p.x)<Math.abs(best-p.x))best=q[0]})});
  xh.setAttribute('x1',best);xh.setAttribute('x2',best);xh.style.display='';
  const lines=d.series.map(se=>{let q=se.pts.reduce((a,b)=>Math.abs(b[0]-best)<Math.abs(a[0]-best)?b:a);return se.name+': '+(+q[3]).toFixed(4)});
  const st=d.series[0].pts.reduce((a,b)=>Math.abs(b[0]-best)<Math.abs(a[0]-best)?b:a)[2];show('step '+st+' · '+lines.join(' · '),e)});
 area.addEventListener('mouseleave',()=>{xh.style.display='none';hide()})});
document.querySelectorAll('.seg').forEach(seg=>{seg.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
 seg.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));const grp=seg.dataset.group;
 document.querySelectorAll('[data-'+grp+']').forEach(el=>el.style.display=el.dataset[grp]===b.dataset.k?'':'none')}))});
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = a.run
    res = json.load(open(os.path.join(R, "results.json")))
    w = json.load(open(os.path.join(R, "calib", "weights.json")))
    pts, con, feat = res["points"], res["contrasts"], res["features"]
    lam_d, lam_g = w["lam_detail"], w["lam_global"]
    cfg = json.load(open(os.path.join(R, "G1cross", "config.json")))
    diag = {r: [x for x in jl(os.path.join(R, r, "diag.jsonl")) if x["model"] in ("init", "ema")] for r in ("G1pool", "G1cross")}
    tlog = {r: jl(os.path.join(R, r, "train_log.jsonl")) for r in ("G1pool", "G1cross")}

    def C(x, y, col):
        s = con[f"{x} - {y}"][col]
        return s["point"], s["lo"], s["hi"]

    COLS = list(next(iter(pts.values())).keys())
    P = lambda m, c: pts[m][c]  # noqa: E731

    # ---------- tiles
    d3 = C("G1-cross + Decoder", "G1-cross Encoder Only", "M0 Overall")
    q1 = C("G1-pool", "Frozen Full", "M0 Overall")
    q2 = C("G1-cross Encoder Only", "Frozen Full", "M0 Overall")
    tiles = f"""<div class="tiles">
<div class="tile"><div class="k">Q1 · G1-pool − Frozen (M0)</div><div class="v">{q1[0]:+.2f}</div><div class="s">95% CI [{q1[1]:+.2f}, {q1[2]:+.2f}]</div></div>
<div class="tile"><div class="k">Q2 · G1-cross Enc − Frozen (M0)</div><div class="v">{q2[0]:+.2f}</div><div class="s">95% CI [{q2[1]:+.2f}, {q2[2]:+.2f}] · M0 {P('G1-cross Encoder Only','M0 Overall'):.2f} ≈ C0 {P('C0 Crop Oracle','M0 Overall'):.2f}</div></div>
<div class="tile"><div class="k">Q3 · Decoder 추가 효과 (M0)</div><div class="v">{d3[0]:+.2f}</div><div class="s">95% CI [{d3[1]:+.2f}, {d3[2]:+.2f}] · 같은 encoder</div></div>
<div class="tile"><div class="k">Q5 · Full CLS cos / 개별 patch token cos</div><div class="v">{feat['G1-cross Encoder Only']['full_cls_cos']:.3f} / {feat['G1-cross Encoder Only']['full_tok_cos']:.3f}</div><div class="s">G1-cross 기준 (G1-pool {feat['G1-pool']['full_cls_cos']:.3f} / {feat['G1-pool']['full_tok_cos']:.3f})</div></div>
</div>"""

    legend = '<div class="legend">' + "".join(f'<span><i style="background:{COL[m]}"></i>{E(m)}</span>' for m, _, _ in METH) + "</div>"

    # ---------- result charts
    dots = (f'<div class="grid2"><div class="card">{dot_plot(pts, "M0 Overall", 43, 48, "R2 M0 Overall (primary)")}</div>'
            f'<div class="card">{dot_plot(pts, "M2 Overall", 44, 49, "R2 M2 Overall")}</div></div>')

    slice_cols = ["M0 Overall", "M2 Overall", "Object", "Part", "Obj D5-16", "Part D1-4", "Obj D17+", "Color", "Material", "Pattern"]
    vs_frozen = []
    for c in slice_cols:
        rows = [(SHORT[m], COL[m], *C(m, "Frozen Full", c)) for m in ("G1-pool", "G1-cross Encoder Only", "G1-cross + Decoder", "C0 Crop Oracle")]
        vs_frozen.append(f'<div class="card">{forest(rows, c, -7, 8, w=470, L=120)}</div>')
    vs_frozen = '<div class="grid2">' + "".join(vs_frozen) + "</div>"

    key_rows = [("Q3  Dec − Enc (같은 ckpt)", "G1-cross + Decoder", "G1-cross Encoder Only", COL["G1-cross + Decoder"]),
                ("Q4  Enc − G1-pool", "G1-cross Encoder Only", "G1-pool", COL["G1-cross Encoder Only"]),
                ("Q4  Dec − G1-pool", "G1-cross + Decoder", "G1-pool", COL["G1-cross + Decoder"]),
                ("gap  C0 − Dec", "C0 Crop Oracle", "G1-cross + Decoder", COL["C0 Crop Oracle"])]
    seg = '<div class="seg" data-group="kc">' + "".join(
        f'<button data-k="{i}" aria-pressed="{str(i == 0).lower()}">{E(c)}</button>' for i, c in enumerate(slice_cols)) + "</div>"
    kc = "".join(f'<div data-kc="{i}" style="{"" if i == 0 else "display:none"}">'
                 + forest([(lab, col, *C(x, y, c)) for lab, x, y, col in key_rows], f"{c}: paired 차이 (95% CI)", -6, 6, w=900, L=230, rowh=36)
                 + "</div>" for i, c in enumerate(slice_cols))

    sc = scatter([("Frozen Full", feat["Frozen Full"]["roi_pool_cos"], P("Frozen Full", "M0 Overall")),
                  ("G1-pool", feat["G1-pool"]["roi_pool_cos"], P("G1-pool", "M0 Overall")),
                  ("G1-cross Encoder Only", feat["G1-cross Encoder Only"]["roi_pool_cos"], P("G1-cross Encoder Only", "M0 Overall")),
                  ("G1-cross + Decoder", feat["G1-cross + Decoder"]["dec_pool_cos"], P("G1-cross + Decoder", "M0 Overall")),
                  ("C0 Crop Oracle", 1.0, P("C0 Crop Oracle", "M0 Overall"))], "Crop teacher 복원 vs PACO 성능 (Q6)")

    pres = hbars([("Full CLS cos", {m: feat[m]["full_cls_cos"] for m in ("G1-pool", "G1-cross Encoder Only")}),
                  ("Full mean-token cos", {m: feat[m]["full_mean_cos"] for m in ("G1-pool", "G1-cross Encoder Only")}),
                  ("개별 patch token cos", {m: feat[m]["full_tok_cos"] for m in ("G1-pool", "G1-cross Encoder Only")}),
                  ("ROI 안 token cos", {m: feat[m]["in_tok_cos"] for m in ("G1-pool", "G1-cross Encoder Only")}),
                  ("ROI 바깥 token cos", {m: feat[m]["out_tok_cos"] for m in ("G1-pool", "G1-cross Encoder Only")}),
                  ("모든 ROI 바깥 token cos", {m: feat[m]["img_out_tok_cos"] for m in ("G1-pool", "G1-cross Encoder Only")})],
                 "원본 DINOv3 대비 보존 (Q5, COCO dev)", 0.70, 1.00)

    def tl(run, k):
        return [(r["step"], r[k]) for r in tlog[run]]

    def dg(run, k):
        return [(r["step"], r[k]) for r in diag[run] if r.get(k) is not None]
    curves = '<div class="grid2">' + "".join(f'<div class="card">{x}</div>' for x in [
        line_chart([("G1-pool", COL["G1-pool"], tl("G1pool", "l_roi")), ("G1-cross", COL["G1-cross Encoder Only"], tl("G1cross", "l_roi"))],
                   "L_roi (train, 100-step 평균)", "1 − cos", fmt="{:.3f}"),
        line_chart([("G1-pool", COL["G1-pool"], tl("G1pool", "l_global")), ("G1-cross", COL["G1-cross Encoder Only"], tl("G1cross", "l_global"))],
                   "L_global (train)", "½[(1−cos CLS)+(1−cos mean)]", fmt="{:.4f}", ylo=0.0),
        line_chart([("G1-cross", COL["G1-cross + Decoder"], tl("G1cross", "l_detail"))], "L_detail (train, G1-cross만)", "token + pooled", fmt="{:.3f}"),
        line_chart([("G1-pool enc", COL["G1-pool"], dg("G1pool", "roi_pool_cos")), ("G1-cross enc", COL["G1-cross Encoder Only"], dg("G1cross", "roi_pool_cos")),
                    ("G1-cross dec", COL["G1-cross + Decoder"], [(s, v) for s, v in dg("G1cross", "dec_pool_cos") if s > 0])],
                   "COCO dev: ROI pooled cos to teacher (EMA)", "cos", fmt="{:.2f}"),
        line_chart([("G1-pool", COL["G1-pool"], dg("G1pool", "full_tok_cos")), ("G1-cross", COL["G1-cross Encoder Only"], dg("G1cross", "full_tok_cos"))],
                   "COCO dev: 개별 patch token cos to 원본 (EMA)", "cos", fmt="{:.2f}"),
        line_chart([("G1-pool", COL["G1-pool"], dg("G1pool", "full_cls_cos")), ("G1-cross", COL["G1-cross Encoder Only"], dg("G1cross", "full_cls_cos"))],
                   "COCO dev: Full CLS cos to 원본 (EMA)", "cos", fmt="{:.3f}"),
    ]) + "</div>"
    cl = lambda run: "G1-pool" if run == "G1pool" else "G1-cross"  # noqa: E731

    # ---------- tables
    def tr(cells, th=False):
        t = "th" if th else "td"
        return "<tr>" + "".join(f"<{t}>{c}</{t}>" for c in cells) + "</tr>"
    main_t = tr(["Method", "R2 M0", "R2 M2", "Obj D5–16", "Part D1–4", "Full CLS Cos", "Crop ROI Cos"], True)
    for m, _, _ in METH:
        f = feat.get(m, {})
        roi = f.get("dec_pool_cos") if m == "G1-cross + Decoder" else f.get("roi_pool_cos")
        main_t += tr([f'<span style="color:{COL[m]}">●</span> {E(m)}', f"{P(m, 'M0 Overall'):.2f}", f"{P(m, 'M2 Overall'):.2f}",
                      f"{P(m, 'Obj D5-16'):.2f}", f"{P(m, 'Part D1-4'):.2f}",
                      "–" if m == "C0 Crop Oracle" else f"{f['full_cls_cos']:.3f}", "1" if m == "C0 Crop Oracle" else f"{roi:.3f}"])
    full_t = tr(["Method", "ep (dev M0)"] + COLS, True)
    for m, _, _ in METH:
        e = res["epochs"][m]
        full_t += tr([E(m), f"{e['epoch']} ({e['dev_M0']:.2f})"] + [f"{P(m, c):.2f}" for c in COLS])
    con_t = tr(["contrast"] + COLS, True)
    for k, v in con.items():
        con_t += tr([E(k.replace(" - ", " − "))] + [f"{v[c]['point']:+.2f} [{v[c]['lo']:+.2f}, {v[c]['hi']:+.2f}]{'*' if v[c]['lo'] > 0 or v[c]['hi'] < 0 else ''}" for c in COLS])
    fk = [("roi_pool_cos", "ROI pooled cos (enc)"), ("dec_tok_cos", "Dec token cos"), ("dec_pool_cos", "Dec pooled cos"), ("sv_ratio", "SV ratio"),
          ("rel_err", "Relation err"), ("full_cls_cos", "Full CLS"), ("full_mean_cos", "Full mean-token"), ("full_tok_cos", "per-token"),
          ("in_tok_cos", "ROI 안"), ("out_tok_cos", "ROI 바깥"), ("img_out_tok_cos", "모든 ROI 바깥")]
    feat_t = tr(["Method"] + [b for _, b in fk], True)
    for m in ("Frozen Full", "G1-pool", "G1-cross Encoder Only", "G1-cross + Decoder"):
        f = feat[m]
        dec = m == "G1-cross + Decoder"
        feat_t += tr([E(m)] + [("–" if (k in ("dec_tok_cos", "dec_pool_cos", "sv_ratio", "rel_err") and not dec) or f.get(k) is None else f"{f[k]:.4f}") for k, _ in fk])
    gr_t = tr(["run", "step", "‖∇enc L_roi‖", "‖∇enc L_global‖", "‖∇enc L_detail‖", "‖∇dec L_detail‖"], True)
    for run in ("G1pool", "G1cross"):
        for r in res["grads"][run]["check"]:
            if r["step"] in (1, 2, 3, 1000, 6000, 12000):
                gr_t += tr([cl(run), r["step"], f"{r['enc_roi']:.3f}", f"{r['enc_global']:.3f}",
                            f"{r['enc_detail']:.3f}" if "enc_detail" in r else "–", f"{r['dec_detail']:.3f}" if "dec_detail" in r else "–"])

    fD = feat["G1-cross + Decoder"]
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1-pool vs G1-cross</title><style>{CSS}</style></head><body><div id="tip" role="tooltip"></div><main style="position:relative">
<h1>G1-pool vs G1-cross: crop distillation into the full-image DINOv3</h1>
<p class="note">2026-11-06 · seed 0 · DINOv3-B/16 @384 · distillation은 COCO만 사용 (PACO attribute label 미사용) · 평가는 PACO val R2 (test 미실행) · run dir <code>output/2026-11-06/g1_cross_distill</code></p>
<p class="lead"><b>결론:</b> encoder에 crop을 직접 distill하면 PACO R2가 확실히 오른다(M0 +{q1[0]:.2f} / +{q2[0]:.2f}). 반면 같은 encoder 위에 ROI cross-attention decoder를 얹으면 오히려 떨어진다({d3[0]:+.2f}*). Decoder 출력이 crop teacher를 가장 잘 복원했는데도(pooled cos {fD['dec_pool_cos']:.3f}) PACO는 오르지 않았다.</p>
{tiles}
{legend}

<h2>1. 방법별 구조와 loss</h2>
<p>두 방법 모두 <b>같은 초기 DINOv3</b>에서 시작하고 <b>같은 frozen teacher</b>를 쓴다. 차이는 G1-cross에만 있는 ROI decoder와 L_detail이다.
L_roi와 L_global의 정의와 가중치는 두 방법에서 똑같다. 실선 화살표는 forward, 점선은 비교 대상(stop-gradient teacher / student 출력)이다.</p>
<h3>1-A. G1-pool: encoder만으로 crop distillation</h3>
<div class="card dgw">{train_diagram(False, lam_d, lam_g)}</div>
<div class="formula">L(G1-pool) = L_roi + {lam_g}·L_global
L_roi    = mean_ROI [ 1 − cos( Σ_k w_k F_k / Σ_k w_k ,  Σ_q c_q T_q / Σ_q c_q ) ]
           w_k = ROI box ∩ full cell 면적,  c_q = crop letterbox 유효 coverage
L_global = mean_img ½[ (1 − cos(CLS_s, CLS_0)) + (1 − cos(mean F, mean F_0)) ]
           기준 = 학습 전 원본 DINOv3 (EMA 아님)</div>
<h3>1-B. G1-cross: encoder + ROI cross-attention decoder 공동 학습</h3>
<div class="card dgw">{train_diagram(True, lam_d, lam_g)}</div>
<div class="formula">L(G1-cross) = L_roi + {lam_d}·L_detail + {lam_g}·L_global
              (L_roi, L_global은 G1-pool과 동일)
L_detail = mean_ROI [ Σ_q c_q (1 − cos(Z_q, T_q)) / Σ_q c_q ]  +  mean_ROI [ 1 − cos(pool_c Z, pool_c T) ]
           Z_q와 T_q는 같은 crop-grid 셀(= 같은 원본 위치)
           T는 stop-gradient, gradient는 decoder → encoder까지 흐름</div>
<p class="note">가중치는 학습 전에 한 번 정하고 고정했다. λ_detail = {lam_d}: 첫 16 stream step에서 잰 median(‖∇enc L_roi‖ / ‖∇enc L_detail‖), IQR {w['ratio_iqr'][0]:.2f}–{w['ratio_iqr'][1]:.2f}.
λ_global = {lam_g}: 초기에는 student가 원본과 같아 L_global의 gradient가 0이므로 비율로 정할 수 없어, 기존 G1 값을 썼다. PACO 결과를 보고 다시 조정하지 않았다.</p>
<h3>1-C. ROI cross-attention decoder 상세 (G1-cross, {cfg['n_params_dec'] / 1e6:.1f}M params)</h3>
<div class="card dgw">{decoder_diagram()}</div>
<h3>1-D. PACO 평가 경로 (encoder·decoder는 모두 frozen, R2만 학습)</h3>
<div class="card dgw">{eval_diagram()}</div>
<p class="note">R2 = learned query + box embedding이 token(+geometry MLP)을 cross-attention 2블록으로 읽는 reader다. <b>576개를 모두 받지만 cov&gt;0 mask가 있어서, Full 경로(위 3줄)는 full image token 중 ROI box와 겹치는 칸만 attend한다</b>(작은 ROI는 몇 칸). Decoder와 C0 경로는 crop grid의 유효 칸을 읽는다. PACO train 37,472장, 10 epoch. Epoch은 PACO dev 1,000장의 M0로 골랐고, 최종 평가는 PACO val이다. 다섯 조건 모두 같은 reader 구조, 같은 split, 같은 선택 규칙을 쓴다. Encoder·decoder의 체크섬이 probe 전후로 동일함을 확인했다.</p>

<h2>2. 결과: PACO val R2</h2>
{dots}
<div class="card tw"><table>{main_t}</table></div>
<p class="note">Crop ROI Cos: encoder 행 = cos(box 면적 가중 평균 full token, crop teacher 평균), G1-cross + Decoder 행 = cos(decoder 출력 평균, crop teacher 평균). COCO dev {cfg['n_dev_images']}장 / {cfg['n_dev_rois']:,} ROI, 최종 EMA 기준.</p>

<h3>2-A. 질문별 핵심 비교 (paired image bootstrap 95% CI, B = 1000)</h3>
<p>지표를 고르면 그 지표의 paired 차이를 보여준다. 채운 점 = CI가 0을 제외, 빈 점 = n.s.</p>
{seg}
<div class="card dgw">{kc}</div>

<h3>2-B. Frozen Full 대비 차이: 지표별</h3>
<p>모든 패널이 같은 축(−7 ~ +8)을 쓴다. C0 Oracle은 ROI마다 crop을 다시 encode하는 참조값이다(추론 비용이 다름).</p>
{vs_frozen}

<h3>2-C. 복원이 좋아지면 PACO도 좋아지는가 (Q6)</h3>
<div class="card">{sc}</div>
<p class="note">Encoder 단계: ROI cos {feat['Frozen Full']['roi_pool_cos']:.3f} → {feat['G1-pool']['roi_pool_cos']:.3f}일 때 PACO도 오른다. Decoder 단계: 복원은 {feat['G1-cross Encoder Only']['roi_pool_cos']:.3f} → {fD['dec_pool_cos']:.3f}로 더 좋아졌지만 PACO는 {d3[0]:+.2f}로 떨어졌다.
Decoder 출력의 spatial-variance 비율은 {fD['sv_ratio']:.2f}(teacher = 1)다. Cosine loss는 크기를 제약하지 않는다. (C0 Oracle의 x = 1은 정의상 값이다.)</p>

<h2>3. 원본 표현 보존 (Q5)</h2>
<div class="card">{pres}</div>
<p>CLS와 평균 token은 0.97–0.98로 유지됐다. 그러나 <b>개별 patch token은 0.79–0.81</b>로 크게 바뀌었고, ROI 바깥 token도 비슷한 크기로 바뀌었다. L_global은 CLS와 평균만 붙잡기 때문이다.
ImageNet이 이 서버에 없어 global 분류 성능은 측정하지 못했다. 이 feature 지표만으로 분류 성능이 유지됐다고 단정하지 않는다.</p>

<h2>4. 학습 곡선과 COCO dev 진단</h2>
{curves}
<div class="card tw"><h3 style="margin-top:0">Gradient flow (loss별 gradient L2 norm, local batch)</h3><table>{gr_t}</table>
<p class="note">L_detail의 gradient는 decoder뿐 아니라 student encoder에도 흐른다(step 1부터 ‖∇enc L_detail‖ &gt; 0). L_global은 step 1에서 0이고, student가 원본에서 멀어지면서 커진다.</p></div>

<h2>5. 질문별 답</h2>
<div class="card qa">
<div class="q">Q1</div><div><b>그렇다.</b> G1-pool − Frozen: M0 {q1[0]:+.2f} [{q1[1]:+.2f}, {q1[2]:+.2f}]*, M2 {C('G1-pool','Frozen Full','M2 Overall')[0]:+.2f}*, Obj D5–16 {C('G1-pool','Frozen Full','Obj D5-16')[0]:+.2f}*. Part D1–4 {C('G1-pool','Frozen Full','Part D1-4')[0]:+.2f}(경계).</div>
<div class="q">Q2</div><div><b>그렇다.</b> G1-cross Enc − Frozen: M0 {q2[0]:+.2f}*, Obj D5–16 {C('G1-cross Encoder Only','Frozen Full','Obj D5-16')[0]:+.2f}*, Part D1–4 {C('G1-cross Encoder Only','Frozen Full','Part D1-4')[0]:+.2f}*. M0 Overall은 C0 Oracle과 같지만, 작은 object는 oracle보다 낮고 작은 part는 높다.</div>
<div class="q">Q3</div><div><b>아니다. 오히려 나빠진다.</b> Dec − Enc(같은 ckpt): M0 {d3[0]:+.2f}*, Part {C('G1-cross + Decoder','G1-cross Encoder Only','Part')[0]:+.2f}*, Color {C('G1-cross + Decoder','G1-cross Encoder Only','Color')[0]:+.2f}*.</div>
<div class="q">Q4</div><div><b>Encoder 기준으로만 소폭 그렇다.</b> Enc − G1-pool: Obj D5–16 {C('G1-cross Encoder Only','G1-pool','Obj D5-16')[0]:+.2f}*, Part D1–4 {C('G1-cross Encoder Only','G1-pool','Part D1-4')[0]:+.2f}(n.s.). Decoder 출력 기준으로는 아니다(Dec − G1-pool, M0 {C('G1-cross + Decoder','G1-pool','M0 Overall')[0]:+.2f}). 두 조건은 L_detail(encoder에도 걸림)도 다르므로 이 차이를 decoder만의 효과로 볼 수 없다.</div>
<div class="q">Q5</div><div>Global 요약은 보존됐다(CLS {feat['G1-pool']['full_cls_cos']:.3f} / {feat['G1-cross Encoder Only']['full_cls_cos']:.3f}). Patch token 단위로는 크게 바뀌었다(0.79–0.81, ROI 바깥도 비슷). 분류 성능은 측정하지 못했다.</div>
<div class="q">Q6</div><div><b>Encoder 단계에서는 이어졌고, decoder 단계에서는 이어지지 않았다.</b> 복원 지표가 가장 좋은 decoder 출력이 PACO는 더 낮다. 이전 decoder 실험(xdec, e6*)과 같은 패턴이다.</div>
</div>
<p class="note">이번 실험은 seed 0 한 번만 돌렸다. CI 경계 결과(Q4 encoder 비교 등)의 재현성은 확인하지 않았다. C0 Oracle은 이전 run(<code>ivc_v3/probe_c0</code>, world 1)을 재사용했다. DINOv2 수치와는 비교하지 않는다.</p>

<h2>6. 전체 표</h2>
<details><summary>PACO val 전체 지표 (dev-selected epoch)</summary><div class="card tw"><table>{full_t}</table></div></details>
<details><summary>Paired bootstrap 95% CI 전체</summary><div class="card tw"><table>{con_t}</table></div></details>
<details><summary>Feature 진단 전체 (COCO dev, 최종 EMA)</summary><div class="card tw"><table>{feat_t}</table></div></details>
</main><script>{JS}</script></body></html>"""
    out = a.out or os.path.join(R, "g1x_visualization.html")
    open(out, "w").write(page)
    print(out, len(page))


if __name__ == "__main__":
    main()
