"""G1-bq selection + report.
  python -m src.g1bq.report select   -> <run>/selection.json, prints the selected epoch (rule fixed earlier: max (Dev Obj AP +
                                        Dev Part AP)/2 over epochs 1..10, ties -> earlier; M0 / M2 never used)
  python -m src.g1bq.report report   -> <run>/results.md (+ results/g1bq_summary.md)
"""
import glob
import json
import os
import re
import sys

import numpy as np

G = "output/2026-11-08/g1bq"
TYPES = ("color", "pattern_marking", "material", "transparency")
HEAD = ["Obj AP", "Obj AP col", "Obj AP pat", "Obj AP mat", "Obj AP ref", "Part AP", "Part AP col", "Part AP pat", "Part AP mat", "Part AP ref"]
COLS = ["obj"] + [f"obj_{t}" for t in TYPES] + ["part"] + [f"part_{t}" for t in TYPES]
DEVROOT = {"g1bq": f"{G}/preds_dev/g1bq", "g1cross_enc": "output/2026-11-07/paco_devap/preds/g1cross_enc"}


def dev_curve(root):
    out = {}
    for f in glob.glob(os.path.join(root, "ep*", "official_dev.json")):
        e = int(re.search(r"ep(\d+)", f).group(1))
        o = json.load(open(f))
        out[e] = dict(obj=o["obj"], part=o["part"], avg=(o["obj"] + o["part"]) / 2)
    return dict(sorted(out.items()))


def select():
    c = dev_curve(DEVROOT["g1bq"])
    assert sorted(c) == list(range(1, 11)), sorted(c)
    b = max(sorted(c), key=lambda e: (c[e]["avg"], -e))
    json.dump(dict(rule="max (Dev Obj AP + Dev Part AP)/2 over epochs 1..10, ties -> earlier (fixed before test)", epoch=b, curve=c),
              open(f"{G}/selection.json", "w"), indent=1)
    print(b)


def ci(boot, a, b, k):
    if not boot or a not in boot["boot"] or b not in boot["boot"]:
        return "n/a"
    d = np.array(boot["boot"][a][k]) - np.array(boot["boot"][b][k])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{boot['points'][a][k] - boot['points'][b][k]:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("\\*" if lo > 0 or hi < 0 else "")


def table(L, boot, rows):
    L += ["| Model | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
    for k, n in rows:
        p = (boot or {}).get("points", {}).get(k)
        L.append(f"| {n} | " + (" | ".join(f"{p[c]:.2f}" for c in COLS) if p else " | ".join(["-"] * len(COLS))) + " |")
    L.append("")


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def report():
    sel = jl(f"{G}/selection.json")
    e = sel["epoch"] if sel else None
    bt, bv, bd = jl(f"{G}/boot_test.json"), jl(f"{G}/boot_val.json"), jl(f"{G}/boot_dev.json")
    L = ["# G1-cross with Bilinear-initialized Decoder Queries (G1-bq) — Encoder Only 평가", "",
         "- 학습: CLIP ViT-L/14@336, original CLIP 초기화, G1-cross와 동일한 COCO 12k stream / crop teacher / loss(L_roi + 1.11·L_detail + 0.5·L_global) / "
         "optimizer / 12,000 step / EMA, seed 0. 바꾼 것은 decoder query뿐이다: q0 = LN(B + pos) + q_grid, B = 학생 encoder token의 bilinear ROI 샘플(24×24 crop grid, letterbox 정렬).",
         "- 평가: decoder를 제거하고 Encoder Only + R2/head(기존과 같은 yaml, 10 epoch)를 학습했다. Checkpoint는 사전 고정한 Dev AP 규칙((Obj+Part)/2 최대)으로 골랐고, 그 checkpoint 하나로 공식 Test 전체(9,443 images / 131,899 annotations, GT box + category oracle)를 평가했다. Test 결과를 보고 바꾼 설정은 없다.",
         "- \\* = image-level paired bootstrap(B=1000) 95% CI가 0을 포함하지 않음.", ""]
    if sel:
        L += [f"## 1. Dev AP 곡선과 checkpoint 선택 (선택: **epoch {e}**)", "", "| epoch | " + " | ".join(str(k) for k in range(1, 11)) + " |", "|---|" + "---|" * 10]
        for n, root in DEVROOT.items():
            c = dev_curve(root)
            L.append(f"| {n} | " + " | ".join(f"{c[k]['obj']:.1f}/{c[k]['part']:.1f}" if k in c else "–" for k in range(1, 11)) + " |")
        L.append("")
    L += ["## 2. PACO-LVIS Official Test — GT Box + Category Oracle", ""]
    table(L, bt, [("clip_full", "Original CLIP (ep6)"), ("maskembed_full", "MaskEmbed (ep6)"), ("g1pool_enc", "G1-pool (ep6)"),
                  ("g1cross_enc_ep6", "G1-cross Encoder Only (ep6, M0 pick)"), ("g1cross_enc_ep4", "G1-cross Encoder Only (ep4, Dev AP pick)"),
                  ("g1bq", f"**G1-bq Encoder Only (ep{e}, Dev AP pick)**")])
    L += ["Test Δ (paired bootstrap 95% CI):", "", "| Contrast | Obj AP | Part AP | Obj mat | Obj pat | Part pat | Part col |", "|---|---|---|---|---|---|---|"]
    for lab, a, b in (("G1-bq − G1-cross Enc (same Dev-AP rule, ep4)", "g1bq", "g1cross_enc_ep4"), ("G1-bq − G1-cross Enc (ep6, reported)", "g1bq", "g1cross_enc_ep6"),
                      ("G1-bq − MaskEmbed", "g1bq", "maskembed_full"), ("G1-bq − Original CLIP", "g1bq", "clip_full")):
        L.append(f"| {lab} | " + " | ".join(ci(bt, a, b, k) for k in ("obj", "part", "obj_material", "obj_pattern_marking", "part_pattern_marking", "part_color")) + " |")
    L += ["", "## 3. 공식 Val (2,410 images)", ""]
    table(L, bv, [("clip_full", "Original CLIP (ep6)"), ("maskembed_full", "MaskEmbed (ep6)"), ("g1cross_enc_ep6", "G1-cross Encoder Only (ep6)"),
                  ("g1cross_enc_ep4", "G1-cross Encoder Only (ep4, Dev AP pick)"), ("g1bq", f"G1-bq Encoder Only (ep{e})"), ("g1bq_ep10", "G1-bq Encoder Only (ep10, fixed)")])
    L += ["| Contrast | Val Obj | Val Part | Dev Obj | Dev Part |", "|---|---|---|---|---|"]
    for lab, a, b in (("G1-bq − G1-cross Enc (Dev-AP picks)", "g1bq", "g1cross_enc_ep4"), ("G1-bq − G1-cross Enc ep6", "g1bq", "g1cross_enc_ep6"), ("G1-bq − MaskEmbed", "g1bq", "maskembed_full")):
        L.append(f"| {lab} | {ci(bv, a, b, 'obj')} | {ci(bv, a, b, 'part')} | {ci(bd, a, b, 'obj')} | {ci(bd, a, b, 'part')} |")
    L += ["", "## 4. Encoder 학습 중 region feature 변화 (COCO dev, EMA; G1-cross vs G1-bq)", "",
          "| step | model | ROI pooled cos (vs crop teacher) | in-ROI token cos (vs original) | out-of-ROI token cos | CLS cos | decoder token cos | decoder pooled cos |", "|---|---|---|---|---|---|---|---|"]
    for name, p in (("G1-cross", "output/2026-11-07/g1clip/G1cross/diag.jsonl"), ("G1-bq", f"{G}/G1bq/diag.jsonl")):
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                L.append(f"| {r['step']} | {name} {r['model']} | {r['roi_pool_cos']:.4f} | {r['in_tok_cos']:.4f} | {r['out_tok_cos']:.4f} | {r['full_cls_cos']:.4f} | "
                         f"{(r.get('dec_tok_cos') or 0):.4f} | {(r.get('dec_pool_cos') or 0):.4f} |")
    ck = jl(f"{G}/smoke/check_bq.json")
    if ck:
        L += ["", "## 5. 검증", "",
              f"- Bilinear query가 학생 encoder token에서 생성됨: dB/dF ≠ 0 ({ck['dB_dftok_nonzero']}); L_detail → encoder token gradient: query 경로 {ck['grad_to_encoder_tokens_query_path']:.2e}, key 경로 {ck['grad_to_encoder_tokens_key_path']:.2e}; encoder parameter gradient " + ", ".join(f"{k} {v:.3f}" for k, v in ck["detail_grad_encoder_params"].items()),
              "- Crop grid ↔ ROI 좌표 정렬: src.decstudy.checks (96 ROI 최대 오차 3.7e-5 px, 실제 픽셀 상관 0.978) 와 같은 bilinear_roi 사용",
              f"- 학습 parameter: decoder {ck['n_params_bq']:,} vs G1-cross {ck['n_params_g1cross']:,} (+ln_qin {ck['n_params_bq'] - ck['n_params_g1cross']:,}); 공유 parameter 초기값 동일 ({ck['shared_params_equal_g1cross_init']}); 추가 연산 ≈ bilinear sampling {ck['extra_flops_per_step_approx']['bilinear_sampling']} / step",
              "- 추론: decoder 미사용(Encoder Only), 이미지당 encoder 1회, ROI crop 재인코딩 없음"]
    dn = jl(f"{G}/G1bq/done.json")
    cfg = jl(f"{G}/G1bq/config.json")
    if dn and cfg:
        L.append(f"- 학습: world {cfg['world']} (G1-cross는 world 1; 64 ROI/step 동일, gradient 전역 합산), {dn['steps']} step, {dn['time_s'] / 3600:.2f} h")
    L += ["", "주요 비교 중 '학습 조건을 맞춘 MaskEmbed'와 'Mask + Crop View 결합 모델'은 이 세션에 결과가 없어 포함하지 못했다 (기존 MaskEmbed 공식 checkpoint 행만 참고로 표기)."]
    txt = "\n".join(L) + "\n"
    open(f"{G}/results.md", "w").write(txt)
    os.makedirs("results", exist_ok=True)
    open("results/g1bq_summary.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    {"select": select, "report": report}[sys.argv[1]]()
