"""Decoder-study report: tables A-D, sanity checks, rule-based answers -> results/decoder_study_summary.md (repo root) and
<run>/results/decoder_study_summary.md (+ summary.json). Numbers only from files written by the queue; rows whose inputs
are missing are printed as 'n/a' (never filled in).
"""
import glob
import json
import os
import re

import numpy as np

ROOT = "/home/irteam/Crop_representation"
RR = "output/2026-11-07/decoder_study"
TYPES = ("color", "pattern_marking", "material", "transparency")
TNAME = {"color": "Color", "pattern_marking": "Pattern", "material": "Material", "transparency": "Reflectance*"}
ROWS = [("g1cross_enc", "G1-cross Encoder Only"), ("g1cross_dec", "Existing G1-cross Decoder (joint, ep6)"), ("b0", "B0 Bilinear Only"),
        ("d1", "D1 Bilinear + Residual"), ("l1", "L1 D1 + Covariance"), ("l2", "L2 D1 + Relational")]
REFS = [("clip_full", "(ref) Original CLIP Full"), ("maskembed_full", "(ref) MaskEmbed Full"), ("c0", "(ref) Original CLIP C0 crop (oracle)")]
CAND = ("d1", "l1", "l2")


def jl(p):
    try:
        return json.load(open(os.path.join(ROOT, p)))
    except Exception:
        return None


def jsonl(p):
    try:
        return [json.loads(x) for x in open(os.path.join(ROOT, p))]
    except Exception:
        return []


def f2(x, s=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.2f}" if s else f"{x:.2f}"


def ci(boot, a, b, k):
    if not boot or a not in boot["boot"] or b not in boot["boot"]:
        return None
    pa, pb = boot["points"][a][k], boot["points"][b][k]
    d = np.array(boot["boot"][a][k]) - np.array(boot["boot"][b][k])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=pa - pb, lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))


def fci(c):
    if c is None:
        return "n/a"
    return f"{c['delta']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]" + ("\\*" if c["sig"] else "")


def main():
    val, dev = jl(f"{RR}/boot/val.json"), jl(f"{RR}/boot/dev.json")
    pv = (val or {}).get("points", {})
    pd_ = (dev or {}).get("points", {})
    S = {}
    L = []
    w = L.append
    # ---------------------------------------------------------------- choose the best candidate on DEV (not val)
    dev_avg = {c: (pd_[c]["obj"] + pd_[c]["part"]) / 2 for c in CAND if c in pd_}
    best = max(dev_avg, key=dev_avg.get) if dev_avg else None
    S["best_candidate_by_dev"] = best
    cons = [("D1 − B0 Bilinear", "d1", "b0"), ("D1 − Encoder Only", "d1", "g1cross_enc"), ("L1 − D1", "l1", "d1"), ("L2 − D1", "l2", "d1"),
            ("B0 − Encoder Only", "b0", "g1cross_enc"), ("D1 − Existing Decoder", "d1", "g1cross_dec"), ("Existing Decoder − Encoder Only", "g1cross_dec", "g1cross_enc"),
            ("L1 − Encoder Only", "l1", "g1cross_enc"), ("L2 − Encoder Only", "l2", "g1cross_enc")]
    if best:
        cons.append((f"Best candidate ({best.upper()}, by Dev) − MaskEmbed", best, "maskembed_full"))
        cons.append(("Encoder Only − MaskEmbed", "g1cross_enc", "maskembed_full"))
    KEYS = ["obj", "part", "obj_material", "part_pattern_marking", "obj_pattern_marking"]
    S["contrasts_val"] = {n: {k: ci(val, a, b, k) for k in KEYS + [f"{lv}_{t}" for lv in ("obj", "part") for t in TYPES]} for n, a, b in cons}
    S["contrasts_dev"] = {n: {k: ci(dev, a, b, k) for k in KEYS} for n, a, b in cons}
    S["val_points"], S["dev_points"] = pv, pd_

    def g(n, k):
        c = S["contrasts_val"].get(n, {}).get(k)
        return c

    # ---------------------------------------------------------------- answers (rule-based; CI = paired image bootstrap 95%)
    def verdict(c):
        if c is None:
            return "n/a"
        return "유의한 향상" if c["sig"] and c["delta"] > 0 else "유의한 하락" if c["sig"] else ("차이 없음(n.s., +)" if c["delta"] >= 0 else "차이 없음(n.s., −)")

    ans = {}
    if pv:
        ex_drop = [g("Existing Decoder − Encoder Only", k) for k in ("obj", "part")]
        d1_enc = [g("D1 − Encoder Only", k) for k in ("obj", "part")]
        ans["A"] = (f"기존 Decoder − Encoder Only: Obj {fci(ex_drop[0])}, Part {fci(ex_drop[1])}. "
                    f"D1 − Encoder Only: Obj {fci(d1_enc[0])} ({verdict(d1_enc[0])}), Part {fci(d1_enc[1])} ({verdict(d1_enc[1])}). "
                    f"D1 − 기존 Decoder: Obj {fci(g('D1 − Existing Decoder', 'obj'))}, Part {fci(g('D1 − Existing Decoder', 'part'))}.")
        ans["B"] = (f"B0 − Encoder Only: Obj {fci(g('B0 − Encoder Only', 'obj'))}, Part {fci(g('B0 − Encoder Only', 'part'))}; "
                    f"D1 − B0: Obj {fci(g('D1 − B0 Bilinear', 'obj'))}, Part {fci(g('D1 − B0 Bilinear', 'part'))}.")
        ans["C"] = "; ".join(f"{c.upper()} − Encoder Only: Obj {fci(g(n, 'obj'))}, Part {fci(g(n, 'part'))}"
                             for c, n in (("d1", "D1 − Encoder Only"), ("l1", "L1 − Encoder Only"), ("l2", "L2 − Encoder Only")))
        ans["D"] = "; ".join(f"{n}: Obj {fci(g(n, 'obj'))}, Part {fci(g(n, 'part'))}, Obj Material {fci(g(n, 'obj_material'))}, Part Pattern {fci(g(n, 'part_pattern_marking'))}"
                             for n in ("L1 − D1", "L2 − D1"))
        if "c0" in pv and best:
            gap_e = {k: pv["c0"][k] - pv["g1cross_enc"][k] for k in ("obj", "part", "obj_material", "part_pattern_marking")}
            gap_b = {k: pv["c0"][k] - pv[best][k] for k in gap_e}
            ans["E"] = ("C0 crop − Encoder Only (val): " + ", ".join(f"{k} {v:+.2f}" for k, v in gap_e.items())
                        + f"; C0 crop − {best.upper()}: " + ", ".join(f"{k} {v:+.2f}" for k, v in gap_b.items()))
        else:
            ans["E"] = "C0 crop val 예측 없음 → 격차는 기존 Test 수치로만 참고"
    S["answers_rule_based"] = ans
    # ---------------------------------------------------------------- document
    w("# Decoder Study on CLIP-L G1-cross — 종합 보고서 (자동 생성)")
    w("")
    w("> 이 파일은 큐가 끝날 때 `src/decstudy/report.py`가 자동으로 만든다. 수치는 모두 저장된 결과 파일에서 읽었고, 없는 값은 n/a로 둔다. "
      "\\* = image-level paired bootstrap(B=1000) 95% CI가 0을 포함하지 않음. Test는 실행하지 않았다.")
    w("")
    w("## 핵심 질문에 대한 답 (규칙 기반 요약, PACO 공식 Val Oracle AP)")
    for k in "ABCDE":
        q = dict(A="Decoder의 기존 성능 하락을 해결했는가?", B="Bilinear 입력 자체가 도움이 되었는가?", C="Decoder 사용 시 Encoder Only보다 실제로 높은 판별 AP를 얻었는가?",
                 D="추가 Loss 중 어떤 것이 실제 PACO AP 개선으로 연결되었는가?", E="Crop 재인코딩 없이 남은 세부 속성 판별 격차를 줄일 가능성이 확인되었는가?")[k]
        w(f"- **{k}. {q}** {ans.get(k, 'n/a')}")
    w("")
    # Table A
    def tab(points, title, note):
        w(f"### {title}")
        w("")
        w(note)
        w("")
        w("| Method | Obj AP | Obj Color | Obj Pattern | Obj Material | Obj Reflectance* | Part AP | Part Color | Part Pattern | Part Material | Part Reflectance* |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c, n in ROWS + REFS:
            p = points.get(c)
            if p is None:
                w(f"| {n} | " + " | ".join(["n/a"] * 10) + " |")
                continue
            w(f"| {n} | {f2(p['obj'])} | " + " | ".join(f2(p[f'obj_{t}']) for t in TYPES) + f" | {f2(p['part'])} | " + " | ".join(f2(p[f'part_{t}']) for t in TYPES) + " |")
        w("")
    w("## Table A. PACO 공식 Val — Official-protocol Oracle AP (GT box + category, R2 epoch 6 고정)")
    w("")
    tab(pv, "A-1. 공식 Val (2,410 images)", "Reflectance\\* = PACO 공식 type 이름 `transparency`(opaque / translucent / transparent / other). "
        "Existing Decoder 행은 기존 G1-cross 공동학습 decoder + 기존 R2 probe의 **epoch 6** checkpoint(기존 Test 보고 38.11/36.79는 M0로 고른 epoch 9)로 같은 Val 프로토콜에서 새로 평가했다.")
    tab(pd_, "A-2. Official-protocol Dev AP (PACO train 내부 dev 1,000 images)", "Dev bootstrap CI는 약 ±3 AP로 거칠다. Best candidate 선택에만 사용.")
    # Table B
    w("## Table B. 핵심 Delta (paired image bootstrap 95% CI)")
    w("")
    w("| Contrast | Val Obj | Val Part | Val Obj Material | Val Part Pattern | Val Obj Pattern | Dev Obj | Dev Part |")
    w("|---|---|---|---|---|---|---|---|")
    for n, a, b in cons:
        cv, cd = S["contrasts_val"][n], S["contrasts_dev"][n]
        w(f"| {n} | {fci(cv['obj'])} | {fci(cv['part'])} | {fci(cv['obj_material'])} | {fci(cv['part_pattern_marking'])} | {fci(cv['obj_pattern_marking'])} | {fci(cd['obj'])} | {fci(cd['part'])} |")
    w("")
    if best:
        w(f"Best candidate = Dev (Obj+Part)/2 최대인 {best.upper()} (Dev: " + ", ".join(f"{c.upper()} {v:.2f}" for c, v in dev_avg.items()) + "). Val은 선택에 쓰지 않았다.")
        w("")
    # Table C
    fd = jl(f"{RR}/featdiag/featdiag.json")
    cal = jl(f"{RR}/calib/weights.json")
    tl = {c: jsonl(f"{RR}/dec/{c}/train_log.jsonl") for c in ("D1", "L1", "L2")}
    w("## Table C. 학습 Loss 및 Feature 분석 (COCO dev ROIs, 최종 EMA decoder, C0 crop teacher 대비)")
    w("")
    if fd:
        w(f"COCO dev {fd['n_dev_images']} images / {fd['n_dev_rois']} ROIs (PACO val/test 일치 이미지 제외). 모든 경로가 같은 ROI와 같은 teacher token을 쓴다.")
        w("")
        w("| Path | train L_orig (last 100) | train L_cov | train L_rel | dev L_detail | dev token cos | dev pooled cos | ‖Δ‖/‖B‖ | cos(Δ, T−B) | spatial var ratio (Z/T) | cov err (L_cov) | cov cosine | ROI rel KL (L_rel) | ROI rel Pearson r | context-swap Δ change | context-swap cos-dist change |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, nm, tc in (("B0_bilinear", "B0 Bilinear", None), ("existing_g1cross_decoder", "Existing decoder", None), ("D1", "D1", "D1"), ("L1", "L1", "L1"), ("L2", "L2", "L2")):
            t = fd["table"].get(key, {})
            ex = fd["residual_and_context"].get(key, {})
            last = tl[tc][-1] if tc and tl[tc] else {}
            w(f"| {nm} | {f2(last.get('l_orig'))} | {f2(last.get('l_cov')) if tc == 'L1' else '–'} | {f2(last.get('l_rel')) if tc == 'L2' else '–'} | "
              f"{t.get('l_detail', 0):.4f} | {t.get('tok_cos', 0):.4f} | {t.get('pool_cos', 0):.4f} | {f2(t.get('delta_over_b')) if 'delta_over_b' in t else '–'} | "
              f"{f2(ex.get('dir_cos')) if ex.get('dir_cos') is not None else '–'} | {t.get('sv_ratio', 0):.3f} | {t.get('cov_err', 0):.4f} | {t.get('cov_sim', 0):.4f} | "
              f"{t.get('rel_kl', 0):.4f} | {t.get('rel_r', 0):.3f} | {f2(ex.get('ctx_rel_change')) if ex.get('ctx_rel_change') is not None else '–'} | "
              f"{ex.get('ctx_tok_cos_change'):+.4f} |" if ex.get("ctx_tok_cos_change") is not None else
              f"| {nm} | – | – | – | {t.get('l_detail', 0):.4f} | {t.get('tok_cos', 0):.4f} | {t.get('pool_cos', 0):.4f} | – | – | {t.get('sv_ratio', 0):.3f} | {t.get('cov_err', 0):.4f} | {t.get('cov_sim', 0):.4f} | {t.get('rel_kl', 0):.4f} | {t.get('rel_r', 0):.3f} | – | – |")
        w("")
        w("- train L_orig = 1.11 × L_detail (G1-cross decoder 목적식 그대로). spatial var ratio = ROI 내부 token 분산(student/teacher). cov err = 정규화 ‖C_Z−C_T‖²/‖C_T‖² (L2-normalised token). "
          "ROI rel = 같은 batch(16 images) ROI pooled vector cosine 행렬의 KL(τ=0.1) / Pearson r. context-swap = ROI box와 겹치지 않는 key token을 다른 이미지 것으로 바꿨을 때 decoder 출력(Δ 또는 Z)의 상대 변화 / teacher와의 token cosine 거리 증가량(>0 = 실제 문맥이 도움).")
        w("")
    if cal:
        w(f"λ 고정값(학습 전 16 step, init gradient-norm 비율의 median, 1회): λ_detail {cal['lam_detail']} (G1-cross 그대로), λ_cov **{cal['lam_cov']}**, λ_rel **{cal['lam_rel']}** "
          f"(τ={cal['tau']}, IQR cov {np.round(cal['ratio_cov_iqr'], 3).tolist()}, rel {np.round(cal['ratio_rel_iqr'], 3).tolist()}; init loss: L_orig {cal['loss_orig_init_median']:.3f}, L_cov {cal['loss_cov_init_median']:.3f}, L_rel {cal['loss_rel_init_median']:.4f}; "
          f"teacher ROI-ROI cosine 평균 {cal['teacher_rel_offdiag_mean']:.3f}±{cal['teacher_rel_offdiag_std']:.3f}, softmax entropy/log(n−1) {cal['teacher_softmax_entropy_over_log_n']:.3f}).")
        w("")
    # diag curves
    dg = jsonl(f"{RR}/dec/diag.jsonl")
    if dg:
        w("학습 중 COCO dev 진단(EMA):")
        w("")
        w("| step | model | token cos | pooled cos | ‖Δ‖/‖B‖ | var ratio | cov err | rel KL |")
        w("|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in dg:
            w(f"| {r['step']} | {r['model']} | {r['tok_cos']:.4f} | {r['pool_cos']:.4f} | {f2(r.get('delta_over_b'))} | {r['sv_ratio']:.3f} | {r['cov_err']:.4f} | {r['rel_kl']:.4f} |")
        w("")
    # Table D
    fa = jl(f"{RR}/failure/failure.json") or jl(f"{RR}/failure_prelim/failure.json")
    w("## Table D. 공통 실패 분석 (PACO Val, pairwise ranking — 공식 AP 대체 아님)")
    w("")
    if fa:
        w(f"규칙: {fa['rule']}; unit당 최대 {fa['K']} pair, seed {fa['seed']}. 사용 가능 모델: " + ", ".join(f"{fa['models'][k]} {'O' if v else 'X'}" for k, v in fa["available"].items()))
        w("")
        w("| Group | units | pairs | CLIP Full | G1-cross Enc | MaskEmbed | C0 crop | 두 Full 공통 실패 | Crop만 구별 | 공통 실패 중 Crop 정답 | 반복 unit (n≥5, crop-only≥30%) | ROI side@336 median (전체 / crop-only) | crop-only rate small / large |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for gn, gg in fa["groups"].items():
            pa = gg["pair_acc"]
            p = lambda k: f"{100 * pa[k]:.1f}" if k in pa else "n/a"  # noqa: E731
            co = gg.get("crop_only_rate")
            w(f"| {gn} | {gg['n_units']} | {gg['n_pairs']} | {p('clip_full')} | {p('g1cross_enc')} | {p('maskembed_full')} | {p('clip_c0')} | {100 * gg['common_full_fail_rate']:.1f}% | "
              + (f"{100 * co:.1f}% | {100 * gg['crop_correct_given_common_fail']:.1f}% | {gg['recurrent_units(n>=5, crop_only_rate>=0.3)']}/{gg['units_with_pairs_ge5']} | "
                 f"{gg['roi_side_336_all_pairs']['median']:.1f} / {gg['roi_side_336_crop_only']['median']:.1f} px | "
                 f"{100 * gg['crop_only_rate_by_size']['small']:.1f}% / {100 * gg['crop_only_rate_by_size']['large']:.1f}% (side<{gg['crop_only_rate_by_size']['threshold_side336']:.0f}px) |"
                 if co is not None else "n/a | n/a | n/a | " + f"{gg['roi_side_336_all_pairs']['median']:.1f} / n/a | n/a |"))
        w("")
        for gn, gg in fa["groups"].items():
            if gg.get("crop_only_by_attribute"):
                w(f"- **{gn}** crop-only 속성 분포: " + ", ".join(f"{x['attribute']} {x['n']}/{x['of_pairs']}" for x in gg["crop_only_by_attribute"][:8])
                  + "; 상위 category: " + ", ".join(f"{x['category']} {x['n']}/{x['of_pairs']}" for x in gg["crop_only_by_category_top10"][:6]))
        w("")
        fdir = "failure" if jl(f"{RR}/failure/failure.json") else "failure_prelim"
        if fa.get("viz"):
            w(f"대표 사례 {len(fa['viz'])}개 (왼쪽: 실제 입력 336 letterbox full image + box, 오른쪽: 원본 tight crop; 초록 = positive, 빨강 = negative). 이미지: `{RR}/{fdir}/viz/`")
            w("")
            for v in fa["viz"]:
                sc = ", ".join(f"{fa['models'][k].replace('Original ', '')} {p_:.2f}/{n_:.2f}" for k, (p_, n_) in v["scores_pos_neg"].items())
                w(f"{v['i'] + 1}. {v['group']} — **{v['category']} / {v['attribute']}** — pos/neg 점수: {sc}; ROI side@336 pos {v['pos']['box_side_336']:.0f}px / neg {v['neg']['box_side_336']:.0f}px")
                w(f"   ![pos](../{RR}/{fdir}/{v['pos_img']}) ![neg](../{RR}/{fdir}/{v['neg_img']})")
            w("")
    else:
        w("n/a (failure analysis 결과 없음)")
        w("")
    # sanity
    w("## Sanity checks")
    w("")
    cpu = jl(f"{RR}/sanity/checks_cpu.json")
    cfg = jl(f"{RR}/dec/config.json")
    dn = {c: jl(f"{RR}/dec/{c}/done.json") for c in ("D1", "L1", "L2")}
    gc = {c: jsonl(f"{RR}/dec/{c}/grad_check.jsonl") for c in ("D1", "L1", "L2")}
    inf = {c: jl(f"{RR}/preds/{c}/infer_val.json") for c, _ in ROWS + [("c0", "")]}
    pr = {c: jl(f"{RR}/probes/{c}/config.json") for c in ("b0", "d1", "l1", "l2", "c0")}
    prd = {c: jl(f"{RR}/probes/{c}/done.json") for c in ("b0", "d1", "l1", "l2", "c0")}
    fz = {c: jl(f"{RR}/probes/{c}/frozen_check.json") for c in ("b0", "d1", "l1", "l2", "c0")}
    rows = []
    if inf.get("g1cross_enc"):
        hs = {c: (inf[c]["path"].get("ckpt_sha256") if inf.get(c) else None) for c, _ in ROWS}
        rows.append(("동일 Encoder Checkpoint (sha256, 6개 경로)", "OK" if len({h for h in hs.values() if h}) == 1 else f"MISMATCH {hs}", f"{RR}/preds/*/infer_val.json"))
    if cfg and dn["D1"]:
        rows.append(("Encoder weight hash 학습 전/후 + freeze", f"before {cfg['encoder']['enc_state_sha256'][:12]}… after {dn['D1']['encoder_hash_after'][:12]}… unchanged={dn['D1']['encoder_unchanged']}, grads none={dn['D1']['encoder_grads_none']}", "dec/*/done.json"))
        rows.append(("Teacher 동일 (원본 CLIP-L/336, hash 불변)", f"{cfg['encoder']['teacher_state_sha256'][:12]}… unchanged={dn['D1']['teacher_unchanged']}", "dec/config.json"))
        rows.append(("세 decoder 동일 초기 상태", f"init sha256 {cfg['decoder_init_sha256'][:12]}… (assert 동일)", "dec/config.json"))
    if cpu:
        al = cpu["alignment"]
        rows.append(("Crop letterbox ↔ Bilinear grid 정렬", f"{al['n_rois']} ROI 최대 오차 {al['max_px_error']:.1e}px; 실제 픽셀 상관 median {al['pixel_corr_median']:.3f} (min {al['pixel_corr_min']:.3f}); 합성 좌표맵 오차 {cpu['synthetic_coordinate_map_max_err_px']:.1e}px", "sanity/checks_cpu.json"))
        rows.append(("Decoder Query/Output shape", f"full {cpu['shapes']['full_tokens']} → B, Δ, Z {cpu['shapes']['B']} (R×576×1024), teacher {cpu['shapes']['teacher']}", "sanity"))
        rows.append(("Residual zero-init", f"Δ max|·| = {cpu['init_delta_max_abs_amp0']} (fp32), {cpu['init_delta_max_abs_amp1']} (bf16); Z==B {cpu['init_z_equals_b_amp1']}; 학습 step 0에서도 assert", "sanity + dec/*/grad_check.jsonl"))
        rows.append(("Padding token 제외", f"padding cell({100 * cpu['padding_cells_fraction']:.0f}%)을 noise로 바꿔도 L_detail/L_cov/L_rel 변화 {cpu['padding_invariance_max_abs']}; R2는 cov=0 cell noise에 logit 변화 0 (아래)", "sanity"))
        rows.append(("Loss 구현 검증", f"L_cov vs torch.cov 상대오차 {cpu['cov_vs_naive_max_rel_err']:.1e}; L_rel vs naive {cpu['rel_vs_naive_max_abs']:.1e}; self=0", "sanity"))
    if any(gc.values()):
        s_ = []
        for c, gg in gc.items():
            g1 = [r for r in gg if r.get("step") == 1 and "g_orig" in r]
            g11 = [r for r in gg if r.get("step") == 11 and "g_orig" in r]
            lastg = [r for r in gg if "g_total" in r][-1:] if gg else []
            if g1 and g11:
                extra = "".join(f", {k} {g11[0].get('g_' + k + '_body', 0):.2e}" for k in ("cov", "rel") if f"g_{k}" in g11[0])
                s_.append(f"{c}: step1 body {g1[0]['g_orig_body']:.1e} / out {g1[0]['g_orig_out']:.1e}; step11 body orig {g11[0]['g_orig_body']:.2e}{extra}")
        rows.append(("Loss별 gradient 전달", "; ".join(s_), "dec/*/grad_check.jsonl"))
    if inf.get("d1"):
        rows.append(("R2는 cov>0 cell만 읽음", ", ".join(f"{c} {inf[c]['checks']['outside_max_abs']['max']:.1e}" for c, _ in ROWS if inf.get(c) and inf[c].get("checks")), "infer_val.json checks"))
        rows.append(("Decoder 출력/Δ 실제 사용", ", ".join(f"{c}: zero→|Δlogit| {inf[c]['checks']['dec_zero_mean_abs']['mean']:.2f}" + (f", Δ 제거→ {inf[c]['checks']['delta_removed_mean_abs']['mean']:.3f}" if inf[c]['checks'].get('delta_removed_mean_abs') else "")
                                                 for c in ("g1cross_dec", "b0", "d1", "l1", "l2") if inf.get(c) and inf[c].get("checks")), "infer_val.json"))
        rows.append(("Full image 인코딩 횟수 / ROI RGB crop 재인코딩", ", ".join(f"{c} {inf[c]['full_image_encodings_per_image']:.0f}회/{inf[c]['rgb_crop_encodings_per_roi']:.0f}" for c, _ in ROWS if inf.get(c)) + "  (C0 oracle만 ROI당 1회)", "infer_val.json"))
    if fd:
        rows.append(("Decoder가 Full image 문맥 참조", ", ".join(f"{k}: {v['ctx_rel_change']:.3f}" for k, v in fd["residual_and_context"].items() if v.get("ctx_rel_change") is not None) + " (out-of-box key 교체 시 출력 상대 변화)", "featdiag"))
    if any(pr.values()):
        bud = {c: (p["total_steps"], p["steps_per_epoch"][:1], p["n_train_rows"]) for c, p in pr.items() if p}
        rows.append(("R2 학습 예산 동일", "; ".join(f"{c}: schedule {v[0]} steps, ep {v[1]}, rows {v[2]}, 사용 epoch 6" for c, v in bud.items()) + " (기존 g1cross probes: 동일 yaml, ep6 ckpt)", "probes/*/config.json"))
        rows.append(("Probe 중 frozen 경로 불변", ", ".join(f"{c} {v['unchanged']}" for c, v in fz.items() if v), "frozen_check.json"))
    rows.append(("공식 Evaluator 동일", "src.pacoattr.official_eval (PACO @cdefd31, lvis-api @7d7f07d, 수정 없음) — 모든 val 행; dev: src.paco_devap.official_dev", ""))
    if cfg:
        rows.append(("학습 이미지 / ROI", f"COCO stream 12,000 steps × 16 images × 4 ROIs (PACO val/test 일치 {len(cfg['excluded_stream_images'])}장 제외, 고유 이미지 {cfg['n_stream_images_unique']}), world 1, 64 ROI/step; PACO attribute label 미사용", "dec/config.json"))
    if dn["D1"]:
        rows.append(("Decoder 학습 GPU/Step/Runtime/Peak VRAM", f"{dn['D1']['gpu']} 1장(물리 GPU 2), 12,000 step, 세 decoder 동시 {dn['D1']['time_s'] / 3600:.2f} h, peak {dn['D1']['peak_mem_gb']} GB", "dec/*/done.json"))
    if any(prd.values()):
        rows.append(("Probe runtime", ", ".join(f"{c} {v['time_s'] / 60:.0f} min" for c, v in prd.items() if v), "probes/*/done.json"))
    q = jl(f"{RR}/qat_pause/paused.json")
    if q:
        rows.append(("기존 작업 충돌", f"GPU 0,1 미사용. QAT(GPU 2,3) 사용자 지시로 즉시 중단: step {q.get('step_at_stop')} → checkpoint-{q.get('checkpoint_step')}에서 재개 (손실 {q.get('unsaved_steps_lost')} step). 재개 상태: {RR}/qat_resume_status.md", "qat_pause/"))
    w("| 항목 | 결과 | 근거 |")
    w("|---|---|---|")
    for a_, b_, c_ in rows:
        w(f"| {a_} | {b_} | {c_} |")
    w("")
    w("추론 시 GT ROI는 Oracle 진단에만 쓴다. 이 보고서의 어떤 경로도 GT box를 필수 입력으로 요구하는 최종 방법이라고 주장하지 않는다.")
    w("")
    # queue status
    qs = os.path.join(ROOT, RR, "queue_status.tsv")
    if os.path.exists(qs):
        w("## Queue status")
        w("")
        w("```")
        w(open(qs).read().strip())
        w("```")
    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.join(ROOT, RR, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    open(os.path.join(ROOT, RR, "results", "decoder_study_summary_auto.md"), "w").write(txt)
    if not os.path.exists(os.path.join(ROOT, "results", "decoder_study_summary.md")) or "(자동 생성)" in open(os.path.join(ROOT, "results", "decoder_study_summary.md")).read():
        open(os.path.join(ROOT, "results", "decoder_study_summary.md"), "w").write(txt)
    json.dump(S, open(os.path.join(ROOT, RR, "results", "summary.json"), "w"), indent=1, default=float)
    print(txt)


if __name__ == "__main__":
    main()
