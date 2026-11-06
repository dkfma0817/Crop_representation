# Shared-feature readout queue: report (PACO VAL only, seed 0, GPU 2+3 DDP)

Run: `output/2026-11-04/readout_queue`. Primary = val M2 Overall at the last epoch vs the queue baseline **Multi-layer + R2 (blocks 4/8/12)** (`ctrl_f336/ml_r2`, defined for this queue, not a reproduction of the 11-04 run). Paired image bootstrap 95% CI, 1,000 reps. PACO test was not evaluated. Object 5-16 is not reported (official M2 on val is NaN).

## Data check

- verified disjoint after exclusion: **True**; train images excluded: 6 (hits by key {'image_id': 0, 'md5': 1, 'pix_md5': 1, 'dhash': 6}); left after exclusion {'image_id': 0, 'md5': 0, 'pix_md5': 0, 'dhash': 0}
- queue reads: paco_train (training + E12 C0 teacher crops), paco_val (evaluation); F672 teacher = the same PACO train images (no other source)

## Controls

| run | status | val M2 Overall | Object | Part | Color | Material | Pattern | train-subset M2 |
|---|---|---|---|---|---|---|---|---|
| ctrl_f336/ml_r2 | done | 46.23 | 47.02 | 45.45 | 46.22 | 38.88 | 44.89 | 67.21 |
| ctrl_f336/r2_final | done | 42.23 | 41.58 | 42.89 | 36.43 | 35.86 | 48.40 | 76.56 |
| ctrl_c0/c0_r2 | done | 45.54 | 48.03 | 43.04 | 40.81 | 38.16 | 53.96 | 84.84 |
| ctrl_f672/f672_ml_r2 | done | 49.03 | 48.87 | 49.18 | 48.42 | 40.83 | 50.42 | 70.19 |

**D0**: slot 1 = **E08** (binding problem confirmed = False); BD(ml_r2) +0.76 [+0.35, +1.16] pp; BD gap vs C0 -0.07 [-0.48, +0.33]; score gap (PA_x) vs C0 +0.86 [+0.46, +1.26]. Details: `D0/d0.md`.

## E08 — Weight-sharing Repeated Readout

**Hypothesis.** Reading the same shared features again with the same cross-attention finds information a single read missed.

**Training.** One cross block: x1 reads once; x3 reads 3 times with shared weights; ffn3 reads once + 2 extra passes of the same block's FFN (same params and extra compute, no re-reading). Loss on the final output only; every step recorded.

| reader | status | epochs | train loss (ep1 → last) | val loss (ep0 → last) | val M2 by epoch | best-val epoch* | train-subset loss / M2 |
|---|---|---|---|---|---|---|---|
| e08_x1 | done | 10 | 0.1223 → 0.0652 | 0.7352 → 0.0925 | [13.62, 39.3, 42.55, 42.9, 44.13, 44.86, 43.6, 44.92, 45.21, 44.71, 44.66] | 8 | 0.0682 / 66.87 |
| e08_x3 | done | 10 | 0.1228 → 0.0636 | 0.7309 → 0.0914 | [13.61, 40.81, 42.57, 43.04, 43.85, 44.6, 45.0, 45.48, 44.82, 44.85, 45.06] | 7 | 0.0665 / 66.25 |
| e08_ffn3 | done | 10 | 0.1218 → 0.064 | 0.7399 → 0.0925 | [13.77, 39.14, 42.24, 42.91, 43.94, 44.38, 44.49, 45.91, 45.0, 44.9, 44.94] | 7 | 0.0666 / 66.12 |

*best-val epoch is shown for information only; all numbers below use the last epoch (no selection on val).

**Direct check** (rule: x3 - x1 CI lo > 0 AND x3 - ffn3 CI lo > 0 (val M2 Overall))

| measure | value |
|---|---|
| e08_x1 step 1 val M2 Overall | 44.66 |
| e08_x3 step 1 val M2 Overall | 39.75 |
| e08_x3 step 2 val M2 Overall | 44.45 |
| e08_x3 step 3 val M2 Overall | 45.06 |
| e08_ffn3 step 1 val M2 Overall | 45.31 |
| e08_ffn3 step 2 val M2 Overall | 44.98 |
| e08_ffn3 step 3 val M2 Overall | 44.94 |
| x3 - x1 (Overall) | +0.40 [-1.08, +1.67] |
| x3 - ffn3 (Overall) | +0.12 [-1.70, +1.38] |
| x3 - x1 (Part) | +0.53 [-1.28, +1.81] |
| x3 - ffn3 (Part) | -0.40 [-2.14, +1.49] |

**Downstream** (e08_x3 vs ml_r2, val M2, last epoch)

| metric | baseline | method | Δ [95% CI] |
|---|---|---|---|
| overall | 46.23 | 45.06 | -1.17 [-2.91, +0.02] |
| object | 47.02 | 45.27 | -1.76 [-3.93, -0.25] |
| part | 45.45 | 44.85 | -0.59 [-2.55, +0.92] |
| color | 46.22 | 45.81 | -0.41 [-2.58, +1.48] |
| pattern_marking | 44.89 | 43.73 | -1.16 [-5.55, +1.92] |
| material | 38.88 | 37.61 | -1.27 [-4.27, +0.54] |
| transparency | 54.95 | 53.10 | -1.85 [-4.80, +1.08] |

**Verdict**

- Training and evaluation completed normally: **True**
- Expected ability shown in the direct check: **False**
- Val M2 Overall improved over Multi-layer + R2 (CI lo > 0): **False**
- Gain concentrated in one attribute type: **False (top None, share n/a)**
- Major degradation of another metric: **['object']**
- Downstream success (pre-registered): **False** — recorded separately from the hypothesis check.

## E09 — Query-conditioned Adaptive Sampling

**Hypothesis.** With the same number of feature samples (36), choosing WHERE to sample from the query improves discrimination.

**Training.** ml_r2 + 36 bilinear samples of the same fused F336 map per block (box coords in [-0.25, 1.25]); ads predicts the points from the query (initialised on the grid), fix keeps the grid. No new pixels, no crop encoding.

| reader | status | epochs | train loss (ep1 → last) | val loss (ep0 → last) | val M2 by epoch | best-val epoch* | train-subset loss / M2 |
|---|---|---|---|---|---|---|---|
| e09_ads | done | 10 | 0.121 → 0.0694 | 0.725 → 0.0895 | [14.48, 39.38, 41.84, 42.19, 43.81, 43.89, 43.98, 45.19, 45.93, 45.55, 45.41] | 8 | 0.0729 / 60.37 |
| e09_fix | done | 10 | 0.1215 → 0.0646 | 0.7682 → 0.0913 | [14.55, 38.57, 42.6, 43.11, 44.0, 43.51, 45.13, 45.77, 44.96, 44.98, 44.89] | 7 | 0.0677 / 65.23 |

*best-val epoch is shown for information only; all numbers below use the last epoch (no selection on val).

**Direct check** (rule: ads - fix val M2 Overall CI lo > 0 (point movement reported alongside))

| measure | value |
|---|---|
| ads - fix (Overall) | +0.52 [-0.83, +1.87] |
| ads - fix (Part) | +0.84 [-0.99, +2.33] |
| ads - fix (Object) | +0.20 [-1.58, +2.25] |
| block 0: mean |ads point - grid| (box units) | 0.338 |
| block 0: fraction of ads points outside the box | 0.040 |
| block 0: mean distance to box centre ads / fix | 0.298 / 0.379 |
| block 1: mean |ads point - grid| (box units) | 1.112 |
| block 1: fraction of ads points outside the box | 0.999 |
| block 1: mean distance to box centre ads / fix | 1.033 / 0.379 |
| fraction of last-block points on the instance mask ads / fix | 0.006 / 0.528 |

**Downstream** (e09_ads vs ml_r2, val M2, last epoch)

| metric | baseline | method | Δ [95% CI] |
|---|---|---|---|
| overall | 46.23 | 45.41 | -0.82 [-2.22, +0.68] |
| object | 47.02 | 45.22 | -1.80 [-3.57, +0.61] |
| part | 45.45 | 45.61 | +0.16 [-1.95, +1.70] |
| color | 46.22 | 47.70 | +1.47 [-0.75, +3.25] |
| pattern_marking | 44.89 | 44.24 | -0.64 [-4.13, +2.12] |
| material | 38.88 | 36.41 | -2.47 [-4.97, +0.40] |
| transparency | 54.95 | 53.31 | -1.64 [-4.52, +2.43] |

**Verdict**

- Training and evaluation completed normally: **True**
- Expected ability shown in the direct check: **False**
- Val M2 Overall improved over Multi-layer + R2 (CI lo > 0): **False**
- Gain concentrated in one attribute type: **True (top color, share 1.00)**
- Major degradation of another metric: **none**
- Downstream success (pre-registered): **False** — recorded separately from the hypothesis check.

## E11 — Local/Context Gated Readout

**Hypothesis.** Reading local and context evidence separately and gating them uses context better than one joint read of the same tokens.

**Training.** concat (primary control): one stream, 2 blocks over [ROI tokens ; all tokens]; gate: local stream (ROI tokens) and context stream (all tokens) + learnable per-channel gate; ctx: context-only ablation. Same tokens, same output dim.

| reader | status | epochs | train loss (ep1 → last) | val loss (ep0 → last) | val M2 by epoch | best-val epoch* | train-subset loss / M2 |
|---|---|---|---|---|---|---|---|
| e11_concat | done | 10 | 0.1214 → 0.0617 | 0.7654 → 0.0931 | [13.97, 39.18, 42.61, 42.82, 44.32, 44.73, 45.02, 45.89, 45.78, 45.51, 45.31] | 7 | 0.0643 / 67.89 |
| e11_gate | done | 10 | 0.1188 → 0.0589 | 0.7375 → 0.0942 | [13.86, 40.33, 41.44, 43.12, 44.39, 44.4, 45.77, 46.64, 45.99, 45.45, 45.35] | 7 | 0.0613 / 70.67 |
| e11_ctx | done | 10 | 0.1221 → 0.0622 | 0.7675 → 0.0922 | [14.31, 39.24, 42.52, 43.28, 45.79, 45.5, 45.47, 45.96, 46.27, 45.75, 45.64] | 8 | 0.0648 / 68.24 |

*best-val epoch is shown for information only; all numbers below use the last epoch (no selection on val).

**Direct check** (rule: gate - concat Part M2 CI lo > 0 AND removing context lowers gate's Part M2 (CI lo > 0))

| measure | value |
|---|---|
| gate - concat (Overall) | +0.04 [-1.67, +1.41] |
| gate - concat (Part) | -0.61 [-2.74, +1.02] |
| gate - concat (Object) | +0.70 [-1.63, +2.76] |
| ctx-only - concat (Overall) | +0.33 [-1.49, +1.65] |
| e11_gate: M2 drop when context removed (Overall / Part) | +5.04 [+3.15, +6.35] / +6.21 [+3.73, +8.15] |
| e11_gate: mean |dp| when context removed | 0.0196 |
| e11_gate: M2 drop when context swapd (Overall / Part) | -0.03 [-0.65, +0.84] / -0.51 [-1.34, +0.81] |
| e11_gate: mean |dp| when context swapd | 0.0051 |
| e11_concat: M2 drop when context removed (Overall / Part) | +0.48 [-0.46, +1.13] / +0.23 [-0.72, +1.23] |
| e11_concat: mean |dp| when context removed | 0.0063 |
| e11_concat: M2 drop when context swapd (Overall / Part) | +0.25 [-0.58, +0.87] / +0.12 [-0.56, +1.17] |
| e11_concat: mean |dp| when context swapd | 0.0057 |
| e11_ctx: M2 drop when context removed (Overall / Part) | -0.15 [-1.16, +1.06] / +0.67 [-0.55, +2.13] |
| e11_ctx: mean |dp| when context removed | 0.0095 |
| e11_ctx: M2 drop when context swapd (Overall / Part) | +0.02 [-0.79, +0.76] / +0.17 [-0.72, +1.22] |
| e11_ctx: mean |dp| when context swapd | 0.0063 |
| mean local gate g [object n_in 1-16] (1 = local only) | 0.659 |
| mean local gate g [object 17+] (1 = local only) | 0.675 |
| mean local gate g [part] (1 = local only) | 0.643 |

**Downstream** (e11_gate vs ml_r2, val M2, last epoch)

| metric | baseline | method | Δ [95% CI] |
|---|---|---|---|
| overall | 46.23 | 45.35 | -0.88 [-2.60, +0.46] |
| object | 47.02 | 45.70 | -1.32 [-3.54, +0.52] |
| part | 45.45 | 45.00 | -0.45 [-2.69, +1.19] |
| color | 46.22 | 46.17 | -0.06 [-2.43, +1.74] |
| pattern_marking | 44.89 | 45.05 | +0.17 [-3.72, +2.94] |
| material | 38.88 | 38.11 | -0.77 [-4.12, +2.80] |
| transparency | 54.95 | 52.08 | -2.87 [-6.08, -0.11] |

**Verdict**

- Training and evaluation completed normally: **True**
- Expected ability shown in the direct check: **False**
- Val M2 Overall improved over Multi-layer + R2 (CI lo > 0): **False**
- Gain concentrated in one attribute type: **True (top pattern_marking, share 1.00)**
- Major degradation of another metric: **['transparency']**
- Downstream success (pre-registered): **False** — recorded separately from the hypothesis check.

## E12 — Scale Decision Consistency

**Hypothesis.** Training the F336 reader to agree with the decisions of a 672-encoded view of the SAME 336 image (no new pixels) reduces the encoding-scale discrimination gap; inference uses F336 only.

**Training.** Student = ml_r2 on F336, BCE + 1.0 x BCE(student, sigmoid(teacher)) on all 59 attributes of the same real ROI. Teacher = ctrl_f672/f672_ml_r2 (336 canvas upsampled x2 -> DINOv2 672, same reader). Control = ml_r2 (BCE only).

| reader | status | epochs | train loss (ep1 → last) | val loss (ep0 → last) | val M2 by epoch | best-val epoch* | train-subset loss / M2 |
|---|---|---|---|---|---|---|---|
| e12_cons | done | 10 | 0.1165 → 0.0619 | 0.7677 → 0.0884 | [14.75, 41.39, 43.56, 44.24, 44.8, 45.91, 46.15, 46.79, 46.88, 46.88, 46.74] | 9 | 0.0647 / 68.28 |

*best-val epoch is shown for information only; all numbers below use the last epoch (no selection on val).

**Direct check** (rule: student closer to the teacher than the baseline (CI hi < 0) AND gap closure > 0)

| measure | value |
|---|---|
| teacher F672 - baseline F336 (val M2 Overall; does the teacher discriminate better?) | +2.79 [+0.96, +4.77] |
| teacher F672 val M2 Overall / Part / Object | 49.03 / 49.18 / 48.87 |
| student - baseline (val M2 Overall) | +0.50 [-1.04, +1.73] |
| student - baseline (Part / Object) | +1.19 [-0.66, +2.48] / -0.18 [-2.18, +1.62] |
| disagreement with teacher: BCE(student) - BCE(baseline) (lower = closer) | -0.0028 [-0.0031, -0.0026] |
| gap closure = d(student) / d(teacher) | 0.180 |

**Downstream** (e12_cons vs ml_r2, val M2, last epoch)

| metric | baseline | method | Δ [95% CI] |
|---|---|---|---|
| overall | 46.23 | 46.74 | +0.50 [-1.04, +1.73] |
| object | 47.02 | 46.84 | -0.18 [-2.18, +1.62] |
| part | 45.45 | 46.63 | +1.19 [-0.66, +2.48] |
| color | 46.22 | 46.57 | +0.34 [-1.60, +2.23] |
| pattern_marking | 44.89 | 47.78 | +2.89 [-0.93, +5.30] |
| material | 38.88 | 38.61 | -0.27 [-3.61, +2.59] |
| transparency | 54.95 | 54.00 | -0.95 [-3.29, +1.94] |

**Verdict**

- Training and evaluation completed normally: **True**
- Expected ability shown in the direct check: **True**
- Val M2 Overall improved over Multi-layer + R2 (CI lo > 0): **False**
- Gain concentrated in one attribute type: **True (top pattern_marking, share 0.89)**
- Major degradation of another metric: **none**
- Downstream success (pre-registered): **False** — recorded separately from the hypothesis check.

