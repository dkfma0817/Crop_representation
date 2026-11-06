# S-series spatial ROI decoder pilot (COCO pre-training -> frozen PACO R2 probe)

PACO VAL only (test never evaluated), R2 epoch by PACO dev (1,000 train images held out), primary = val **M0 Overall**, paired image bootstrap 95% CI. References come from the earlier run with the identical probe protocol (`output/2026-11-05/msd`). Direct checks: 700 PACO val images (report only). Cache unchanged in every direct-check forward: **True**.

**Common loss / recipe.** Same as H1: L12 + 0.5 L8 + 0.5 L4, L_d = coverage-weighted [mean_c (p - t)^2 + (1 - cos)] on per-depth teacher-normalised features, teacher = frozen DINOv2 on the original-pixel COCO GT crop (C0), p_4 / p_8 from aux heads (removed for R2). COCO 12k stream (64 ROIs/step), AdamW 1e-4 (0.9, 0.95), wd 0.05, warmup 600 + cosine, clip 1.0, seed 0, frozen encoder, checkpoint = best COCO-dev L12. Output Z = Bilinear(F12) + sd12 * Linear_out(h) with Linear_out zero-init (all four).

## Pair A: S1 Hierarchical vs Flat (A1 vs A0)

**Hypothesis.** Coarse-to-fine refinement (4x4 -> 8x8 -> 16x16 -> 24x24) builds ROI structure first and adds detail later, giving more spatially varied, teacher-like ROI maps than the same blocks run flat at 24x24.

**Decoder.** A0: 4 blocks [SA, CA to ROI+ring cache tokens, FFN x4], d 384, all at 24x24. A1: the same 4 blocks at 4/8/16/24, bilinear state upsampling + stage grid input between stages. Same params (10.71M w/o aux heads).

**Training / checks**

| cond | trainable (decoder incl. aux) | dev step | dev L12 | dev L4 | dev L8 | output shape | cache unchanged | trunk grad step1 / step2 | aux-only grad into trunk | weight change trunk | time | peak mem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 11,890,176 | 12000 | 1.1868 | 1.3324 | 1.3742 | [32, 576, 768] | True | 0.0e+00 / 2.0e-05 | 0.122 | 32.99 | 14 min | 4.6 GB |
| A1 | 11,890,176 | 12000 | 1.1811 | 1.3427 | 1.3849 | [32, 576, 768] | True | 0.0e+00 / 1.1e-05 | 0.126 | 35.51 | 18 min | 4.0 GB |

**Direct checks** (normalised block-12 space; teacher = C0 crop; ratios pred / teacher, 1 = teacher-like)

| cond | L12 | var ratio | eff. rank (ratio) | neighbour-diff ratio | centred cos | affinity gap | L12 obj 1-16 | L12 part | L12 patterned | var ratio obj 1-16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bilinear (ref) | 1.844 | 0.321 | 20.0 (0.137) | 0.191 | 0.227 | 0.384 | 1.967 | 1.951 | 1.619 | 0.259 |
| H1 (CoordMLP, ref) | 1.236 | 0.142 | 56.1 (0.384) | 0.185 | 0.313 | 0.313 | 1.338 | 1.250 | 1.213 | 0.100 |
| A0 flat | 1.352 | 0.304 | 44.9 (0.309) | 0.253 | 0.316 | 0.300 | 1.456 | 1.401 | 1.254 | 0.237 |
| A1 hierarchical | 1.339 | 0.282 | 40.1 (0.277) | 0.241 | 0.312 | 0.305 | 1.445 | 1.386 | 1.246 | 0.214 |

**Stage-wise spatial representation** (A: output head on each block state, var / eff-rank ratio to the teacher)

| cond | stage1_var | stage1_erank | stage2_var | stage2_erank | stage3_var | stage3_erank | stage4_var | stage4_erank |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.496 | 0.232 | 0.499 | 0.240 | 0.417 | 0.270 | 0.304 | 0.309 |
| A1 | 0.448 | 0.166 | 0.469 | 0.188 | 0.377 | 0.236 | 0.282 | 0.277 |

**PACO** (val at the dev-selected epoch)

| probe | epoch | **M0 Overall** | Object | Part | Color | Material | Pattern | M2 | M0 n_in 1-4 | 5-16 | 17+ | frozen check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 flat | 6 | **43.21** | 45.19 | 41.37 | 25.46 | 73.09 | 44.55 | 44.22 | 47.40 | 48.64 | 54.14 | ok |
| A1 hierarchical | 6 | **43.53** | 45.40 | 41.79 | 26.12 | 73.30 | 44.43 | 44.68 | 47.27 | 48.91 | 55.01 | ok |
| H1 (CoordMLP, ref) | 8 | **43.67** | 46.01 | 41.51 | 26.87 | 73.56 | 43.08 | 42.82 | 46.68 | 49.37 | 54.43 | ok |
| Bilinear (ref) | 6 | **42.30** | 44.80 | 39.98 | 24.18 | 73.72 | 42.70 | 42.47 | 46.65 | 47.50 | 53.65 | ok |
| Full ML+R2 (ref) | 7 | **45.02** | 47.11 | 43.08 | 29.92 | 72.99 | 42.28 | 45.51 | 49.32 | 51.56 | 56.35 | n/a |
| GT crop C0 (ref) | 5 | **45.74** | 49.77 | 42.01 | 28.02 | 72.60 | 51.22 | 47.19 | 47.50 | 53.37 | 56.80 | n/a |

| contrast | M0 Overall | Object | Part | Color | Material | Pattern | M2 | M0 5-16 |
|---|---|---|---|---|---|---|---|---|
| A1 - A0 | +0.32 [-0.36, +0.91] | +0.21 [-0.57, +0.99] | +0.42 [-0.51, +1.24] | +0.66 [+0.01, +1.43] | +0.21 [-1.57, +1.78] | -0.12 [-1.87, +1.34] | +0.46 [-0.61, +2.07] | +0.28 [-0.37, +0.85] |
| A1 - H1 | -0.15 [-1.09, +0.78] | -0.61 [-1.69, +0.49] | +0.29 [-0.91, +1.41] | -0.75 [-2.01, +0.48] | -0.25 [-2.82, +2.36] | +1.35 [-1.04, +3.64] | +1.86 [-0.70, +2.83] | -0.45 [-1.46, +0.71] |
| A0 - H1 | -0.47 [-1.40, +0.49] | -0.82 [-1.92, +0.33] | -0.14 [-1.26, +1.04] | -1.41 [-2.78, -0.24] | -0.47 [-2.45, +1.78] | +1.47 [-0.65, +3.77] | +1.40 [-1.59, +2.44] | -0.73 [-1.63, +0.34] |
| A1 - Bilinear | +1.23 [+0.35, +2.04] | +0.60 [-0.31, +1.60] | +1.81 [+0.68, +2.82] | +1.94 [+0.86, +3.03] | -0.41 [-2.19, +0.92] | +1.73 [-0.47, +4.32] | +2.20 [+0.38, +3.51] | +1.42 [+0.48, +2.31] |
| A1 - Full ML+R2 | -1.49 [-2.57, -0.63] | -1.71 [-2.93, -0.53] | -1.29 [-2.57, -0.34] | -3.80 [-5.07, -2.73] | +0.32 [-1.64, +2.17] | +2.15 [-0.53, +4.72] | -0.83 [-2.90, +0.64] | -2.64 [-3.88, -1.55] |
| A1 - C0 | -2.21 [-3.37, -1.04] | -4.37 [-5.77, -2.99] | -0.22 [-1.60, +1.25] | -1.90 [-3.17, -0.60] | +0.71 [-1.92, +3.38] | -6.79 [-10.20, -3.38] | -2.51 [-6.33, -1.07] | -4.46 [-5.92, -2.94] |

**Cost**: A0: pretrain 14 min, peak 4.6 GB/GPU, probe 19 min; A1: pretrain 18 min, peak 4.0 GB/GPU, probe 17 min

**Verdict**

- Spatial ability shown in the direct check (A1 vs A0: higher var ratio AND higher eff-rank ratio AND lower affinity gap): **False**
- PACO val M0 Overall A1 - A0: +0.32 [-0.36, +0.91] -> improved (CI lo > 0): **False**
- Versus H1 (same protocol): -0.15 [-1.09, +0.78]; versus Full ML+R2: -1.49 [-2.57, -0.63]
- **Verdict: not a success**

## Pair B: S3 Two-way vs One-way (B1 vs B0)

**Hypothesis.** Letting the ROI working map be updated by the queries (two-way) keeps and sharpens spatial structure better than updating only the queries over a fixed working map.

**Decoder.** 16 box-conditioned queries, 24x24 working map W (from the fixed cache), 3 stages: q reads cache and W, FFN; B0 adds SA(q) + FFN2(q) (control), B1 instead W <- CA(W, q) + FFN_w(W). Final readout W + CA(W, q) + FFN -> output head. Same params (11.47M w/o aux heads).

**Training / checks**

| cond | trainable (decoder incl. aux) | dev step | dev L12 | dev L4 | dev L8 | output shape | cache unchanged | trunk grad step1 / step2 | aux-only grad into trunk | weight change trunk | time | peak mem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 12,648,576 | 12000 | 1.1873 | 1.3200 | 1.3784 | [32, 576, 768] | True | 0.0e+00 / 2.0e-05 | 0.117 | 27.95 | 16 min | 3.7 GB |
| B1 | 12,648,576 | 12000 | 1.1796 | 1.3084 | 1.3614 | [32, 576, 768] | True | 0.0e+00 / 2.5e-05 | 0.124 | 29.79 | 16 min | 4.1 GB |

**Direct checks** (normalised block-12 space; teacher = C0 crop; ratios pred / teacher, 1 = teacher-like)

| cond | L12 | var ratio | eff. rank (ratio) | neighbour-diff ratio | centred cos | affinity gap | L12 obj 1-16 | L12 part | L12 patterned | var ratio obj 1-16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bilinear (ref) | 1.844 | 0.321 | 20.0 (0.137) | 0.191 | 0.227 | 0.384 | 1.967 | 1.951 | 1.619 | 0.259 |
| H1 (CoordMLP, ref) | 1.236 | 0.142 | 56.1 (0.384) | 0.185 | 0.313 | 0.313 | 1.338 | 1.250 | 1.213 | 0.100 |
| B0 one-way | 1.329 | 0.275 | 40.9 (0.283) | 0.240 | 0.316 | 0.310 | 1.453 | 1.370 | 1.249 | 0.210 |
| B1 two-way | 1.326 | 0.300 | 46.3 (0.320) | 0.265 | 0.327 | 0.299 | 1.448 | 1.367 | 1.246 | 0.232 |

**Stage-wise spatial representation** (B: working map W in hidden space after each stage (stage1 = W0); B0 keeps W fixed by design)

| cond | stage1_var | stage1_erank | stage2_var | stage2_erank | stage3_var | stage3_erank | stage4_var | stage4_erank |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.130 | 30.882 | 0.130 | 30.882 | 0.130 | 30.882 | 0.130 | 30.882 |
| B1 | 0.192 | 30.621 | 0.196 | 38.366 | 0.190 | 42.211 | 0.168 | 48.353 |

**PACO** (val at the dev-selected epoch)

| probe | epoch | **M0 Overall** | Object | Part | Color | Material | Pattern | M2 | M0 n_in 1-4 | 5-16 | 17+ | frozen check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 one-way | 4 | **42.94** | 45.19 | 40.86 | 25.53 | 73.54 | 42.41 | 44.41 | 47.46 | 49.32 | 54.26 | ok |
| B1 two-way | 6 | **43.28** | 45.28 | 41.43 | 26.01 | 73.02 | 43.87 | 43.65 | 47.58 | 49.01 | 54.39 | ok |
| H1 (CoordMLP, ref) | 8 | **43.67** | 46.01 | 41.51 | 26.87 | 73.56 | 43.08 | 42.82 | 46.68 | 49.37 | 54.43 | ok |
| Bilinear (ref) | 6 | **42.30** | 44.80 | 39.98 | 24.18 | 73.72 | 42.70 | 42.47 | 46.65 | 47.50 | 53.65 | ok |
| Full ML+R2 (ref) | 7 | **45.02** | 47.11 | 43.08 | 29.92 | 72.99 | 42.28 | 45.51 | 49.32 | 51.56 | 56.35 | n/a |
| GT crop C0 (ref) | 5 | **45.74** | 49.77 | 42.01 | 28.02 | 72.60 | 51.22 | 47.19 | 47.50 | 53.37 | 56.80 | n/a |

| contrast | M0 Overall | Object | Part | Color | Material | Pattern | M2 | M0 5-16 |
|---|---|---|---|---|---|---|---|---|
| B1 - B0 | +0.34 [-0.36, +1.10] | +0.10 [-0.66, +0.96] | +0.57 [-0.40, +1.69] | +0.48 [-0.59, +1.68] | -0.52 [-1.98, +0.89] | +1.46 [-0.31, +3.54] | -0.76 [-2.23, +0.81] | -0.32 [-1.08, +0.70] |
| B1 - H1 | -0.39 [-1.27, +0.50] | -0.73 [-1.75, +0.34] | -0.08 [-1.16, +1.07] | -0.86 [-2.01, +0.34] | -0.54 [-2.88, +1.78] | +0.79 [-1.30, +2.93] | +0.83 [-1.70, +2.11] | -0.36 [-1.27, +0.78] |
| B0 - H1 | -0.73 [-1.75, +0.21] | -0.82 [-1.92, +0.24] | -0.65 [-1.94, +0.45] | -1.34 [-2.60, -0.14] | -0.02 [-2.43, +2.26] | -0.67 [-2.94, +1.67] | +1.59 [-1.24, +2.76] | -0.04 [-1.02, +0.94] |
| B1 - Bilinear | +0.98 [+0.22, +1.73] | +0.48 [-0.37, +1.31] | +1.45 [+0.47, +2.40] | +1.83 [+0.69, +2.85] | -0.70 [-2.33, +0.67] | +1.17 [-0.60, +3.27] | +1.18 [-0.86, +2.49] | +1.51 [+0.74, +2.43] |
| B1 - Full ML+R2 | -1.73 [-2.68, -0.93] | -1.82 [-3.02, -0.84] | -1.65 [-2.75, -0.68] | -3.92 [-5.29, -2.78] | +0.03 [-1.82, +1.90] | +1.59 [-0.70, +3.99] | -1.86 [-3.87, -0.36] | -2.55 [-3.62, -1.37] |
| B1 - C0 | -2.46 [-3.64, -1.31] | -4.49 [-5.84, -3.16] | -0.58 [-2.12, +0.91] | -2.01 [-3.37, -0.60] | +0.42 [-1.92, +3.04] | -7.35 [-10.77, -4.13] | -3.53 [-7.25, -1.92] | -4.37 [-5.86, -2.90] |

**Cost**: B0: pretrain 16 min, peak 3.7 GB/GPU, probe 15 min; B1: pretrain 16 min, peak 4.1 GB/GPU, probe 10 min

**Verdict**

- Spatial ability shown in the direct check (B1 vs B0: higher var ratio AND higher eff-rank ratio AND lower affinity gap): **True**
- PACO val M0 Overall B1 - B0: +0.34 [-0.36, +1.10] -> improved (CI lo > 0): **False**
- Versus H1 (same protocol): -0.39 [-1.27, +0.50]; versus Full ML+R2: -1.73 [-2.68, -0.93]
- **Verdict: not a success** (direct-check gain without PACO gain is not a success)

