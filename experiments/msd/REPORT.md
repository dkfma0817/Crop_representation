# MSD pilot: Multi-depth Crop-to-Shared Distillation (COCO pre-training -> frozen PACO R2 probe)

Protocol: `configs/msd/protocol.md`. Decoder pre-training on COCO train (existing 12k stream, 64 ROIs/step, seed 0, no PACO labels); checkpoint = best COCO-dev L12. PACO: encoder + decoder frozen, new R2 on PACO train (minus a 1,000-image dev for epoch selection), **final numbers on PACO VAL**; test never evaluated. Primary = val **M0 Overall**; CIs = paired image bootstrap (1,000).

## 1. COCO pre-training (what was trained)

| cond | encoder | supervision | trainable params | selected step (dev) | dev L12 | dev L4 | dev L8 | dev cos12 (raw) | time |
|---|---|---|---|---|---|---|---|---|---|
| H0 | frozen | L12 | 11,639,424 | 12000 | 1.0872 | 2.0819 | 2.0959 | 0.6335 | 12 min |
| H1 | frozen | L12 + 0.5 L8 + 0.5 L4 | 11,639,424 | 12000 | 1.1194 | 1.3245 | 1.3794 | 0.6269 | 12 min |
| H2 | adapters 4/8/12 (backbone frozen) | L12 + 1.0 L_glob | 12,531,648 | 12000 | 1.0624 | 2.0864 | 2.1005 | 0.6418 | 22 min |
| H3 | adapters 4/8/12 (backbone frozen) | L12 + 0.5 L8 + 0.5 L4 + 1.0 L_glob | 12,531,648 | 12000 | 1.0962 | 1.3202 | 1.3720 | 0.6346 | 25 min |

Dev L4 / L8 of H0 / H2 come from aux heads that receive no supervision (weight 0) and are not meaningful.

**Gradient / update checks** (`<cond>/grad_check.json`, after DDP all-reduce and clipping):

| cond | encoder backbone grad | decoder body grad (step 1) | enc adapter fc2 grad step 1 / step 2 | enc adapter fc1 grad step 3-5 (max) | aux-loss-only grad into decoder body | total weight change: body / enc adapters |
|---|---|---|---|---|---|---|
| H0 | None | 1.000 | n/a (no adapters) | n/a | 0.311 | 32.61 / n/a |
| H1 | None | 1.000 | n/a (no adapters) | n/a | 0.311 | 31.75 / n/a |
| H2 | None | 1.000 | 6.6e-16 / 1.5e-05 | 1.4e-08 | 0.311 | 32.88 / 22.014 |
| H3 | None | 1.000 | 6.6e-16 / 1.5e-05 | 1.1e-08 | 0.311 | 31.99 / 22.777 |

## 2. Direct capability checks (PACO val, 700 images, report only)

| cond | L4 | L8 | L12 | cos12 raw | pair-cos pred / teacher | spatial var ratio | affinity gap | L12 obj 1-16 | L12 part | L12 multi-material | L12 patterned (non-plain) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bil | 2.411 | 2.165 | 1.844 | 0.351 | 0.762 / 0.389 | 0.321 | 0.384 | 1.967 | 1.951 | 1.618 | 1.619 |
| H0 | 2.059 | 2.053 | 1.206 | 0.578 | 0.720 / 0.389 | 0.129 | 0.340 | 1.322 | 1.224 | 1.174 | 1.172 |
| H1 | 1.402 | 1.480 | 1.236 | 0.574 | 0.692 / 0.389 | 0.142 | 0.313 | 1.338 | 1.250 | 1.206 | 1.213 |
| H2 | 2.063 | 2.057 | 1.188 | 0.584 | 0.720 / 0.389 | 0.137 | 0.339 | 1.300 | 1.208 | 1.148 | 1.150 |
| H3 | 1.398 | 1.475 | 1.218 | 0.580 | 0.694 / 0.389 | 0.148 | 0.315 | 1.316 | 1.234 | 1.181 | 1.191 |

Lower L / affinity gap = closer to the C0 crop teacher; var ratio 1 = teacher-like spatial variation. Multi-material = >= 2 material positives; patterned = a pattern positive other than plain. Bilinear L4/L8 = bilinear read of the original block 4/8 maps; H0/H2 L4/L8 come from unsupervised aux heads.

**Global representation change vs the original encoder** (all 2,051 val images; ImageNet is not on this server):

| cond | CLS cos | mean-patch cos | per-patch cos |
|---|---|---|---|
| H0 | 1.0000 | 1.0000 | 1.0000 |
| H1 | 1.0000 | 1.0000 | 1.0000 |
| H2 | 0.9881 | 0.9936 | 0.9529 |
| H3 | 0.9867 | 0.9924 | 0.9457 |

## 3. PACO (frozen encoder + decoder, new R2; val at the dev-selected epoch)

| probe | sel. epoch (dev M0) | **M0 Overall** | Object | Part | Color | Material | Pattern | M2 Overall | M0 n_in 1-4 | 5-16 | 17+ | frozen check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H0 frozen enc + dec, L12 | 6 (48.18) | **42.31** | 44.62 | 40.17 | 24.46 | 73.15 | 42.88 | 42.38 | 45.82 | 48.09 | 54.19 | ok (no grad, checksum equal) |
| H1 frozen enc + dec, L4/8/12 | 8 (49.41) | **43.67** | 46.01 | 41.51 | 26.87 | 73.56 | 43.08 | 42.82 | 46.68 | 49.37 | 54.43 | ok (no grad, checksum equal) |
| H2 adapters + dec, L12 | 6 (48.71) | **42.66** | 44.97 | 40.53 | 24.78 | 73.38 | 43.26 | 42.68 | 46.08 | 48.28 | 54.29 | ok (no grad, checksum equal) |
| H3 adapters + dec, L4/8/12 | 6 (49.57) | **43.67** | 45.94 | 41.57 | 26.29 | 74.71 | 43.08 | 43.63 | 46.64 | 48.96 | 55.29 | ok (no grad, checksum equal) |
| Bilinear (no decoder) | 6 (47.93) | **42.30** | 44.80 | 39.98 | 24.18 | 73.72 | 42.70 | 42.47 | 46.65 | 47.50 | 53.65 | ok (no grad, checksum equal) |
| Full Multi-layer + R2 | 7 (51.73) | **45.02** | 47.11 | 43.08 | 29.92 | 72.99 | 42.28 | 45.51 | 49.32 | 51.56 | 56.35 | n/a |
| GT crop C0 (oracle) | 5 (51.24) | **45.74** | 49.77 | 42.01 | 28.02 | 72.60 | 51.22 | 47.19 | 47.50 | 53.37 | 56.80 | n/a |

**Paired contrasts (val M0; M2 overall last column)**

| contrast | M0 Overall | Object | Part | Color | Material | Pattern | M2 Overall |
|---|---|---|---|---|---|---|---|
| H1 - H0 (multi-depth supervision) | +1.36 [+0.49, +2.31] | +1.39 [+0.35, +2.38] | +1.34 [+0.37, +2.48] | +2.41 [+1.40, +3.63] | +0.41 [-1.59, +2.81] | +0.19 [-1.81, +1.92] | +0.44 [-0.91, +2.23] |
| H2 - H0 (shared-encoder adaptation + L_glob) | +0.35 [-0.08, +0.90] | +0.35 [-0.30, +1.15] | +0.36 [-0.15, +1.03] | +0.32 [-0.18, +0.99] | +0.23 [-0.78, +1.52] | +0.37 [-1.05, +1.93] | +0.31 [-0.72, +1.72] |
| H3 - H2 (multi-depth on top of adaptation) | +1.01 [+0.40, +1.53] | +0.97 [+0.14, +1.58] | +1.04 [+0.37, +1.72] | +1.51 [+0.84, +2.29] | +1.33 [-0.02, +2.44] | -0.18 [-1.82, +1.29] | +0.94 [-0.32, +2.53] |
| H3 - H0 (both) | +1.36 [+0.72, +2.05] | +1.31 [+0.43, +2.12] | +1.40 [+0.63, +2.26] | +1.83 [+1.24, +2.71] | +1.56 [-0.17, +3.25] | +0.20 [-1.58, +1.79] | +1.25 [-0.08, +3.01] |
| H0 - Bilinear | +0.01 [-0.78, +0.73] | -0.18 [-1.09, +0.79] | +0.19 [-0.80, +1.03] | +0.28 [-0.74, +1.14] | -0.57 [-2.55, +0.83] | +0.18 [-1.56, +2.71] | -0.09 [-1.81, +2.02] |
| H0 - Full ML+R2 | -2.71 [-3.67, -1.89] | -2.49 [-3.69, -1.36] | -2.91 [-4.12, -2.00] | -5.46 [-6.86, -4.48] | +0.16 [-1.73, +1.70] | +0.61 [-1.76, +3.20] | -3.13 [-4.82, -1.01] |
| H0 - C0 | -3.43 [-4.57, -2.35] | -5.15 [-6.59, -3.80] | -1.84 [-3.05, -0.45] | -3.56 [-4.86, -2.37] | +0.55 [-1.70, +3.00] | -8.33 [-11.47, -5.16] | -4.81 [-7.92, -2.79] |
| H1 - Bilinear | +1.37 [+0.30, +2.37] | +1.21 [-0.08, +2.37] | +1.52 [+0.28, +2.78] | +2.69 [+1.22, +4.02] | -0.16 [-2.93, +2.49] | +0.38 [-1.73, +2.67] | +0.35 [-1.30, +2.92] |
| H1 - Full ML+R2 | -1.34 [-2.51, -0.33] | -1.09 [-2.51, +0.21] | -1.57 [-2.84, -0.44] | -3.05 [-4.56, -1.82] | +0.57 [-2.02, +3.00] | +0.80 [-1.98, +3.59] | -2.69 [-4.14, -0.44] |
| H1 - C0 | -2.07 [-3.21, -0.93] | -3.76 [-5.20, -2.27] | -0.50 [-1.69, +0.87] | -1.15 [-2.59, +0.36] | +0.96 [-0.97, +2.94] | -8.14 [-11.88, -4.71] | -4.37 [-7.15, -2.33] |
| H2 - Bilinear | +0.36 [-0.34, +1.12] | +0.16 [-0.62, +1.18] | +0.55 [-0.35, +1.44] | +0.60 [-0.27, +1.45] | -0.34 [-2.17, +1.29] | +0.56 [-1.04, +2.79] | +0.21 [-1.13, +2.46] |
| H2 - Full ML+R2 | -2.35 [-3.28, -1.48] | -2.14 [-3.20, -0.98] | -2.55 [-3.65, -1.57] | -5.14 [-6.40, -4.07] | +0.39 [-1.68, +2.28] | +0.98 [-1.29, +3.37] | -2.82 [-4.38, -0.58] |
| H2 - C0 | -3.08 [-4.21, -1.88] | -4.80 [-6.15, -3.26] | -1.48 [-2.64, -0.11] | -3.24 [-4.54, -1.82] | +0.78 [-1.30, +3.18] | -7.96 [-11.22, -4.62] | -4.50 [-7.48, -2.45] |
| H3 - Bilinear | +1.37 [+0.54, +2.14] | +1.13 [+0.13, +2.15] | +1.59 [+0.66, +2.43] | +2.11 [+1.15, +3.09] | +0.99 [-1.24, +2.83] | +0.38 [-1.46, +2.54] | +1.15 [-0.17, +3.62] |
| H3 - Full ML+R2 | -1.35 [-2.42, -0.46] | -1.17 [-2.30, -0.13] | -1.51 [-2.75, -0.41] | -3.64 [-4.90, -2.47] | +1.72 [-0.85, +3.78] | +0.80 [-1.67, +3.04] | -1.88 [-3.48, +0.65] |
| H3 - C0 | -2.07 [-3.23, -0.87] | -3.83 [-5.29, -2.45] | -0.44 [-1.82, +0.99] | -1.73 [-3.05, -0.33] | +2.11 [+0.02, +4.17] | -8.14 [-11.67, -4.79] | -3.56 [-6.48, -1.34] |

## 4. Verdicts

- **H1 - H0 (multi-depth supervision)**: feature matching (val L12) improved = **False** (1.206 -> 1.236); PACO val M0 Overall +1.36 [+0.49, +2.31] -> improved (CI lo > 0) = **True**. Verdict: **success**.
- **H2 - H0 (shared-encoder adaptation + L_glob)**: feature matching (val L12) improved = **True** (1.206 -> 1.188); PACO val M0 Overall +0.35 [-0.08, +0.90] -> improved (CI lo > 0) = **False**. Verdict: **not a success** (feature matching improved but PACO did not -> not a success).
- **H3 - H2 (multi-depth on top of adaptation)**: feature matching (val L12) improved = **False** (1.188 -> 1.218); PACO val M0 Overall +1.01 [+0.40, +1.53] -> improved (CI lo > 0) = **True**. Verdict: **success**.
- Is the spatial feature made by the shared encoder + decoder more useful than reading the shared map directly? H0 - Bilinear +0.01 [-0.78, +0.73]; H1 - Bilinear +1.37 [+0.30, +2.37]; H2 - Bilinear +0.36 [-0.34, +1.12]; H3 - Bilinear +1.37 [+0.54, +2.14]; H0 - Full ML+R2 -2.71 [-3.67, -1.89]; H1 - Full ML+R2 -1.34 [-2.51, -0.33]; H2 - Full ML+R2 -2.35 [-3.28, -1.48]; H3 - Full ML+R2 -1.35 [-2.42, -0.46].

## 5. Cost

| run | wall-clock | note |
|---|---|---|
| pretrain H0 | 12 min | peak mem 3.1 GB / GPU, 12k steps x 64 ROIs |
| pretrain H1 | 12 min | peak mem 3.1 GB / GPU, 12k steps x 64 ROIs |
| pretrain H2 | 22 min | peak mem 4.0 GB / GPU, 12k steps x 64 ROIs |
| pretrain H3 | 25 min | peak mem 4.0 GB / GPU, 12k steps x 64 ROIs |
| probe H0 | 18 min | 10 epochs R2 (run concurrently with pre-training) |
| probe H1 | 22 min | 10 epochs R2 (run concurrently with pre-training) |
| probe H2 | 24 min | 10 epochs R2 (run concurrently with pre-training) |
| probe H3 | 12 min | 10 epochs R2 (run concurrently with pre-training) |
| probe bil | 13 min | 10 epochs R2 (run concurrently with pre-training) |
| probe mlr2 | 19 min | 10 epochs R2 (run concurrently with pre-training) |
| probe c0 | 31 min | 10 epochs R2 (run concurrently with pre-training) |

Inference (student): one encoder forward per image (+ adapters in H2/H3: 3 x 0.30M params) and one decoder pass per ROI (CoordMLP, 576 queries); no crop or per-ROI encoder pass. The C0 reference re-encodes every ROI.

