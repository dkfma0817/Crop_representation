# E6 + Centered Direction Loss: tables (auto-generated)

PACO **val** only (test never evaluated). Probe-config agreement across the 3 methods: {'n_train_rows': True, 'n_val_rows': True, 'n_val_images': True, 'n_train_images': True}. Reader epoch = max PACO-dev M0.

## 1. Main table

| Method | Token MSE | Token Cos | **Centered Dir Cos** | Pooled Cos | Mean-feat MSE | Spatial Var / C0 | **R2 M0** | Obj D5–16 | Part D1–4 | R2 M2 | R0 M0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 3.5666 | 0.3503 | 0.2385 | 0.4741 | 1.9466 | 0.4368 | 42.71 | 46.47 | 40.10 | 44.05 | 37.04 |
| Full Multi-layer + R2 | n/a | n/a | n/a | n/a | n/a | n/a | 45.02 | 50.52 | 43.72 | 45.51 | n/a |
| E6-original | 2.1721 | 0.5148 | 0.3865 | 0.6626 | 0.8237 | 0.3819 | 42.70 | 47.53 | 41.19 | 43.43 | 38.07 |
| Matched L12 (A) | 2.1691 | 0.5151 | 0.3866 | 0.6629 | 0.8216 | 0.3813 | 42.12 | 47.14 | 41.04 | 42.80 | 38.09 |
| Multi-layer Memory (B) | 2.1590 | 0.5191 | 0.3879 | 0.6685 | 0.8137 | 0.3814 | 42.84 | 47.82 | 41.23 | 43.31 | 38.41 |

Representation metrics: PACO val, all 11,109 ROIs, per-ROI coverage-weighted, mean over ROIs (c_dir = the training definition).

## 2. M0 (R2)

| Method | epoch (dev M0) | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 6 (47.80) | 42.71 | 46.47 | 40.10 | 71.82 | 54.06 | 45.44 | 40.19 | 24.33 | 73.87 | 44.00 |
| Full Multi-layer + R2 | 7 (51.73) | 45.02 | 50.52 | 43.72 | 70.71 | 55.43 | 47.11 | 43.08 | 29.92 | 72.99 | 42.28 |
| E6-original | 6 (47.06) | 42.70 | 47.53 | 41.19 | 68.45 | 53.20 | 45.05 | 40.53 | 24.76 | 73.15 | 43.65 |
| Matched L12 (A) | 6 (47.19) | 42.12 | 47.14 | 41.04 | 68.39 | 52.45 | 44.25 | 40.14 | 24.29 | 72.76 | 42.42 |
| Multi-layer Memory (B) | 6 (47.25) | 42.84 | 47.82 | 41.23 | 69.33 | 52.71 | 44.88 | 40.94 | 24.79 | 73.35 | 44.04 |

Paired bootstrap (M0, R2): difference [95% CI]

| Contrast | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|
| Multi-layer Memory (B) − Matched L12 (A) | +0.72 [+0.09, +1.17] | +0.68 [-0.03, +1.31] | +0.19 [-0.38, +0.96] | +0.94 [-1.56, +3.10] | +0.26 [-0.87, +1.14] | +0.63 [-0.15, +1.25] | +0.80 [+0.02, +1.33] | +0.50 [-0.26, +1.00] | +0.59 [-0.35, +1.65] | +1.61 [-0.17, +3.24] |
| Multi-layer Memory (B) − E6-original | +0.13 [-0.46, +0.66] | +0.30 [-0.43, +1.01] | +0.04 [-0.64, +1.05] | +0.88 [-2.53, +4.43] | -0.49 [-1.50, +0.49] | -0.17 [-0.97, +0.59] | +0.41 [-0.32, +0.93] | +0.03 [-0.68, +0.51] | +0.21 [-0.87, +1.32] | +0.39 [-1.18, +1.75] |
| Multi-layer Memory (B) − Bicubic | +0.12 [-0.65, +0.68] | +1.35 [+0.21, +2.27] | +1.13 [+0.06, +2.52] | -2.48 [-6.41, +1.39] | -1.35 [-2.61, -0.37] | -0.56 [-1.54, +0.23] | +0.75 [-0.15, +1.50] | +0.45 [-0.54, +1.23] | -0.52 [-2.05, +0.79] | +0.04 [-1.58, +1.56] |
| Multi-layer Memory (B) − Full Multi-layer + R2 | -2.18 [-3.30, -1.33] | -2.70 [-4.17, -1.44] | -2.49 [-3.91, -0.82] | -1.38 [-5.58, +2.74] | -2.72 [-4.65, -1.38] | -2.23 [-3.62, -1.07] | -2.14 [-3.50, -1.02] | -5.13 [-6.57, -3.90] | +0.37 [-1.81, +2.48] | +1.76 [-1.21, +4.48] |
| Matched L12 (A) − E6-original | -0.59 [-1.04, -0.04] | -0.38 [-1.03, +0.36] | -0.15 [-0.77, +0.47] | -0.06 [-3.22, +2.91] | -0.75 [-1.50, +0.15] | -0.79 [-1.39, -0.01] | -0.39 [-1.03, +0.17] | -0.47 [-1.03, +0.12] | -0.39 [-1.57, +0.67] | -1.22 [-2.59, +0.40] |
| Full Multi-layer + R2 − Bicubic | +2.30 [+1.41, +3.28] | +4.05 [+2.64, +5.37] | +3.62 [+2.21, +5.22] | -1.10 [-5.28, +2.42] | +1.37 [-0.09, +3.17] | +1.67 [+0.49, +2.83] | +2.89 [+1.79, +4.07] | +5.59 [+4.23, +6.94] | -0.88 [-2.92, +1.08] | -1.72 [-4.33, +0.84] |

## 2. M0 (R0)

| Method | epoch (dev M0) | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 10 (42.58) | 37.04 | 43.83 | 36.31 | 69.11 | 47.12 | 40.03 | 34.26 | 18.62 | 68.10 | 37.33 |
| E6-original | 9 (43.61) | 38.07 | 44.50 | 36.91 | 69.70 | 47.58 | 41.15 | 35.22 | 19.13 | 69.37 | 39.70 |
| Matched L12 (A) | 10 (43.59) | 38.09 | 44.56 | 36.87 | 69.29 | 47.57 | 41.21 | 35.21 | 19.11 | 69.38 | 39.70 |
| Multi-layer Memory (B) | 10 (43.93) | 38.41 | 44.91 | 37.23 | 70.02 | 47.82 | 41.55 | 35.50 | 19.59 | 69.55 | 39.89 |

Paired bootstrap (M0, R0): difference [95% CI]

| Contrast | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|
| Multi-layer Memory (B) − Matched L12 (A) | +0.32 [+0.16, +0.49] | +0.35 [-0.00, +0.70] | +0.36 [+0.09, +0.62] | +0.72 [-0.85, +2.20] | +0.26 [+0.02, +0.57] | +0.34 [+0.10, +0.61] | +0.30 [+0.12, +0.44] | +0.48 [+0.28, +0.65] | +0.17 [-0.28, +0.71] | +0.19 [-0.11, +0.54] |
| Multi-layer Memory (B) − E6-original | +0.34 [+0.16, +0.54] | +0.40 [+0.02, +0.76] | +0.32 [-0.02, +0.63] | +0.31 [-1.28, +1.74] | +0.24 [-0.03, +0.61] | +0.40 [+0.12, +0.74] | +0.28 [+0.10, +0.44] | +0.47 [+0.26, +0.64] | +0.18 [-0.26, +0.80] | +0.19 [-0.17, +0.62] |
| Multi-layer Memory (B) − Bicubic | +1.37 [+0.97, +1.72] | +1.08 [+0.31, +1.78] | +0.92 [+0.25, +1.51] | +0.91 [-1.54, +3.38] | +0.70 [+0.21, +1.23] | +1.51 [+0.92, +1.99] | +1.24 [+0.82, +1.63] | +0.97 [+0.61, +1.35] | +1.45 [+0.33, +2.30] | +2.56 [+1.60, +3.67] |
| Multi-layer Memory (B) − Full Multi-layer + R2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Matched L12 (A) − E6-original | +0.02 [-0.10, +0.14] | +0.05 [-0.21, +0.29] | -0.04 [-0.28, +0.16] | -0.41 [-1.22, +0.38] | -0.01 [-0.23, +0.21] | +0.06 [-0.14, +0.24] | -0.02 [-0.13, +0.12] | -0.02 [-0.11, +0.09] | +0.01 [-0.27, +0.35] | +0.01 [-0.39, +0.41] |
| Full Multi-layer + R2 − Bicubic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## 2. M2 (R2)

| Method | epoch (dev M0) | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 6 (47.80) | 44.05 | 32.85 | 18.73 | n/a | 55.58 | 43.88 | 44.23 | 36.21 | 39.61 | 49.28 |
| Full Multi-layer + R2 | 7 (51.73) | 45.51 | 37.29 | 32.94 | n/a | 57.99 | 45.46 | 45.56 | 47.16 | 35.47 | 45.60 |
| E6-original | 6 (47.06) | 43.43 | 34.50 | 27.24 | n/a | 58.53 | 44.10 | 42.77 | 37.61 | 36.38 | 48.22 |
| Matched L12 (A) | 6 (47.19) | 42.80 | 34.86 | 22.37 | n/a | 56.66 | 42.28 | 43.31 | 36.40 | 35.42 | 48.73 |
| Multi-layer Memory (B) | 6 (47.25) | 43.31 | 36.49 | 20.68 | n/a | 55.18 | 43.37 | 43.25 | 38.58 | 35.97 | 49.02 |

Paired bootstrap (M2, R2): difference [95% CI]

| Contrast | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|
| Multi-layer Memory (B) − Matched L12 (A) | +0.51 [-0.52, +1.89] | +1.63 [-1.92, +5.99] | -1.69 [-7.20, +3.03] | n/a | -1.48 [-5.20, +3.58] | +1.09 [-0.47, +3.11] | -0.06 [-1.43, +1.54] | +2.18 [+0.24, +3.41] | +0.55 [-1.40, +3.18] | +0.29 [-1.69, +3.61] |
| Multi-layer Memory (B) − E6-original | -0.12 [-1.14, +1.42] | +1.99 [-1.95, +5.18] | -6.57 [-11.98, +1.63] | n/a | -3.34 [-6.63, +1.70] | -0.73 [-1.92, +1.84] | +0.48 [-1.15, +1.80] | +0.97 [-0.65, +2.40] | -0.41 [-2.97, +2.64] | +0.80 [-1.29, +3.37] |
| Multi-layer Memory (B) − Bicubic | -0.75 [-1.96, +1.21] | +3.64 [-1.53, +10.01] | +1.95 [-4.27, +12.58] | n/a | -0.40 [-2.43, +2.65] | -0.51 [-1.96, +2.13] | -0.98 [-2.87, +1.18] | +2.37 [-0.10, +4.05] | -3.64 [-6.08, +1.18] | -0.26 [-1.94, +3.12] |
| Multi-layer Memory (B) − Full Multi-layer + R2 | -2.20 [-4.07, -0.68] | -0.81 [-10.66, +8.80] | -12.26 [-25.60, +3.79] | n/a | -2.81 [-8.36, +1.13] | -2.09 [-4.69, +0.24] | -2.31 [-4.53, -0.50] | -8.58 [-11.17, -6.38] | +0.50 [-3.32, +3.97] | +3.42 [-1.38, +7.21] |
| Matched L12 (A) − E6-original | -0.64 [-1.74, +0.53] | +0.36 [-4.32, +3.60] | -4.87 [-10.61, +2.53] | n/a | -1.86 [-4.49, +1.12] | -1.81 [-2.97, +0.15] | +0.54 [-1.09, +1.59] | -1.21 [-2.54, +0.55] | -0.95 [-3.23, +1.49] | +0.51 [-1.85, +2.03] |
| Full Multi-layer + R2 − Bicubic | +1.45 [-0.00, +3.95] | +4.45 [-3.05, +12.50] | +14.21 [+1.21, +28.67] | n/a | +2.41 [-1.29, +8.71] | +1.57 [-0.44, +5.05] | +1.33 [-0.70, +3.87] | +10.95 [+7.92, +13.44] | -4.14 [-7.13, +1.84] | -3.68 [-6.65, +1.70] |

## 2. M2 (R0)

| Method | epoch (dev M0) | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 10 (42.58) | 35.82 | 25.34 | 16.40 | n/a | 45.16 | 36.48 | 35.16 | 26.37 | 29.89 | 37.89 |
| E6-original | 9 (43.61) | 35.22 | 25.70 | 17.99 | n/a | 45.56 | 36.50 | 33.94 | 26.74 | 28.25 | 38.60 |
| Matched L12 (A) | 10 (43.59) | 35.03 | 26.07 | 15.66 | n/a | 45.53 | 36.50 | 33.57 | 26.64 | 28.41 | 38.25 |
| Multi-layer Memory (B) | 10 (43.93) | 35.62 | 26.16 | 16.85 | n/a | 46.07 | 36.66 | 34.58 | 27.29 | 29.30 | 39.24 |

Paired bootstrap (M2, R0): difference [95% CI]

| Contrast | Overall | Obj D5-16 | Part D1-4 | Obj D1-4 | Obj D17+ | Object | Part | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|
| Multi-layer Memory (B) − Matched L12 (A) | +0.59 [-0.24, +1.34] | +0.09 [-2.85, +2.75] | +1.18 [-1.85, +7.47] | n/a | +0.53 [-0.81, +1.89] | +0.16 [-0.58, +1.06] | +1.02 [-0.32, +2.26] | +0.65 [-0.03, +1.21] | +0.89 [-1.08, +2.96] | +0.98 [-0.46, +1.89] |
| Multi-layer Memory (B) − E6-original | +0.41 [-0.39, +1.20] | +0.46 [-1.43, +3.74] | -1.14 [-4.73, +2.83] | n/a | +0.51 [-1.50, +1.78] | +0.17 [-0.63, +1.06] | +0.64 [-0.78, +1.96] | +0.55 [-0.15, +1.15] | +1.05 [-0.77, +3.37] | +0.63 [-1.21, +1.58] |
| Multi-layer Memory (B) − Bicubic | -0.20 [-1.13, +1.48] | +0.83 [-2.41, +3.89] | +0.45 [-4.85, +8.94] | n/a | +0.91 [-1.45, +3.10] | +0.18 [-0.84, +1.75] | -0.57 [-2.10, +1.92] | +0.92 [-0.47, +1.97] | -0.60 [-3.51, +3.36] | +1.35 [-0.55, +3.69] |
| Multi-layer Memory (B) − Full Multi-layer + R2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Matched L12 (A) − E6-original | -0.18 [-0.61, +0.20] | +0.37 [-0.65, +3.88] | -2.33 [-9.06, +1.45] | n/a | -0.02 [-1.96, +0.85] | +0.00 [-0.47, +0.49] | -0.37 [-1.11, +0.19] | -0.10 [-0.60, +0.36] | +0.16 [-0.54, +1.02] | -0.35 [-1.56, +0.44] |
| Full Multi-layer + R2 − Bicubic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## 3. Representation metrics: paco_val

| Method | mse | tok_cos | c_dir | pool_cos | e_mean | sv_ratio | e_cent | loss |
|---|---|---|---|---|---|---|---|---|
| Bicubic | 3.5666 | 0.3503 | 0.2385 | 0.4741 | 1.9466 | 0.4368 | 1.6200 | 4.2163 |
| E6-original | 2.1721 | 0.5148 | 0.3865 | 0.6626 | 0.8237 | 0.3819 | 1.3484 | 2.6573 |
| Matched L12 (A) | 2.1691 | 0.5151 | 0.3866 | 0.6629 | 0.8216 | 0.3813 | 1.3476 | 2.6541 |
| Multi-layer Memory (B) | 2.1590 | 0.5191 | 0.3879 | 0.6685 | 0.8137 | 0.3814 | 1.3453 | 2.6399 |

Centered Dir Cos by group (paco_val):

| Method | all | object | part | object D1-4 | object D5-16 | object D17+ | part D1-4 | part D5-16 | part D17+ |
|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 0.2385 | 0.3346 | 0.2006 | 0.1080 | 0.1785 | 0.4792 | 0.0460 | 0.1559 | 0.4600 |
| E6-original | 0.3865 | 0.4391 | 0.3657 | 0.2980 | 0.3253 | 0.5430 | 0.3070 | 0.3052 | 0.5212 |
| Matched L12 (A) | 0.3866 | 0.4395 | 0.3658 | 0.2981 | 0.3258 | 0.5433 | 0.3065 | 0.3055 | 0.5217 |
| Multi-layer Memory (B) | 0.3879 | 0.4407 | 0.3671 | 0.3001 | 0.3275 | 0.5441 | 0.3079 | 0.3070 | 0.5224 |

Spatial Var / C0 by group (paco_val):

| Method | all | object | part | object D1-4 | object D5-16 | object D17+ | part D1-4 | part D5-16 | part D17+ |
|---|---|---|---|---|---|---|---|---|---|
| Bicubic | 0.4368 | 0.5441 | 0.3946 | 0.2776 | 0.3838 | 0.6946 | 0.1749 | 0.3846 | 0.6930 |
| E6-original | 0.3819 | 0.4304 | 0.3628 | 0.2526 | 0.3066 | 0.5450 | 0.2551 | 0.3271 | 0.5494 |
| Matched L12 (A) | 0.3813 | 0.4295 | 0.3622 | 0.2515 | 0.3060 | 0.5438 | 0.2545 | 0.3267 | 0.5488 |
| Multi-layer Memory (B) | 0.3814 | 0.4303 | 0.3621 | 0.2518 | 0.3067 | 0.5447 | 0.2541 | 0.3262 | 0.5494 |

c_dir(Multi-layer Memory (B)) − c_dir(Matched L12 (A)), paco_val: +0.0012 [+0.0012, +0.0013] (image bootstrap 1000, seed 0)

tok_cos(Multi-layer Memory (B)) − tok_cos(Matched L12 (A)), paco_val: +0.0040 [+0.0038, +0.0043] (image bootstrap 1000, seed 0)

mse(Multi-layer Memory (B)) − mse(Matched L12 (A)), paco_val: -0.0101 [-0.0107, -0.0096] (image bootstrap 1000, seed 0)

c_dir(Multi-layer Memory (B)) − c_dir(E6-original), paco_val: +0.0014 [+0.0013, +0.0015] (image bootstrap 1000, seed 0)

tok_cos(Multi-layer Memory (B)) − tok_cos(E6-original), paco_val: +0.0043 [+0.0041, +0.0045] (image bootstrap 1000, seed 0)

mse(Multi-layer Memory (B)) − mse(E6-original), paco_val: -0.0131 [-0.0137, -0.0125] (image bootstrap 1000, seed 0)

c_dir(Matched L12 (A)) − c_dir(E6-original), paco_val: +0.0002 [+0.0001, +0.0002] (image bootstrap 1000, seed 0)

tok_cos(Matched L12 (A)) − tok_cos(E6-original), paco_val: +0.0003 [+0.0002, +0.0003] (image bootstrap 1000, seed 0)

mse(Matched L12 (A)) − mse(E6-original), paco_val: -0.0030 [-0.0032, -0.0028] (image bootstrap 1000, seed 0)

## 3. Representation metrics: coco_dev

| Method | mse | tok_cos | c_dir | pool_cos | e_mean | sv_ratio | e_cent | loss |
|---|---|---|---|---|---|---|---|---|
| Bicubic | 3.0653 | 0.4479 | 0.3300 | 0.6018 | 1.3334 | 0.5608 | 1.7319 | 3.6174 |
| E6-original | 1.9718 | 0.5810 | 0.4556 | 0.7519 | 0.5573 | 0.4661 | 1.4145 | 2.3908 |
| Matched L12 (A) | 1.9678 | 0.5822 | 0.4561 | 0.7533 | 0.5553 | 0.4653 | 1.4126 | 2.3857 |
| Multi-layer Memory (B) | 1.9583 | 0.5856 | 0.4575 | 0.7579 | 0.5485 | 0.4658 | 1.4098 | 2.3727 |

Centered Dir Cos by group (coco_dev):

| Method | all | D1-4 | D5-16 | D17+ |
|---|---|---|---|---|
| Bicubic | 0.3300 | 0.0627 | 0.1657 | 0.5099 |
| E6-original | 0.4556 | 0.3256 | 0.3287 | 0.5722 |
| Matched L12 (A) | 0.4561 | 0.3258 | 0.3293 | 0.5728 |
| Multi-layer Memory (B) | 0.4575 | 0.3278 | 0.3310 | 0.5738 |

Spatial Var / C0 by group (coco_dev):

| Method | all | D1-4 | D5-16 | D17+ |
|---|---|---|---|---|
| Bicubic | 0.5608 | 0.2125 | 0.4187 | 0.7504 |
| E6-original | 0.4661 | 0.2662 | 0.3533 | 0.5943 |
| Matched L12 (A) | 0.4653 | 0.2657 | 0.3528 | 0.5932 |
| Multi-layer Memory (B) | 0.4658 | 0.2662 | 0.3525 | 0.5943 |

c_dir(Multi-layer Memory (B)) − c_dir(Matched L12 (A)), coco_dev: +0.0013 [+0.0012, +0.0015] (image bootstrap 1000, seed 0)

tok_cos(Multi-layer Memory (B)) − tok_cos(Matched L12 (A)), coco_dev: +0.0034 [+0.0031, +0.0038] (image bootstrap 1000, seed 0)

mse(Multi-layer Memory (B)) − mse(Matched L12 (A)), coco_dev: -0.0095 [-0.0104, -0.0086] (image bootstrap 1000, seed 0)

c_dir(Multi-layer Memory (B)) − c_dir(E6-original), coco_dev: +0.0019 [+0.0018, +0.0021] (image bootstrap 1000, seed 0)

tok_cos(Multi-layer Memory (B)) − tok_cos(E6-original), coco_dev: +0.0046 [+0.0042, +0.0050] (image bootstrap 1000, seed 0)

mse(Multi-layer Memory (B)) − mse(E6-original), coco_dev: -0.0135 [-0.0145, -0.0124] (image bootstrap 1000, seed 0)

c_dir(Matched L12 (A)) − c_dir(E6-original), coco_dev: +0.0006 [+0.0005, +0.0007] (image bootstrap 1000, seed 0)

tok_cos(Matched L12 (A)) − tok_cos(E6-original), coco_dev: +0.0012 [+0.0010, +0.0013] (image bootstrap 1000, seed 0)

mse(Matched L12 (A)) − mse(E6-original), coco_dev: -0.0040 [-0.0043, -0.0036] (image bootstrap 1000, seed 0)

### E6-original: COCO-dev at checkpoints (selection = min L_old): selected step 12000

| step | L_old (dev) | MSE | token cos | L_dir | L_new | c_dir | excluded |
|---|---|---|---|---|---|---|---|
| 3000 | 2.5071 | 2.0718 | 0.5647 | nan | nan | nan | – |
| 6000 | 2.4201 | 1.9967 | 0.5766 | nan | nan | nan | – |
| 9000 | 2.3953 | 1.9754 | 0.5801 | nan | nan | nan | – |
| 12000 | 2.3908 | 1.9718 | 0.5810 | nan | nan | nan | – |

time 503 s, peak mem 1.78 GB

### Matched L12 (A): COCO-dev at checkpoints (selection = min L_old): selected step 12000

| step | L_old (dev) | MSE | token cos | L_dir | L_new | c_dir | excluded |
|---|---|---|---|---|---|---|---|
| 3000 | 2.5072 | 2.0721 | 0.5649 | nan | nan | nan | – |
| 6000 | 2.4169 | 1.9942 | 0.5773 | nan | nan | nan | – |
| 9000 | 2.3906 | 1.9717 | 0.5811 | nan | nan | nan | – |
| 12000 | 2.3857 | 1.9678 | 0.5822 | nan | nan | nan | – |

time 525 s, peak mem 1.81 GB

### Multi-layer Memory (B): COCO-dev at checkpoints (selection = min L_old): selected step 12000

| step | L_old (dev) | MSE | token cos | L_dir | L_new | c_dir | excluded |
|---|---|---|---|---|---|---|---|
| 3000 | 2.5009 | 2.0683 | 0.5674 | nan | nan | nan | – |
| 6000 | 2.4072 | 1.9874 | 0.5802 | nan | nan | nan | – |
| 9000 | 2.3780 | 1.9625 | 0.5845 | nan | nan | nan | – |
| 12000 | 2.3727 | 1.9583 | 0.5856 | nan | nan | nan | – |

time 518 s, peak mem 1.81 GB

