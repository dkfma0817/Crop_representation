# Smoke results

## 1. CPU transform checks (PACO val, first 12 images / 100 ROIs; `smoke/transform_checks_smoke.json`)

| check | result |
|---|---|
| mean preservation max abs  (mu(V_k) vs mu_B) | 3.81e-06 (max relative 2.4e-07) |
| O2 mean vs mu_B, max abs | 3.81e-06 |
| invalid (padding) cells changed, max abs | 0.0 (23376 invalid cells) |
| R0 pooled input vs V0, max abs | 3.81e-06 |
| R0/R2 coverage cov_c vs valid_w, max abs | 0.0 |
| measured sv(V_k)/sv(V0) - k^2, max abs | V0 0.0e+00, V1 2.6e-07, V2 5.3e-07, V3 8.0e-07 |
| sv(O2)/sv(T) - 1, max abs | 2.4e-07 |

Full-val repetition of these checks: `transform_checks.json` (written by the CPU feature-statistics pass).

## 2. GPU smoke (2026-11-05 23:45-23:46 UTC, GPU 2,3 DDP, `smoke_gpu/`)

- V2 and O2 probes through the real path (`src.svamp.probe` -> `src.ivc.train`), 60 images per split, 1 epoch x 4 steps, PACO dev 10: both `done.json`.
- O2 received the C0 crops of its own batch (`SVAMP_CROPS=1`; forward would fail with KeyError otherwise) and its frozen encoder copy is unchanged (frozen_check: requires_grad False, no grads, checksum before == after).
- Transform modules have no learned state (checkpoint = {}); reader-only training.
