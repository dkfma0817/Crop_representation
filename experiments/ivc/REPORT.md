# PACO Bilinear / Bicubic / C0 Crop: 성능과 Feature 차이 (Val 분석 + Test-primary 1회, 2026-11-05)

> 사본: 원본은 `output/2026-11-05/ivc/REPORT.md`(git 제외). 그 폴더 기준 경로(`probe_*`, `test/`, `logs/`, `qat_pause/`)는 서버의 실행 디렉터리를 가리킨다.

새 decoder 학습 없음. Frozen DINOv2-B/14 최종층 feature 위에서 조건별 R0/R2 reader만 학습했다(seed 0). 분석은 Val로 했고, 사용자 승인 후 **고정 checkpoint로 PACO Test-primary를 1회** 평가했다(G절). 실행 전 규칙은 [`configs/ivc/protocol.md`](../../configs/ivc/protocol.md), 모든 자동 생성 표는 [`tables.md`](tables.md), 수치 원본은 `analysis.json`, `extra_checks.json`에 있다.

## 결론 먼저

- **Q1. 가장 크게 다른 부분.** Bicubic(그리고 Bilinear) feature와 C0 feature는 **(a) ROI의 평균 표현 자체**와 **(b) ROI 내부의 공간 구조**에서 모두 크게 다르다. 두 차이는 크기가 비슷하다. 차이를 결정하는 가장 큰 요인은 ROI 크기다.
  - 평균 표현: pooled cosine 0.47. Token MSE 중 평균 차이 몫(E_mean)이 전체의 약 절반이고, D1–4에서는 72%다.
  - 공간 구조: 보간 feature는 ROI 안에서 거의 일정하다. 인접 token cosine 0.97 vs C0 0.78, effective rank 6.3 vs 22.4, mid/high 주파수 energy는 C0의 1% 미만이다.
  - ROI 크기: token cosine이 D1–4 0.12, D5–16 0.31, D17+ 0.58이다.
  - **Bicubic은 C0에 가까워지지 않았다.** 공간 분산은 Bilinear보다 커졌지만(C0 대비 0.30 → 0.44), token MSE와 pooled L2는 오히려 커졌고 token cosine은 사실상 같다(−0.002).
- **Q2. 성능 차이와의 연결.**
  - 성능 차이 자체는 분명하다. C0 − Bicubic이 val M0 Overall에서 R2 +2.73 [+1.55, +3.89], R0 +2.60 [+1.67, +3.35]다. 반면 Bicubic − Bilinear는 R2 +0.41 [−0.03, +0.97], R0 +0.03으로 유의하지 않다.
  - 그러나 ROI 단위에서 feature 차이와 예측 차이의 연결은 **약하게만** 확인됐다. 어떤 지표도 ROI별 C0 이득을 크게 설명하지 못한다.
    - 일관되게 나온 것은 **공간 구조 차이(E_cent, self-similarity 차이)**뿐이다. ROI 크기와 level을 통제한 partial ρ가 +0.12로 작다. G1(C0만 정답) > G2 > G4(Bicubic만 정답) 순서이고, 작은·중간 ROI와 material/pattern에서 더 분명하다.
    - **평균 표현 차이**(pooled cosine, E_mean)는 C0의 이득이 아니라, 어느 쪽이든 두 조건의 예측이 갈리는 ROI와 같이 움직인다(G4에서 오히려 더 크다).
  - Feature 차이가 가장 큰 Part D1–4에서는 C0의 성능 이득이 없다(R2 −1.03 [−3.18, +0.93]).

---

## A. Experimental Setup

**데이터와 split.**
- PACO train에서 중복 6장을 제외하고, PACO dev 1,000장(train에서 이미지 단위로 고정, rng 0)을 뺐다. 학습 데이터는 37,472장 / 200,088 ROI다.
- Dev는 epoch 선택과 정오답 threshold 보정에만 썼다.
- 평가는 PACO **val** 2,051장 / 11,109 ROI(object 3,140, part 7,969)다.
- Train/val 이미지 id는 겹치지 않는다(assert). 세 조건의 score 파일은 모두 (11,109, 59)이고 같은 manifest 행 순서와 같은 label 배열을 쓴다. Probe config의 train/val 행·이미지 수와 step 수도 같음을 확인했다.

**Feature 추출.**
- Encoder: timm `vit_base_patch14_dinov2.lvd142m`, 336 letterbox(PIL bicubic resize, ImageNet 정규화, padding = 정규화 공간에서 0이고 encoder가 이 pixel을 본다). Final LayerNorm을 적용한 patch token 24×24×768을 쓴다. bf16 autocast로 forward한 뒤 fp32로 바꾸고, sampling과 모든 지표는 fp32로 계산했다.
- A. **Bilinear**: 기존 `roi_reader.bilinear` 호출 그대로. C0 crop grid의 cell 중심을 full canvas로 옮긴 좌표에서 `grid_sample(align_corners=False, border)`.
- B. **Bicubic**: 같은 grid와 설정에 `mode='bicubic'`(PyTorch, A = −0.75). F336 forward는 A와 공유한다.
- C. **C0**: 정수 GT box로 원본을 crop → 336 letterbox → 같은 encoder.
- 선택 참고 조건 C1, F672는 생략했다(이 서버에 reader가 없어 새로 학습해야 하고, F672는 grid가 달라 token 단위 비교 대상이 아님).
- 이 서버에는 `paco_c0_multilayer`, `multilevel_crop_native`, `crop_res_context`의 코드·cache·결과가 없고, `paco_token_upsample`은 코드만 있다. 그래서 기존 feature cache는 쓰지 않고 새로 추출했다.

**ROI 좌표 정렬 검증** ([`figures/align/`](figures/align/), `alignment.json`).

| 검사 | 결과 |
|---|---|
| 독립 수식으로 다시 계산한 C0 cell 중심 → F336 sampling 좌표 vs 실제 사용 `bil_grid` (val 전체 11,109 ROI, 유효 cell 389만 개) | 최대 오차 **9.8e-6 px** (float32 저장 오차) |
| 합성 좌표 map(채널 = full cell 중심 X/Y)을 실제 호출로 sampling | Bilinear 최대 오차 7.7e-5 px. **Bicubic 평균 0.55, 최대 0.67 px**: kernel(A = −0.75)이 선형 ramp를 정확히 재현하지 못하는 특성이다(해석적으로 1/4 cell 위치에서 0.047 cell = 0.66 px, 측정값과 일치). Grid는 A와 같으므로 B 조건의 정의로 두고 수정하지 않았다 |
| Feature shift scan (고정 600 ROI, Bicubic grid를 ±7/±14 px 이동) | C0와의 평균 token cosine이 **(0, 0)에서 최대**(0.4415, 7 px 이동 시 0.41). 추정 오프셋 dx +0.11, dy −0.06 px. ROI별 argmax가 (0, 0)인 비율 71% |
| Border clamp / 이미지 밖 | 유효 cell의 2.1%가 full-grid 중심 hull 밖(border clamp), 0.3%가 이미지 content 밖(C0 가장자리 부분 cell) |

- 그림 12장(종횡비 × 크기, seed 0으로 사전 고정): 원본 + GT box + C0 cell 중심(유효/padding), C0 입력 crop + grid, F336 위 sampling 위치, coverage mask.
- 정렬 오류는 발견되지 않아 수정한 것이 없다.
- 유효 mask: **M = C0 coverage > 0**(coverage 가중, R0/R2와 같음), **M1 = coverage = 1인 직사각형**(구조 지표용). Padding cell은 주 지표에서 빼고 따로 보고했다.

**Reader 학습** (11-05 MSD probe와 같은 프로토콜, 조건마다 같은 설정).
- 공통: 10 epoch, 이미지 pack ≈128 ROI/step, hflip 0.5, masked BCE, AdamW(R2 lr 3e-4, head lr 1e-3, wd 0.05), warmup 1 epoch + cosine, clip 1.0, reader init seed 7, GPU 2+3 DDP.
- R2: 2 cross-attention block, crop-grid geometry, cov > 0 mask. **세 조건이 완전히 같은 geometry를 쓰고 token만 다르다.**
- R0: coverage 가중 평균 → level별 z-score(각 조건 자기 train feature 4,000장으로 계산) → Linear head.
- Checkpoint: epoch 1–10 중 **dev M0 Overall 최대** epoch. 모든 epoch을 저장했다.

**평가.**
- **M0**(primary) = (level, attribute) pooled AP의 평균(val pos ≥ 10, neg ≥ 10, train pos ≥ 10).
- **M2** = 기존 프로젝트의 category-macro AP(PACO 공식 detection AP 아님).
- 크기 bin = manifest `n_in`(F336 24×24 grid, D1–4 / D5–16 / D17+). MSD 보고서의 SigLIP2 bin과 다르다.
- Paired image cluster bootstrap: 1,000회, seed 0, 모든 조건이 같은 draw. 평가 표본의 불확실성이며 학습 seed 분산이 아니다.

**실행 중 생긴 일 (투명성).**
1. 사용자 요청으로 QAT를 checkpoint 저장을 기다리지 않고 08:07에 강제 중단했다. 09:04에 checkpoint-10000부터 `--resume`으로 재개했다(10,000 이후 약 4,500 step은 다시 학습).
2. 첫 C0 학습은 제가 큐 tmux 세션을 닫으면서 SIGHUP으로 6 epoch 후 종료됐다(`probe_c0_partial_sighup_ep6/`에 보관). 처음부터 다시 학습했다. 이때 DataLoader worker 수를 8 → 12로 바꿔서 **C0의 hflip 난수 순서가 Bilinear/Bicubic과 다르다**(seed는 같음).
   - 강건성 확인: flip 순서가 같은 11-05 MSD C0 R2로 계산하면 C0 − Bicubic = **+3.03 [+1.83, +4.14]**다.
   - 두 C0 실행의 차이는 −0.30 [−0.84, +0.29]로 표본 잡음 수준이다.
   - Bilinear R2는 MSD `probe_bil`과 val M0 42.30, epoch 6으로 정확히 재현됐다.
3. 첫 feature pass는 학습 3개와 GPU를 같이 써서 느렸다. 같은 코드를 8 shard로 다시 돌려 병합했다(ROI 누락·중복 없음 확인). Effective rank 계산에서 퇴화 행렬 1건이 GPU `eigh`에서 실패해, 같은 양을 CPU float64 SVD로 계산하도록 fallback을 넣었다.

## B. PACO Validation Performance

**M0 (val, dev로 고른 epoch)**

| Method | Reader | ep (dev M0) | Overall | Object | Part | Obj D5–16 | Part D1–4 | Obj D17+ | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bilinear | R0 | 10 (42.57) | 37.01 | 39.91 | 34.32 | 43.42 | 36.23 | 47.01 | 18.57 | 67.97 | 37.50 |
| Bicubic | R0 | 10 (42.58) | 37.04 | 40.03 | 34.26 | 43.83 | 36.31 | 47.12 | 18.62 | 68.10 | 37.33 |
| C0 Crop | R0 | 10 (45.45) | **39.64** | 43.77 | 35.80 | 48.84 | 35.02 | 48.52 | 20.35 | 69.83 | 43.70 |
| Bilinear | R2 | 6 (47.93) | 42.30 | 44.80 | 39.98 | 46.39 | 40.37 | 53.01 | 24.18 | 73.72 | 42.70 |
| Bicubic | R2 | 6 (47.80) | 42.71 | 45.44 | 40.19 | 46.47 | 40.10 | 54.06 | 24.33 | 73.87 | 44.00 |
| C0 Crop | R2 | 5 (51.23) | **45.45** | 49.31 | 41.86 | 54.92 | 39.07 | 55.43 | 27.99 | 72.39 | 50.14 |

- R0는 dev M0가 10 epoch까지 계속 올라서 10 epoch이 선택됐다.
- D17+ 전체(object + part): R0 46.56 / 46.65 / 48.02, R2 53.64 / 54.56 / 56.07.
- M0 unit 수: D1–4 53개(object 11), D5–16 87개, D17+ 87개, 전체 106개.

**Paired bootstrap 95% CI (M0)**

| Contrast | Reader | Overall | Object | Part | Obj D5–16 | Part D1–4 | Obj D17+ | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic − Bilinear | R0 | +0.03 [−0.12, +0.15] | +0.12 [−0.08, +0.29] | −0.06 [−0.26, +0.10] | +0.41 [+0.14, +0.69] | +0.08 [−0.25, +0.34] | +0.11 [−0.14, +0.35] | +0.05 [−0.07, +0.17] | +0.14 [−0.11, +0.34] | −0.17 [−0.71, +0.32] |
| Bicubic − Bilinear | R2 | +0.41 [−0.03, +0.97] | +0.64 [+0.02, +1.39] | +0.20 [−0.30, +0.83] | +0.08 [−0.58, +0.73] | −0.27 [−0.67, +0.18] | +1.05 [+0.08, +2.06] | +0.16 [−0.35, +0.76] | +0.15 [−0.83, +1.04] | +1.30 [−0.15, +3.18] |
| **C0 − Bicubic** | R0 | **+2.60 [+1.67, +3.35]** | +3.74 [+2.50, +4.62] | +1.54 [+0.56, +2.52] | +5.01 [+3.34, +6.50] | −1.29 [−3.74, +0.58] | +1.40 [+0.20, +2.56] | +1.73 [+0.91, +2.55] | +1.73 [−0.27, +3.40] | +6.36 [+3.74, +8.90] |
| **C0 − Bicubic** | R2 | **+2.73 [+1.55, +3.89]** | +3.87 [+2.48, +5.29] | +1.68 [+0.18, +3.06] | **+8.44 [+6.59, +10.30]** | −1.03 [−3.18, +0.93] | +1.37 [−0.34, +3.21] | +3.65 [+2.03, +4.98] | −1.48 [−3.87, +0.74] | +6.14 [+3.08, +9.40] |
| C0 − Bilinear (참고) | R0 | +2.62 [+1.65, +3.40] | +3.86 | +1.48 | +5.42 | −1.20 | +1.51 | +1.78 | +1.87 | +6.20 |
| C0 − Bilinear (참고) | R2 | +3.15 [+1.95, +4.34] | +4.51 | +1.88 | +8.52 | −1.30 | +2.42 | +3.81 | −1.33 | +7.44 |

- **M2(보조)** Overall: R0 35.06 / 35.82 / 40.12, R2 42.47 / 44.05 / 47.72.
  - C0 − Bicubic: R2 +3.66 [+1.90, +7.32], R0 +4.30 [+1.87, +7.09].
  - Bicubic − Bilinear: R2 +1.58 [−0.11, +2.18], R0 +0.76 [−0.02, +1.26].
  - M2 bin 값은 CI가 매우 넓다(예: Part D1–4 ±15). 전체 표는 `tables.md`에 있다.
- **관찰**
  - Bicubic − Bilinear는 Overall에서 유의하지 않다. R2에서는 Object와 D17+에서만 CI 하한이 0을 조금 넘는다.
  - C0의 이득은 Obj D5–16(R2 +8.4)과 Pattern(+6.1)에서 가장 크다. Color는 R2에서 +3.7이다.
  - Material과 Part D1–4에서는 이득이 없다.

## C. Feature Difference (val 전체 11,109 ROI, raw final-layer, fp32)

**Token / pooled / 분해** (ROI별 값의 평균 [이미지 bootstrap CI], 괄호는 중앙값)

| 지표 | Bilinear vs C0 | Bicubic vs C0 | Bicubic − Bilinear |
|---|---|---|---|
| Token cosine (M, coverage 가중) | 0.353 (0.341) | 0.350 (0.341) | −0.0023 [−0.0025, −0.0021] |
| ROI 내 token cosine p10 | 0.192 | 0.180 | −0.011 |
| Token MSE | 3.26 | 3.57 | **+0.30** [+0.30, +0.31] |
| Pooled cosine | 0.473 | 0.474 | +0.001 |
| Pooled L2 | 34.6 | 36.1 | +1.47 |
| E_mean (평균 차이 몫) | 1.78 | 1.95 | +0.17 |
| E_cent (중심화한 공간 차이 몫) | 1.48 | 1.62 | +0.14 |
| E_cent / MSE | 0.52 | 0.52 | +0.002 |
| Centred token cosine | 0.240 | 0.234 | −0.006 |
| 전체 576 cell token cosine | 0.257 | 0.253 | −0.004 |
| Padding cell만 token cosine | 0.108 | 0.101 | −0.007 |

- 분해 MSE = E_mean + E_cent는 정확히 성립한다(같은 coverage 가중).
- Token cosine 분포(유효 cell, coverage 가중)의 p10 / p50 / p90:
  - 전체 0.05 / 0.32 / 0.74
  - D1–4: −0.01 / 0.09 / 0.29
  - D17+: 0.29 / 0.60 / 0.88
- Bilinear와 Bicubic의 분포는 거의 겹친다(`figures/fig_tokcos_dist.png`).

**공간 구조 (관찰 지표)**

| 지표 | Bilinear | Bicubic | C0 |
|---|---|---|---|
| Spatial variance (채널당, M) | 0.53 | 0.76 | 1.58 |
| Effective rank (M1) | 5.6 | 6.3 | 22.4 |
| 인접 token cosine (M1) | 0.979 | 0.972 | 0.776 |
| 비대각 self-similarity 평균 (M1) | 0.79 | 0.74 | 0.44 |
| Self-similarity 차이 mean\|S_a − S_c\| | 0.362 | **0.325** | – |
| Token L2 norm | 43.1 | **46.9** | 45.5 |

**공간 주파수** (M1 직사각형의 2-D DCT, DC 제외. padding 경계가 DCT에 들어가지 않는다)
- Non-DC energy 비율 low / mid / high:
  - Bilinear 0.957 / 0.037 / 0.007
  - Bicubic 0.951 / 0.042 / 0.007
  - C0 0.601 / 0.246 / 0.153
- 절대 energy를 C0와 비교하면(중앙값): Bicubic이 low 0.69, mid 0.005, high < 0.001이다.
- 그런데 **차이(a − C0) energy의 61%는 low band에 있다.** 보간 feature에 고주파가 거의 없는 것은 사실이지만, 차이의 대부분은 저주파(거친 layout과 평균) 쪽이다. 높은 energy를 좋은 feature로 해석하지 않는다.

**ROI 크기별** (Bilinear / Bicubic, ROI 평균)

| level | bin | n | token cos | pooled cos | E_cent / MSE | \|dS\| | var a/c |
|---|---|---|---|---|---|---|---|
| object | D1–4 | 102 | 0.240 / 0.241 | 0.354 / 0.358 | 0.45 / 0.44 | 0.48 / 0.44 | 0.17 / 0.28 |
| object | D5–16 | 1,384 | 0.323 / 0.322 | 0.476 / 0.479 | 0.53 / 0.53 | 0.45 / 0.40 | 0.24 / 0.38 |
| object | D17+ | 1,654 | 0.584 / 0.579 | 0.756 / 0.756 | 0.75 / 0.76 | 0.27 / 0.23 | 0.52 / 0.70 |
| part | D1–4 | 2,871 | 0.114 / 0.114 | 0.155 / 0.155 | 0.28 / 0.27 | 0.40 / 0.38 | 0.11 / 0.18 |
| part | D5–16 | 2,889 | 0.302 / 0.301 | 0.426 / 0.428 | 0.46 / 0.47 | 0.41 / 0.37 | 0.24 / 0.39 |
| part | D17+ | 2,209 | 0.578 / 0.573 | 0.740 / 0.741 | 0.71 / 0.72 | 0.27 / 0.22 | 0.51 / 0.69 |

- n_in과의 Spearman(Bicubic): token cosine +0.88, E_mean −0.91, E_cent/MSE +0.90.
- 작은 ROI는 **평균 표현이 다르고**(D1–4에서 MSE의 72%가 E_mean), 큰 ROI는 평균은 비슷하지만 **공간 패턴이 다르다**(D17+에서 MSE의 74%가 E_cent). `figures/fig10_size_vs_featdiff.png`, `fig_decomposition.png`.

**PACO mask 영역** (mask 11,109개 모두 정렬. interior ≥ 0.9, boundary 0.1–0.9, background ≤ 0.1, 실행 전 고정)
- 전 영역에서 차이가 크다. 영역 간 차이는 작다.
- 같은 ROI 안에서 interior − background token cosine은 Bilinear +0.049 [+0.047, +0.052], Bicubic +0.055.
- Boundary는 그 중간이다(예: object 전체 interior 0.49 / boundary 0.45 / background 0.40).
- 차이가 경계에 집중되지 않는다.

## D. Prediction-linked Analysis (Bicubic vs C0, 같은 ROI-attribute)

**정오답 정의.**
- (조건, reader, level, attribute)마다 PACO dev에서 F1을 최대로 하는 threshold를 쓴다(val을 보지 않고 고정).
- M0 unit 106개 중 dev positive 또는 negative가 5개 미만인 6개는 제외했다.
- 적용 pair는 521,678개(positive 45,545 / negative 476,133)다.

**R2 그룹 수.**
- Positive: G1 3,429 (7.5%) / G2 32,912 (72.3%) / G3 5,945 (13.1%) / G4 3,259 (7.2%).
- Negative: G1 10,472 / G4 9,427.
- Threshold 기준 정오답의 순변화는 작다(positive +170). M0 +2.7의 대부분은 threshold를 넘나드는 변화가 아니라 **순위 변화**에서 온다.

**그룹별 feature** (R2, positive label, level × D-bin 층화, ROI-level SD 단위, 이미지 bootstrap CI)

| Bicubic vs C0 지표 | G1 − G2 | G4 − G2 | G1 − G4 |
|---|---|---|---|
| 1 − token cosine | +0.11 [+0.09, +0.14] | +0.16 [+0.13, +0.18] | −0.04 [−0.08, −0.01] |
| 1 − pooled cosine | +0.11 [+0.08, +0.13] | +0.18 [+0.15, +0.20] | −0.07 [−0.10, −0.04] |
| E_mean | +0.05 [+0.03, +0.07] | +0.14 [+0.12, +0.16] | −0.09 [−0.12, −0.06] |
| **E_cent** | +0.08 [+0.04, +0.11] | **−0.11** [−0.15, −0.08] | **+0.17** [+0.13, +0.22] |
| **Self-sim 차이** | +0.08 [+0.05, +0.12] | **−0.06** [−0.09, −0.03] | **+0.14** [+0.09, +0.18] |
| erank a/c | −0.06 [−0.09, −0.04] | – | – |
| high-band a/c (log) | −0.07 [−0.10, −0.04] | – | – |
| var a/c (log) | −0.03 [−0.06, −0.00] | – | – |

- 층화 전 G1의 ROI는 G2보다 작다(n_in 25.5 vs 37.2). 층화하면 G1 − G2 차이는 원래 크기(0.2–0.28 SD)의 절반 이하로 줄어든다.
- R0도 방향이 같다(`tables.md`).

**상관** (ROI당 값 하나. 점수 변화 = 그 ROI positive label의 C0 − Bicubic rank margin 평균)
- Logit 대신 rank margin을 쓴 이유: logit SD 비율 C0/Bicubic이 unit 중앙값 1.04(IQR 0.96–1.09)로 대체로 비슷하지만, scale-free이고 AP와 직접 연결되는 지표를 쓰기 위해서다.
- Partial ρ = rank n_in과 level을 회귀로 제거한 Spearman, 이미지 bootstrap 300회.

| Bicubic vs C0 지표 | 전체 ρ | **partial ρ** | D1–4 | D5–16 | D17+ | color | material | pattern |
|---|---|---|---|---|---|---|---|---|
| 1 − token cosine | −0.10 | −0.05 [−0.08, −0.03] | −0.13 | −0.03 | −0.00 | +0.00 | −0.12 | −0.02 |
| 1 − pooled cosine | −0.10 | −0.06 [−0.09, −0.04] | −0.15 | −0.04 | −0.00 | −0.00 | −0.13 | −0.03 |
| E_mean | −0.11 | −0.08 [−0.10, −0.05] | −0.15 | −0.05 | −0.00 | −0.01 | −0.12 | −0.05 |
| **E_cent** | +0.16 | **+0.12 [+0.10, +0.14]** | +0.20 | +0.09 | +0.02 | +0.05 | +0.18 | +0.09 |
| **Self-sim 차이** | +0.06 | **+0.12 [+0.10, +0.14]** | +0.14 | +0.08 | +0.03 | +0.05 | +0.14 | +0.10 |
| var a/c (log) | +0.05 | −0.04 | −0.01 | −0.05 | −0.03 | −0.03 | −0.05 | −0.02 |
| erank a/c | +0.06 | −0.01 | +0.03 | −0.01 | −0.01 | +0.00 | −0.00 | −0.04 |
| high-band a/c (log) | +0.04 | −0.04 | −0.01 | −0.05 | −0.01 | −0.01 | −0.02 | −0.05 |

- 보조 지표에서도 E_cent의 partial ρ는 양수다. Negative label rank margin +0.15, dev 보정 logit margin +0.27(`tables.md`).
- R0에서는 더 약하다. E_cent +0.06, self-sim +0.01, E_mean −0.04.
- Bicubic − Bilinear: ROI별 token cosine 변화와 rank margin 변화의 partial ρ는 R2 +0.03 [+0.01, +0.05], R0 +0.06이다. 평균 점수 변화는 +0.001 수준이다.
- 산점도 `figures/fig11_featdiff_vs_scorediff_r2.png`에서 decile 평균은 0 근처에서 ±0.02 이내로만 움직인다.
- 지표 11개 × 여러 하위 집합을 동시에 보므로, |ρ| < 0.05 정도의 값은 다중 비교를 고려하면 의미를 두지 않는다.

**실제 성능 차이를 설명하는 지표가 있는가?**
- 크게 설명하는 지표는 **없다.** 공간 구조 차이(E_cent, self-similarity 차이)가 방향까지 일관되게 C0 이득과 같이 움직이는 유일한 지표군이지만, 크기는 ρ ≈ 0.12(분산의 약 1.5%)다.
- Spatial variance, effective rank, 고주파 energy의 **비율**(Bicubic이 C0 구조를 얼마나 재현했는가)은 C0 이득과 양의 연결이 없다(partial ρ −0.04~0).

## E. Visualization

- 정렬: [`figures/align/align_*.png`](figures/align/) 12장(seed 0으로 사전 고정).
- 고정 ROI 12개(object/part × D-bin에서 2개씩, 사전 고정): [`figures/rois/`](figures/rois/).
  - 각 그림: 원본 + GT, C0 입력, Bilinear/Bicubic/C0 PCA-RGB(같은 투영), token cosine heatmap(척도 [−0.2, 1] 고정), Bilinear−C0와 Bicubic−C0 MSE map(같은 척도), Bicubic/C0 self-similarity와 그 차이([−1, 1]).
  - Padding은 회색이고 모든 수치에서 제외했다.
  - 예: `fixed06`(part D1–4, bench:back)에서 보간 feature는 ROI 전체가 거의 한 색이고(self-sim 약 0.9), C0는 구조가 다양하다(token cosine 0.31). `fixed04`(n_in 432)에서는 세 조건이 거의 같다(cosine 0.97).
- G1–G4 예시(사전 규칙: positive pair 중 그룹마다 3개를 rng 0으로 균등 추출): [`figures/groups_r2/`](figures/groups_r2/), [`figures/groups_r0/`](figures/groups_r0/). 고른 pair 목록은 `group_examples_r2.json`, `group_examples_r0.json`.
- 크기 vs feature 차이: `figures/fig10_size_vs_featdiff.png`. Feature 차이 vs 점수 차이: `figures/fig11_featdiff_vs_scorediff_r2.png`. 분해, DCT, token cosine 분포: `figures/fig_decomposition.png`, `fig_dct_bands.png`, `fig_tokcos_dist.png`.

## F. Conclusions and Limitations

**Q1. Bicubic Feature와 C0 Crop Feature 사이에서 가장 크게 다른 부분은 무엇인가?**
- *관찰.*
  - ROI의 평균 표현이 크게 다르다: pooled cosine 0.47, D1–4는 0.16, 오차의 절반 이상이 평균 차이.
  - ROI 내부 공간 구조가 크게 다르다: 보간 feature는 인접 token cosine 0.97, effective rank 6, mid/high 주파수 energy가 C0의 1% 미만으로 거의 평평하다.
  - 어느 쪽이 지배적인지는 ROI 크기로 갈린다. 작을수록 평균 차이, 클수록 공간 패턴 차이다.
  - 차이는 mask 내부·경계·배경 전체에 퍼져 있고, 내부가 약간(+0.05 cosine) 덜 다르다.
  - Bicubic은 공간 분산과 self-similarity 차이를 조금 줄였지만, token MSE와 pooled L2는 늘었고 token cosine은 그대로다. 즉 C0에 더 가까워지지 않았다.
- *해석(추정).* C0 crop은 같은 영역을 encoder 안에서 더 많은 token으로 처리하므로, 같은 위치라도 평균 표현 자체가 다른 값이 된다. 보간은 이미 계산된 coarse map을 다시 읽는 것이라 이 차이를 만들 수 없다. Bicubic의 추가 분산(norm overshoot 46.9 vs C0 45.5)은 C0 구조와 정렬되지 않은 kernel 효과로 보인다.

**Q2. 그 차이가 실제 속성 판별 성능의 차이와 연결되는가?**
- *관찰.*
  - 조건 수준에서는 연결된다. C0가 feature도 다르고 M0도 +2.6~2.7 높다.
  - Bicubic은 feature를 C0 쪽으로 옮기지 못했고 성능도 Bilinear와 유의하게 다르지 않다(R2 +0.41 [−0.03, +0.97]).
  - **ROI 단위에서는 연결이 약하다.**
    - 공간 구조 차이(E_cent, self-similarity 차이)만 C0 이득과 일관된 방향으로 움직인다: partial ρ +0.12, G1 > G2 > G4, 작은·중간 ROI와 material/pattern에서 더 분명.
    - 평균 표현 차이는 C0 이득보다 예측 불일치(G1과 G4 모두)와 같이 움직인다.
    - Feature 차이가 가장 큰 Part D1–4에서는 C0의 성능 이득이 없다.
- *결론.* "Feature 차이가 크면 C0가 더 잘 맞힌다"는 ROI 단위의 강한 연결은 **확인되지 않았다.** 확인된 것은 공간 구조 차이와의 약한(ρ ≈ 0.1) 연결뿐이며, 이것도 인과를 보여주지 않는다.
  - E_cent는 C0 자신의 공간 다양성에 따라 커진다. 그래서 "보간이 잃은 것"이 아니라 "공간적으로 정보가 많은 ROI"를 나타내는 지표일 수 있다.
  - 이번 분석은 공유 feature에 세부 정보가 없다는 것을 보여주지 않는다. Reader가 R0/R2로 제한되어 있고, 보간 feature에 남아 있는 정보를 다른 방식으로 읽을 수 있는지는 시험하지 않았다.

**한계.**
- 학습 seed 0 하나. CI는 평가 표본의 불확실성만 반영한다.
- 재학습한 C0의 flip 순서가 다른 조건과 다르다(위 A 참고). MSD C0로 바꿔도 결론은 같다.
- 정오답은 dev F1-max threshold에 의존하고, AP 기반 성능 차이 중 threshold로 잡히는 부분은 작다.
- 상관은 지표 11개 × 하위 집합의 다중 비교이며 효과 크기가 작다.
- Raw feature만 분석했다(정규화 feature 분석 없음).
- Bicubic은 PyTorch A = −0.75 정의다.
- M2 bin 값은 CI가 넓다.
- Val만 평가했다.


## G. PACO Test-primary (one-time, fixed checkpoints)

Test-primary = PACO test ROIs outside the decoder COCO pool: 26,442 ROIs / 4,848 images (object 7,493, part 18,949); all test 43,718 ROIs. Checkpoints = dev-selected epochs (bil/r0 ep10, bil/r2 ep6, bic/r0 ep10, bic/r2 ep6, c0/r0 ep10, c0/r2 ep5). Val re-score with the same code reproduced the saved val scores bit-exactly. Train/test image overlap: none (asserted).

### Test 요약 (사용자 승인 후 1회, 고정 checkpoint, 선택이나 조정 없음)
- **Test-primary M2(주 지표)**:
  - C0 − Bicubic: R2 **+5.43 [+3.59, +6.72]**, R0 +4.03 [+2.81, +5.46]
  - Bicubic − Bilinear: R2 +0.07 [−0.56, +0.79], R0 −0.15 [−0.34, +0.28]
- **Test M0(참고)**: C0 − Bicubic R2 +2.76 [+2.14, +3.59], R0 +2.83; Bicubic − Bilinear R2 +0.10 [−0.14, +0.34].
- **Val과 같은 패턴**:
  - C0 이득은 Obj D5–16에서 가장 크다(M2 R2 +16.7, M0 +6.7). Pattern(M2 +8.3)과 Color(+5.4)가 그다음이다.
  - Part D1–4는 M0에서 이득이 없다(−0.02 [−1.65, +1.48]). M2는 +2.24 [−3.12, +6.66]로 CI에 0이 들어간다.
  - Bicubic은 Bilinear와 구분되지 않는다.
- Val과 다른 점: Material이 M2에서 +4.94 [+1.26, +6.12]이지만 M0에서는 −1.29 [−2.70, +0.28]로, 지표에 따라 다르다.
- Test에서는 feature나 예측 연결 분석을 하지 않았다(Val 분석만).

### M2 (primary)

| Method | Reader | Overall | Object | Part | Obj D5-16 | Part D1-4 | Obj D17+ | All D17+ | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bilinear | R0 | 33.67 | 34.26 | 33.08 | 32.74 | 29.66 | 38.05 | 39.26 | 22.76 | 31.20 | 34.97 |
| Bicubic | R0 | 33.52 | 34.17 | 32.87 | 33.06 | 30.67 | 38.15 | 39.28 | 22.98 | 30.78 | 35.05 |
| C0 Crop | R0 | 37.55 | 38.63 | 36.46 | 45.94 | 36.01 | 40.49 | 41.45 | 25.53 | 33.39 | 45.39 |
| Bilinear | R2 | 40.42 | 40.44 | 40.40 | 37.90 | 33.56 | 47.07 | 49.31 | 30.74 | 34.39 | 44.95 |
| Bicubic | R2 | 40.49 | 40.68 | 40.31 | 36.36 | 34.08 | 47.02 | 49.48 | 31.13 | 34.95 | 45.33 |
| C0 Crop | R2 | 45.92 | 48.21 | 43.63 | 53.06 | 36.33 | 50.75 | 52.62 | 36.56 | 39.89 | 53.62 |

**Paired image bootstrap (M2, B=1000, seed 0)**

| Contrast | Reader | Overall | Object | Part | Obj D5-16 | Part D1-4 | Obj D17+ | All D17+ | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic - Bilinear | R0 | -0.15 [-0.34, +0.28] | -0.08 [-0.35, +0.45] | -0.21 [-0.53, +0.32] | +0.32 [-0.37, +1.02] | +1.01 [-0.70, +3.03] | +0.10 [-0.22, +0.53] | +0.02 [-0.34, +0.41] | +0.22 [-0.11, +0.43] | -0.43 [-0.77, +0.24] | +0.08 [-0.49, +0.86] |
| C0 Crop - Bicubic | R0 | +4.03 [+2.81, +5.46] | +4.46 [+2.97, +6.90] | +3.59 [+2.04, +4.86] | +12.88 [+7.07, +15.31] | +5.33 [-0.41, +8.54] | +2.34 [+0.79, +5.31] | +2.17 [+1.27, +4.86] | +2.55 [+1.83, +3.87] | +2.61 [+0.50, +4.30] | +10.34 [+7.76, +13.29] |
| C0 Crop - Bilinear | R0 | +3.88 [+2.75, +5.48] | +4.38 [+3.00, +6.98] | +3.39 [+1.87, +4.74] | +13.19 [+7.27, +15.73] | +6.35 [+0.19, +9.27] | +2.44 [+0.97, +5.44] | +2.18 [+1.29, +4.93] | +2.77 [+1.96, +4.00] | +2.18 [+0.21, +4.12] | +10.42 [+7.90, +13.43] |
| Bicubic - Bilinear | R2 | +0.07 [-0.56, +0.79] | +0.24 [-0.85, +1.20] | -0.10 [-0.78, +0.73] | -1.54 [-2.73, +0.99] | +0.53 [-1.25, +2.66] | -0.05 [-1.61, +1.58] | +0.17 [-1.11, +1.83] | +0.40 [-0.23, +0.90] | +0.56 [-0.74, +1.54] | +0.38 [-1.31, +2.08] |
| C0 Crop - Bicubic | R2 | +5.43 [+3.59, +6.72] | +7.53 [+5.06, +9.26] | +3.33 [+1.40, +4.88] | +16.69 [+11.38, +19.61] | +2.24 [-3.12, +6.66] | +3.73 [+0.84, +5.62] | +3.14 [+0.13, +5.10] | +5.42 [+4.46, +6.98] | +4.94 [+1.26, +6.12] | +8.29 [+4.44, +11.68] |
| C0 Crop - Bilinear | R2 | +5.50 [+3.69, +6.85] | +7.77 [+5.33, +9.48] | +3.23 [+1.28, +4.94] | +15.16 [+11.03, +18.91] | +2.77 [-2.03, +6.79] | +3.68 [+0.72, +5.76] | +3.31 [+0.82, +5.39] | +5.82 [+4.74, +7.40] | +5.50 [+1.36, +6.90] | +8.67 [+4.90, +11.84] |

### M0 (reference)

| Method | Reader | Overall | Object | Part | Obj D5-16 | Part D1-4 | Obj D17+ | All D17+ | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bilinear | R0 | 33.23 | 34.68 | 31.79 | 37.16 | 32.36 | 37.50 | 37.10 | 16.71 | 58.89 | 34.59 |
| Bicubic | R0 | 33.25 | 34.79 | 31.73 | 37.38 | 32.41 | 37.61 | 37.20 | 16.77 | 58.95 | 34.50 |
| C0 Crop | R0 | 36.08 | 38.19 | 34.00 | 42.81 | 31.49 | 39.53 | 38.84 | 18.87 | 60.48 | 41.71 |
| Bilinear | R2 | 38.47 | 39.56 | 37.39 | 41.23 | 36.76 | 42.76 | 42.93 | 21.53 | 64.61 | 41.59 |
| Bicubic | R2 | 38.56 | 39.82 | 37.33 | 41.08 | 36.49 | 43.29 | 43.19 | 21.55 | 64.77 | 41.70 |
| C0 Crop | R2 | 41.33 | 43.43 | 39.26 | 47.76 | 36.47 | 45.22 | 45.23 | 25.01 | 63.48 | 48.28 |

**Paired image bootstrap (M0, B=1000, seed 0)**

| Contrast | Reader | Overall | Object | Part | Obj D5-16 | Part D1-4 | Obj D17+ | All D17+ | Color | Material | Pattern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bicubic - Bilinear | R0 | +0.02 [-0.07, +0.10] | +0.10 [+0.01, +0.21] | -0.06 [-0.19, +0.05] | +0.23 [+0.09, +0.38] | +0.05 [-0.16, +0.24] | +0.11 [+0.02, +0.25] | +0.10 [+0.02, +0.19] | +0.06 [-0.00, +0.12] | +0.06 [-0.14, +0.22] | -0.09 [-0.38, +0.22] |
| C0 Crop - Bicubic | R0 | +2.83 [+2.33, +3.32] | +3.41 [+2.79, +4.00] | +2.26 [+1.65, +2.85] | +5.43 [+4.29, +6.64] | -0.93 [-2.07, +0.25] | +1.92 [+1.25, +2.69] | +1.64 [+1.05, +2.27] | +2.10 [+1.76, +2.54] | +1.53 [+0.42, +2.61] | +7.21 [+5.41, +8.89] |
| C0 Crop - Bilinear | R0 | +2.85 [+2.32, +3.34] | +3.51 [+2.88, +4.12] | +2.20 [+1.58, +2.80] | +5.66 [+4.56, +6.84] | -0.88 [-2.06, +0.32] | +2.03 [+1.32, +2.83] | +1.74 [+1.13, +2.37] | +2.16 [+1.80, +2.59] | +1.58 [+0.45, +2.65] | +7.12 [+5.32, +8.75] |
| Bicubic - Bilinear | R2 | +0.10 [-0.14, +0.34] | +0.26 [-0.06, +0.60] | -0.06 [-0.34, +0.21] | -0.15 [-0.65, +0.41] | -0.26 [-0.60, +0.19] | +0.52 [+0.04, +0.96] | +0.26 [-0.13, +0.70] | +0.03 [-0.19, +0.28] | +0.16 [-0.29, +0.68] | +0.11 [-0.77, +0.90] |
| C0 Crop - Bicubic | R2 | +2.76 [+2.14, +3.59] | +3.61 [+2.85, +4.55] | +1.93 [+1.09, +2.94] | +6.68 [+5.42, +8.06] | -0.02 [-1.65, +1.48] | +1.93 [+0.95, +3.13] | +2.04 [+1.20, +3.27] | +3.46 [+2.77, +4.41] | -1.29 [-2.70, +0.28] | +6.59 [+4.10, +9.15] |
| C0 Crop - Bilinear | R2 | +2.86 [+2.21, +3.69] | +3.87 [+3.16, +4.78] | +1.87 [+1.03, +2.89] | +6.53 [+5.25, +7.99] | -0.28 [-1.86, +1.16] | +2.46 [+1.53, +3.59] | +2.30 [+1.45, +3.56] | +3.49 [+2.84, +4.47] | -1.13 [-2.44, +0.49] | +6.69 [+4.10, +9.08] |

Units (M2 category units / M0 units) per subset: all: 4173 / 113; D1-4: 508 / 72; D5-16: 1521 / 100; D17+: 1620 / 110. NaN = no eligible unit (never imputed).


## 재현

- 코드: `src/ivc/{common,train,features,masks,align,analyze,viz,group_examples}.py`
- 설정: `configs/ivc/probe_{bil,bic,c0}.yaml`, `configs/ivc/sample_rois.json`(사전 고정 ROI), `configs/ivc/protocol.md`
- 실행: `scripts/ivc/queue.sh`(smoke, probe, align), `scripts/ivc/features_sharded.sh`, `scripts/ivc/run_c0.sh`, 마지막으로 `python -m src.ivc.analyze --run output/2026-11-05/ivc`와 `group_examples` / `viz`
- Checkpoint: `output/2026-11-05/ivc/probe_<c>/<r0|r2>/ep<k>.pt`. 선택 epoch은 Bilinear R0 10 / R2 6, Bicubic R0 10 / R2 6, C0 R0 10 / R2 5. Val/dev score는 `scores_{val,dev}_ep<k>.npy`, R0 통계는 `probe_<c>/r0_stats.pt`.
- 로그와 상태: `queue_status.tsv`, `logs/`. QAT 중단·재개 기록: `qat_pause/`.
- Test: 사용자 승인 후 1회 평가. `src/ivc/test_eval.py`(val 재채점이 저장된 val score와 bitwise 일치하는 것을 확인한 뒤 test scoring), `src/ivc/test_report.py`. 결과는 `test/`.
