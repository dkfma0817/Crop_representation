# Decoder Study on CLIP-L G1-cross — 종합 보고서 (2026-11-07)

- 실행: 2026-11-07 17:08–22:39 UTC, 물리 GPU 2·3만 사용.
- 평가: PACO 공식 Val + Official-protocol Dev Oracle AP. **Test는 실행하지 않음.**
- 유의성 표기: \* = image-level paired bootstrap(B=1000) 95% CI가 0을 포함하지 않음.
- 아래 표들(Table A–D, Sanity)은 `src/decstudy/report.py`가 결과 파일에서 직접 만들었다. 이 절의 해석은 그 표를 근거로 썼다.

## 결론 (먼저)

1. **Decoder를 어떻게 바꿔도 Encoder Only보다 높은 판별 AP를 얻지 못했다.**
   - Val Obj/Part 기준으로 Encoder Only(52.08 / 54.03)를 넘은 decoder 경로는 하나도 없다.
   - 가장 가까운 L1도 −0.43 / −1.00(n.s.)이다.
2. **Residual + Bilinear Query(D1)는 기존 Decoder와 성능이 같다.**
   - D1 − 기존 Decoder: +0.28 / +0.24, n.s.
   - D1 − Encoder Only: Part −2.12\*로, 기존 Decoder의 하락이 그대로 남았다.
   - 출발점(Z=B)을 바꾸고 zero-init residual로 정보 보존을 보장해도, 같은 crop-feature matching loss로 12k step을 학습하면 기존 decoder와 같은 지점(COCO dev token cos 0.699 vs 0.698)에 도달한다.
3. **학습된 residual이 bilinear feature의 material/pattern 정보를 오히려 지운다.**
   - D1 − B0: Obj Material −6.32\*, Part Pattern −4.24\*.
   - Teacher와의 token cosine은 0.646 → 0.699로 올라갔지만 판별 AP는 내려갔다.
   - **Crop feature 복원 지표 개선 ≠ 속성 판별 개선**이 이번에도 재현됐다.
4. **추가 Loss는 feature 지표를 크게 바꿨지만 AP 개선으로 이어지지 않았다.**
   - L1(covariance): cov err 0.68 → 0.39로 개선됐다. D1이 잃은 Obj Material은 일부 되찾았다(L1 − D1 +4.18\*). 그러나 Obj/Part 전체는 n.s.이고, Dev에서는 반대 방향이다(−1.06 / −0.38).
   - L2(relational): relation r 0.894 → 0.912로 개선됐다. 그러나 L2 − Encoder Only가 Obj −2.61\*, Part −1.90\*로 가장 나쁘다.
5. **이번 Val에서는 Crop 재인코딩 자체의 우위가 Material에 없다.**
   - CLIP-L C0 crop은 Material에서 Full보다 유의하게 나쁘다: Obj −4.31\*, Part −6.72\*(vs Encoder Only).
   - 실패 분석에서도 "두 Full 모델은 틀리고 Crop만 맞히는" pair(7.5%)와 "Crop만 틀리는" pair(8.6%)가 비슷하다.
   - 따라서 **"Material / Part Pattern = crop이 푸는 격차"라는 전제가 이번 백본·Val에서는 성립하지 않는다.** Crop의 이점은 Color(Part +4.91\*)와 Object Pattern(+5.87, n.s.)에 집중돼 있다.

## A–E 답

**A. Decoder의 기존 성능 하락을 해결했는가? — 아니오.**

| 비교 (Val) | Obj | Part |
|---|---|---|
| 기존 Decoder − Encoder Only | −1.54 (n.s.) | −2.36\* |
| D1 − Encoder Only | −1.26 (n.s.) | −2.12\* |
| D1 − 기존 Decoder | +0.28 | +0.24 |

- Dev에서는 D1 − Encoder Only가 +0.92 / −0.44(n.s.)로 하락이 보이지 않는다. 반면 기존 Decoder는 Dev Part −3.00\*이다.
- 정리하면 Part 하락은 Val에서는 그대로 남았고, Dev에서는 완화된 정도다. "해결"이라고 할 근거는 없다.

**B. Bilinear 입력 자체가 도움이 되었는가? — 전체 AP에는 아니오.**
- B0 − Encoder Only: Val −0.93 / −0.71(n.s.), Dev +1.38 / −0.08(n.s.).
- Obj Material만 Val +2.71(n.s., CI −4.2~+7.7), Dev +5.78로 두 평가에서 같은 방향이지만 유의하지 않다.
- Decoder 경로 안에서 보면 B0는 D1·L1·L2보다 나쁘지 않은 가장 안전한 "crop-grid 표현"이다. D1이 B0보다 Obj Material −6.32\*, Part Pattern −4.24\* 나쁘다.

**C. Decoder 사용 시 Encoder Only보다 실제로 더 높은 판별 AP를 얻었는가? — 아니오.**

| 비교 (Val) | Obj | Part |
|---|---|---|
| D1 − Encoder Only | −1.26 | −2.12\* |
| L1 − Encoder Only | −0.43 | −1.00 |
| L2 − Encoder Only | −2.61\* | −1.90\* |

- Dev에서도 Encoder Only를 유의하게 넘는 경로가 없다.

**D. 추가 Loss 중 어떤 것이 실제 PACO AP 개선으로 연결되었는가? — 두 Loss 모두 전체 AP 개선은 없다.**
- **L1 − D1:**
  - Val: Obj +0.83, Part +1.13(n.s.). Obj Material +4.18\*, Part Pattern +4.10(경계, n.s.).
  - Dev: −1.06 / −0.38. Val의 이득이 Dev에서 재현되지 않았다.
  - 해석: covariance loss는 D1이 망가뜨린 material/pattern을 B0 수준 근처로 되돌린 것이지, Encoder Only를 넘는 새 정보를 만든 것이 아니다(L1 − Encoder Only Obj Material +0.56).
- **L2 − D1:**
  - Part Pattern +3.95\*이지만 Obj −1.35, Dev −1.87 / −1.76이다.
  - L2 − Encoder Only가 가장 나쁘다.
- **Feature 지표와 AP의 괴리:** feature 지표는 각 Loss가 겨냥한 대로 좋아졌다(L1 cov cosine 0.62 → 0.77, L2 relation r 0.894 → 0.912). AP는 그렇지 않았다.
- **L2 해석 시 주의:** teacher ROI 간 cosine이 0.85±0.055로 몰려 있어, τ=0.1에서 target 분포가 거의 균등하다(entropy 비율 0.98). L2는 사실상 "중심화된 ROI 유사도의 MSE"로 동작했다. τ는 사전에 고정했고 탐색하지 않았다.

**E. Crop 재인코딩 없이 남은 세부 속성 판별 격차를 줄일 가능성이 확인되었는가? — 이번 설정에서는 확인되지 않았다. 격차의 위치도 예상과 다르다.**
- CLIP-L C0 crop − Encoder Only(Val):

  | | Obj | Part |
  |---|---|---|
  | 전체 AP | +1.56 (n.s.) | −0.41 |
  | Color | +1.60 | +4.91\* |
  | Pattern | +5.87 (n.s.) | −0.41 |
  | Material | −4.31\* | −6.72\* |

- 남은 crop 우위는 Color와 Object Pattern에 있다. Material에서는 이미 Full(문맥)이 낫다.
- 어떤 decoder도 Color / Object Pattern 격차를 줄이지 못했다.
  - Obj Pattern: D1 52.22, L1 52.10, L2 51.96으로 Encoder Only 52.48 이하.
  - Part Color: D1 51.26 등으로 Encoder Only 51.35 이하.

## 1. 실험 설정 요약

- **경로 (모두 같은 G1-cross EMA encoder sha256 `b740f4c4…`, 이미지당 full 인코딩 1회, ROI RGB 재인코딩 0회):**
  - Encoder Only
  - 기존 공동학습 Decoder
  - B0: bilinear ROIAlign을 C0 crop letterbox grid(24×24)에 aspect 보존으로 정렬
  - D1: Query = LN(B + pos) + q_grid, zero-init `out`, Z = B + Δ
  - L1: D1 + λ_cov 0.655 · L_cov
  - L2: D1 + λ_rel 1.56 · L_rel
- **Decoder 학습:** G1-cross와 동일.
  - COCO 12k stream(16 images × 4 ROI/step, PACO val/test 일치 2장 제외, 고유 이미지 50,489장)
  - AdamW 3e-4, warmup 600 + cosine, EMA, 12,000 step
  - Teacher: 원본 CLIP-L/336 C0 crop
  - L_original = 1.11 × L_detail
  - PACO attribute label 미사용
  - D1·L1·L2는 같은 초기 상태(sha256 동일)에서 한 프로세스 안에서 독립 optimizer/EMA로 학습했다. 공유한 것은 teacher/encoder forward 결과뿐이다.
- **R2/Head:** 모든 경로가 같은 yaml/seed/schedule(10 epoch cosine)을 쓰고, **epoch 6을 사전 고정**했다.
  - 근거: 기존 baseline 보고가 모두 ep6이었고, Dev 곡선에서 ep10이 2–4 AP 낮았다.
  - 기존 Encoder Only / 기존 Decoder probe도 ep6 checkpoint를 썼다. 기존 Decoder의 Test 보고(38.11)는 ep9였다.
- **λ 결정:** init gradient-norm 비율의 median(16 step)으로 1회 정했고 탐색하지 않았다.
  - L_cov: L2-normalised valid token(cc > 0, N ≥ 4)의 covariance, 정규화 Frobenius 오차
  - L_rel: ccov-pooled ROI 벡터, 64 ROI/step, τ = 0.1, KL(teacher‖student), 대각 제외

## 2. Baseline 재현 / 검증
- 같은 경로를 Dev 추론으로 다시 계산했을 때 기존 probe의 dev M0와 0.002 이내로 같다(g1cross_dec 48.849 vs 48.847).
- 기존 Encoder Only / MaskEmbed / CLIP의 Dev AP는 2026-11-07 Dev AP 실험 수치와 일치한다(같은 파일 재사용).
- **Val과 Test의 순위가 다르다.** Val에서는 MaskEmbed(48.77 / 53.84)가 Encoder Only(52.08 / 54.03)보다 Obj −3.31\* 낮다. Test에서는 두 모델이 비슷했다(39.32 vs 39.21).
  - "가장 좋은 후보 − MaskEmbed"(D1, Dev로 선택)는 Val에서 Obj +2.05 / Part −1.93(n.s.)이다.
  - Val 2,410장의 type 단위 CI는 ±4–8 AP로 넓다. Type 수준의 단일 차이는 신중히 읽어야 한다.

## 3. 공통 실패 분석 (Table D)
- 두 Full 모델(G1-cross, MaskEmbed)이 모두 틀린 pair 비율: Obj Material 16.9%, Part Pattern 16.6%, Obj Pattern 12.5%.
- 그중 Crop이 맞힌 비율은 44.3% / 47.8% / 53.8%로, **pair 순위의 우연 수준(50%) 근처**다.
- 반대 방향(Crop만 틀리고 두 Full은 맞힘)도 8.6% / 8.8% / 5.8%로, Crop만 맞히는 비율(7.5% / 7.9% / 6.7%)과 비슷하다.
  - → "Full은 일관되게 못 하고 Crop은 일관되게 하는" 반복 실패 집합은 작다.
  - crop-only 비율 30% 이상인 unit: 9/216, 26/458, 9/245.
  - crop-only pair 중 그런 unit에 속하는 비율: 17% / 25% / 15%.
- **작은 ROI 쪽으로 기운다.**
  - crop-only 비율: 작은 ROI 8.9% vs 큰 ROI 6.1%(Obj Material), 9.2% vs 6.7%(Part Pattern).
  - crop-only pair의 ROI 변 길이(336 입력 기준) median: Part Pattern 25 px(전체 32 px), Obj Material 42 px(전체 47 px).
- **대표 12사례**(`failure/viz/`, 336 letterbox 입력 + 원본 tight crop): bucket/plastic, shoe:quarter/striped, mouse:logo/plain 등 대부분 336 입력에서 변이 10–35 px인 ROI다.

## 4–7. 경로별 효과 요약 (Table A–C)
- **B0:** 파라미터 없이 Encoder Only와 통계적으로 같은 수준이다(Val −0.93 / −0.71, Dev +1.38 / −0.08). Crop-grid 표현으로 옮기는 것만으로 정보가 크게 손실되지는 않는다.
  - feature 수준: token 공간 분산이 teacher의 9%뿐인 매우 평탄한 표현이다(var ratio 0.093).
- **D1:** Δ가 실제로 쓰인다.
  - ‖Δ‖/‖B‖ = 1.37
  - Δ 제거 시 logit 변화 0.90
  - box 밖 key를 교체하면 Δ가 17% 변한다(문맥을 참조함).
  - 그러나 cos(Δ, T − B) = 0.08로, Δ의 방향은 teacher 쪽과 거의 무관하다.
  - Teacher 유사도는 기존 Decoder와 같은 수준에 도달했고 AP도 같다.
- **L1:** covariance를 가장 잘 맞추지만(cov err 0.39), 공간 분산이 teacher의 1.46배로 과대해지고 token cos는 낮아진다(0.678). AP는 D1보다 조금 낫고 Encoder Only보다 조금 낮다(둘 다 n.s.).
- **L2:** relation 보존이 가장 좋지만(r 0.912) Obj AP와 Reflectance가 떨어진다(Obj Reflectance 53.83).

## 8. Object Material / Part Pattern 개선 여부
- **Encoder Only 대비 유의한 개선은 없다.**
  - Obj Material: B0 +2.71, L1 +0.56, D1 −3.61, L2 −4.19
  - Part Pattern: L1 −0.29, L2 −0.44, D1 −4.39
- 이 두 type은 이번 Val에서 Crop도 Encoder Only보다 낫지 않다(Obj Material −4.31\*, Part Pattern −0.41). 따라서 "crop 격차를 메운다"는 목표 자체가 성립하지 않았다.

## 9. Sanity 요약
- Encoder hash: 학습 전후 동일, grad 없음.
- Teacher: hash 불변.
- 그리드 정렬: 오차 < 1e-4 px, 실제 픽셀 상관 0.978.
- Residual 초기화: step 0에서 Δ = 0, Z = B(assert).
- Padding 불변성: padding cell을 noise로 바꿔도 모든 loss와 R2 출력 변화가 0이다.
- Loss 구현: 단순 구현과 비교해 일치(상대오차 < 4e-6).
- Gradient 전달: step 0에는 `out`에만 전달되고, 이후 body에 전달된다. L_cov / L_rel 각각의 body gradient가 있다.
- 공식 evaluator: 동일.
- R2 예산: 동일(15,594-step schedule, ep6).
- 런타임: decoder 3종 동시 학습 2.86 h(H200 1장, peak 9.9 GB). Probe 26–27 min, C0 probe 190 min.
- 운영 이슈: 큐의 CPU 평가 단계에 경로 버그(`$ROOT/$R` 중복)가 있었다.
  - GPU 결과에는 영향이 없다.
  - 별도 post 스크립트(`scripts/run_post.sh`)로 같은 evaluator를 다시 실행했다.
- QAT:
  - 17:08에 지시대로 즉시 중단했다(step 49,749).
  - 22:27에 checkpoint-45000에서 재개했다(손실 4,749 step). step 45,051 진행을 확인했다.

## 10. 다음 Method로 가져갈 것과 근거
- **주 경로는 G1-cross Encoder Only로 유지한다.** 이번 모든 decoder 변형이 Val/Dev 어느 쪽에서도 이를 넘지 못했다.
- **Crop-grid 표현이 필요하면 B0(parameter-free bilinear)를 기준선으로 쓴다.**
  - Encoder Only와 통계적으로 같다.
  - Obj Material이 Val과 Dev 모두에서 같은 방향(+2.7 / +5.8, n.s.)이다.
  - 학습된 decoder들은 B0보다 나아지지 않았다(D1 − B0 Obj Material −6.32\*).
- **Crop feature matching 계열 decoder 학습(L_detail ± covariance ± relation)은 PACO 판별에 대해 소진된 것으로 본다.**
  - E6 계열(11-06)과 같은 결론이 CLIP-L + residual + bilinear query에서도 반복됐다.
  - feature 지표는 원하는 방향으로 움직였지만 AP는 따라오지 않았다.
- **남은 crop 우위는 Color(특히 Part Color)와 Object Pattern, 그리고 작은 ROI에 있다.** 다음 단계에서 격차를 겨냥한다면 material이 아니라 이쪽이다. 이는 이번 결과에서 나온 관찰이며, 새 실험은 하지 않았다.
- **이번 결론은 seed 0 하나, Val 2,410장(type CI ±4–8)에 기반한다.** Test는 실행하지 않았다.

---

## Table A. PACO 공식 Val — Official-protocol Oracle AP (GT box + category, R2 epoch 6 고정)

### A-1. 공식 Val (2,410 images)

Reflectance\* = PACO 공식 type 이름 `transparency`(opaque / translucent / transparent / other). Existing Decoder 행은 기존 G1-cross 공동학습 decoder + 기존 R2 probe의 **epoch 6** checkpoint(기존 Test 보고 38.11/36.79는 M0로 고른 epoch 9)로 같은 Val 프로토콜에서 새로 평가했다.

| Method | Obj AP | Obj Color | Obj Pattern | Obj Material | Obj Reflectance* | Part AP | Part Color | Part Pattern | Part Material | Part Reflectance* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1-cross Encoder Only | 52.08 | 46.23 | 52.48 | 52.03 | 57.59 | 54.03 | 51.35 | 52.15 | 55.60 | 57.04 |
| Existing G1-cross Decoder (joint, ep6) | 50.54 | 44.33 | 54.71 | 47.80 | 55.32 | 51.67 | 49.85 | 48.71 | 51.93 | 56.20 |
| B0 Bilinear Only | 51.16 | 43.86 | 50.83 | 54.74 | 55.19 | 53.32 | 50.85 | 52.00 | 53.23 | 57.21 |
| D1 Bilinear + Residual | 50.82 | 45.42 | 52.22 | 48.42 | 57.23 | 51.91 | 51.26 | 47.76 | 53.25 | 55.38 |
| L1 D1 + Covariance | 51.66 | 43.54 | 52.10 | 52.60 | 58.38 | 53.04 | 50.36 | 51.86 | 53.58 | 56.35 |
| L2 D1 + Relational | 49.47 | 44.27 | 51.96 | 47.84 | 53.83 | 52.13 | 50.25 | 51.71 | 52.53 | 54.04 |
| (ref) Original CLIP Full | 50.12 | 40.72 | 52.37 | 51.73 | 55.67 | 52.07 | 48.21 | 46.14 | 55.19 | 58.75 |
| (ref) MaskEmbed Full | 48.77 | 45.05 | 48.67 | 48.71 | 52.65 | 53.84 | 50.95 | 51.40 | 54.57 | 58.44 |
| (ref) Original CLIP C0 crop (oracle) | 53.65 | 47.83 | 58.35 | 47.72 | 60.69 | 53.63 | 56.26 | 51.74 | 48.88 | 57.63 |

### A-2. Official-protocol Dev AP (PACO train 내부 dev 1,000 images)

Dev bootstrap CI는 약 ±3 AP로 거칠다. Best candidate 선택에만 사용.

| Method | Obj AP | Obj Color | Obj Pattern | Obj Material | Obj Reflectance* | Part AP | Part Color | Part Pattern | Part Material | Part Reflectance* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1-cross Encoder Only | 58.74 | 53.99 | 63.53 | 48.95 | 68.47 | 59.16 | 55.25 | 65.31 | 58.89 | 57.18 |
| Existing G1-cross Decoder (joint, ep6) | 59.29 | 54.24 | 62.64 | 52.45 | 67.84 | 56.16 | 52.45 | 59.43 | 58.19 | 54.58 |
| B0 Bilinear Only | 60.11 | 54.95 | 64.54 | 54.73 | 66.23 | 59.07 | 55.56 | 65.62 | 58.72 | 56.40 |
| D1 Bilinear + Residual | 59.66 | 54.23 | 63.16 | 54.06 | 67.19 | 58.72 | 53.99 | 62.49 | 61.92 | 56.48 |
| L1 D1 + Covariance | 58.60 | 55.44 | 65.44 | 51.07 | 62.47 | 58.34 | 54.02 | 63.95 | 60.73 | 54.67 |
| L2 D1 + Relational | 57.79 | 54.37 | 63.64 | 50.82 | 62.34 | 56.96 | 54.03 | 61.06 | 58.91 | 53.82 |
| (ref) Original CLIP Full | 57.30 | 49.58 | 63.30 | 51.15 | 65.15 | 58.83 | 52.44 | 67.49 | 57.45 | 57.96 |
| (ref) MaskEmbed Full | 60.12 | 54.43 | 63.49 | 52.51 | 70.03 | 61.23 | 54.94 | 67.98 | 61.43 | 60.57 |
| (ref) Original CLIP C0 crop (oracle) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Table B. 핵심 Delta (paired image bootstrap 95% CI)

| Contrast | Val Obj | Val Part | Val Obj Material | Val Part Pattern | Val Obj Pattern | Dev Obj | Dev Part |
|---|---|---|---|---|---|---|---|
| D1 − B0 Bilinear | -0.33 [-2.30, +1.33] | -1.41 [-2.78, +0.28] | -6.32 [-9.53, -1.39]\* | -4.24 [-8.23, -0.33]\* | +1.39 [-2.75, +4.35] | -0.45 [-3.02, +2.68] | -0.35 [-2.56, +1.82] |
| D1 − Encoder Only | -1.26 [-3.42, +0.54] | -2.12 [-3.67, -0.41]\* | -3.61 [-8.49, +1.15] | -4.39 [-8.76, +0.09] | -0.26 [-4.31, +3.58] | +0.92 [-1.28, +3.44] | -0.44 [-2.47, +1.37] |
| L1 − D1 | +0.83 [-0.87, +2.89] | +1.13 [-0.42, +2.32] | +4.18 [+0.07, +8.66]\* | +4.10 [-0.28, +7.02] | -0.11 [-3.67, +3.81] | -1.06 [-3.60, +1.54] | -0.38 [-2.19, +1.74] |
| L2 − D1 | -1.35 [-2.73, +0.99] | +0.22 [-0.95, +1.51] | -0.58 [-3.76, +3.86] | +3.95 [+0.07, +6.72]\* | -0.26 [-3.20, +3.95] | -1.87 [-4.24, +0.41] | -1.76 [-2.94, +0.64] |
| B0 − Encoder Only | -0.93 [-3.09, +1.05] | -0.71 [-2.05, +0.54] | +2.71 [-4.17, +7.73] | -0.16 [-3.57, +3.23] | -1.65 [-4.53, +1.65] | +1.38 [-1.36, +3.75] | -0.08 [-1.93, +1.59] |
| D1 − Existing Decoder | +0.28 [-1.77, +1.73] | +0.24 [-1.23, +1.30] | +0.62 [-3.57, +3.74] | -0.95 [-3.83, +1.72] | -2.49 [-5.96, +1.23] | +0.37 [-1.87, +3.00] | +2.56 [-0.01, +3.56] |
| Existing Decoder − Encoder Only | -1.54 [-3.55, +0.39] | -2.36 [-3.69, -0.45]\* | -4.24 [-8.70, +0.00] | -3.44 [-7.61, +1.28] | +2.23 [-1.80, +5.28] | +0.55 [-1.81, +2.71] | -3.00 [-4.44, -0.29]\* |
| L1 − Encoder Only | -0.43 [-2.68, +1.60] | -1.00 [-2.46, +0.15] | +0.56 [-5.44, +5.80] | -0.29 [-4.66, +2.14] | -0.38 [-4.15, +3.18] | -0.13 [-2.69, +2.45] | -0.82 [-2.88, +1.38] |
| L2 − Encoder Only | -2.61 [-4.36, -0.35]\* | -1.90 [-3.18, -0.21]\* | -4.19 [-8.44, +0.17] | -0.44 [-5.10, +2.39] | -0.52 [-4.58, +3.77] | -0.94 [-3.21, +1.42] | -2.20 [-4.14, +0.57] |
| Best candidate (D1, by Dev) − MaskEmbed | +2.05 [-0.60, +3.72] | -1.93 [-3.78, +0.26] | -0.29 [-5.03, +3.05] | -3.63 [-9.75, +3.04] | +3.55 [-0.87, +7.28] | -0.46 [-2.94, +2.99] | -2.51 [-4.54, +1.13] |
| Encoder Only − MaskEmbed | +3.31 [+0.97, +5.06]\* | +0.20 [-1.34, +1.95] | +3.32 [-2.21, +8.08] | +0.76 [-3.11, +5.86] | +3.81 [+0.17, +7.48]\* | -1.38 [-3.93, +1.87] | -2.07 [-3.81, +1.42] |

Best candidate = Dev (Obj+Part)/2 최대인 D1 (Dev: D1 59.19, L1 58.47, L2 57.37). Val은 선택에 쓰지 않았다.

## Table C. 학습 Loss 및 Feature 분석 (COCO dev ROIs, 최종 EMA decoder, C0 crop teacher 대비)

COCO dev 468 images / 1872 ROIs (PACO val/test 일치 이미지 제외). 모든 경로가 같은 ROI와 같은 teacher token을 쓴다.

| Path | train L_orig (last 100) | train L_cov | train L_rel | dev L_detail | dev token cos | dev pooled cos | ‖Δ‖/‖B‖ | cos(Δ, T−B) | spatial var ratio (Z/T) | cov err (L_cov) | cov cosine | ROI rel KL (L_rel) | ROI rel Pearson r | context-swap Δ change | context-swap cos-dist change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Bilinear | – | – | – | 0.4089 | 0.6455 | 0.9456 | – | – | 0.093 | 0.9475 | 0.2637 | 0.1076 | 0.289 | – | – |
| Existing decoder | n/a | – | – | 0.3293 | 0.6979 | 0.9728 | – | – | 0.633 | 0.6764 | 0.6111 | 0.0183 | 0.901 | 0.12 | +0.0064 |
| D1 | 0.36 | – | – | 0.3284 | 0.6990 | 0.9726 | 1.37 | 0.08 | 0.779 | 0.6763 | 0.6158 | 0.0195 | 0.894 | 0.17 | +0.0020 |
| L1 | 0.38 | 0.37 | – | 0.3513 | 0.6781 | 0.9706 | 1.29 | 0.11 | 1.464 | 0.3948 | 0.7747 | 0.0219 | 0.881 | 0.15 | +0.0016 |
| L2 | 0.37 | – | 0.01 | 0.3371 | 0.6924 | 0.9705 | 0.98 | 0.09 | 0.472 | 0.7268 | 0.5785 | 0.0176 | 0.912 | 0.14 | +0.0009 |

- train L_orig = 1.11 × L_detail (G1-cross decoder 목적식 그대로). spatial var ratio = ROI 내부 token 분산(student/teacher). cov err = 정규화 ‖C_Z−C_T‖²/‖C_T‖² (L2-normalised token). ROI rel = 같은 batch(16 images) ROI pooled vector cosine 행렬의 KL(τ=0.1) / Pearson r. context-swap = ROI box와 겹치지 않는 key token을 다른 이미지 것으로 바꿨을 때 decoder 출력(Δ 또는 Z)의 상대 변화 / teacher와의 token cosine 거리 증가량(>0 = 실제 문맥이 도움).

λ 고정값(학습 전 16 step, init gradient-norm 비율의 median, 1회): λ_detail 1.11 (G1-cross 그대로), λ_cov **0.655**, λ_rel **1.56** (τ=0.1, IQR cov [0.602, 0.746], rel [1.386, 1.764]; init loss: L_orig 0.437, L_cov 0.939, L_rel 0.0599; teacher ROI-ROI cosine 평균 0.852±0.055, softmax entropy/log(n−1) 0.978).

학습 중 COCO dev 진단(EMA):

| step | model | token cos | pooled cos | ‖Δ‖/‖B‖ | var ratio | cov err | rel KL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | D1 | 0.6455 | 0.9456 | 0.00 | 0.093 | 0.9475 | 0.1076 |
| 0 | L1 | 0.6455 | 0.9456 | 0.00 | 0.093 | 0.9475 | 0.1076 |
| 0 | L2 | 0.6455 | 0.9456 | 0.00 | 0.093 | 0.9475 | 0.1076 |
| 0 | B0_bilinear | 0.6455 | 0.9456 | n/a | 0.093 | 0.9475 | 0.1076 |
| 0 | existing_g1cross_decoder | 0.6979 | 0.9728 | n/a | 0.633 | 0.6764 | 0.0183 |
| 3000 | D1 | 0.6902 | 0.9692 | 0.88 | 0.367 | 0.7523 | 0.0239 |
| 3000 | L1 | 0.6739 | 0.9662 | 0.92 | 0.825 | 0.4488 | 0.0271 |
| 3000 | L2 | 0.6815 | 0.9671 | 0.63 | 0.224 | 0.8357 | 0.0236 |
| 6000 | D1 | 0.6969 | 0.9719 | 1.20 | 0.634 | 0.6925 | 0.0205 |
| 6000 | L1 | 0.6767 | 0.9697 | 1.17 | 1.252 | 0.4053 | 0.0230 |
| 6000 | L2 | 0.6897 | 0.9696 | 0.85 | 0.382 | 0.7542 | 0.0189 |
| 9000 | D1 | 0.6987 | 0.9725 | 1.34 | 0.756 | 0.6788 | 0.0197 |
| 9000 | L1 | 0.6779 | 0.9705 | 1.27 | 1.430 | 0.3961 | 0.0221 |
| 9000 | L2 | 0.6920 | 0.9704 | 0.96 | 0.457 | 0.7310 | 0.0178 |
| 12000 | D1 | 0.6990 | 0.9726 | 1.37 | 0.779 | 0.6763 | 0.0195 |
| 12000 | L1 | 0.6781 | 0.9706 | 1.29 | 1.464 | 0.3948 | 0.0219 |
| 12000 | L2 | 0.6924 | 0.9705 | 0.98 | 0.472 | 0.7268 | 0.0176 |

## Table D. 공통 실패 분석 (PACO Val, pairwise ranking — 공식 AP 대체 아님)

규칙: pair correct iff p(pos) > p(neg) (ties wrong); pairs within (level, category, attribute); known labels only; unit당 최대 50 pair, seed 0. 사용 가능 모델: Original CLIP Full O, G1-cross Encoder Only O, MaskEmbed Full O, Original CLIP C0 crop O

| Group | units | pairs | CLIP Full | G1-cross Enc | MaskEmbed | C0 crop | 두 Full 공통 실패 | Crop만 구별 | 공통 실패 중 Crop 정답 | 반복 unit (n≥5, crop-only≥30%) | ROI side@336 median (전체 / crop-only) | crop-only rate small / large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| obj_material | 224 | 9542 | 74.7 | 74.9 | 75.4 | 75.9 | 16.9% | 7.5% | 44.3% | 9/216 | 46.7 / 41.5 px | 8.9% / 6.1% (side<33px) |
| part_pattern | 488 | 18139 | 73.3 | 76.5 | 75.9 | 77.6 | 16.6% | 7.9% | 47.8% | 26/458 | 32.4 / 25.3 px | 9.2% / 6.7% (side<21px) |
| obj_pattern | 253 | 11053 | 78.6 | 81.2 | 80.5 | 84.6 | 12.5% | 6.7% | 53.8% | 9/245 | 50.2 / 44.7 px | 6.9% / 6.5% (side<34px) |

- **obj_material** crop-only 속성 분포: plastic 142/1794, metal 123/1641, other(material) 110/969, wood 74/922, fabric 63/773, glass 57/1040, paper 43/618, leather 30/435; 상위 category: vase 47/250, basket 36/350, shoe 34/250, handbag 33/350, car_(automobile) 29/200, helmet 29/250
- **part_pattern** crop-only 속성 분포: plain 518/5605, other(pattern_marking) 338/4200, striped 165/2071, text 144/2057, logo 72/929, perforated 66/655, woven 47/781, studded 47/275; 상위 category: handbag:handle 44/100, lamp:base 39/62, lamp:pipe 37/88, knife:handle 35/300, book:cover 34/200, shoe:quarter 34/200
- **obj_pattern** crop-only 속성 분포: other(pattern_marking) 176/2111, plain 145/2428, text 106/1662, striped 100/1318, logo 83/914, perforated 48/498, studded 24/200, checkered 20/426; 상위 category: knife 49/350, lamp 42/250, bench 35/300, shoe 32/200, book 29/250, newspaper 28/104

대표 사례 12개 (왼쪽: 실제 입력 336 letterbox full image + box, 오른쪽: 원본 tight crop; 초록 = positive, 빨강 = negative). 이미지: `output/2026-11-07/decoder_study/failure/viz/`

1. obj_material — **bucket / plastic** — pos/neg 점수: CLIP Full 0.53/0.82, G1-cross Encoder Only 0.42/0.85, MaskEmbed Full 0.26/0.76, CLIP C0 crop 0.98/0.05; ROI side@336 pos 19px / neg 35px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/00_obj_material_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/00_obj_material_neg.jpg)
2. obj_material — **glass_(drink_container) / glass** — pos/neg 점수: CLIP Full 0.15/0.67, G1-cross Encoder Only 0.34/0.78, MaskEmbed Full 0.11/0.91, CLIP C0 crop 0.94/0.16; ROI side@336 pos 26px / neg 42px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/01_obj_material_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/01_obj_material_neg.jpg)
3. obj_material — **box / plastic** — pos/neg 점수: CLIP Full 0.02/0.20, G1-cross Encoder Only 0.20/0.52, MaskEmbed Full 0.26/0.80, CLIP C0 crop 0.94/0.03; ROI side@336 pos 33px / neg 18px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/02_obj_material_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/02_obj_material_neg.jpg)
4. obj_material — **bottle / glass** — pos/neg 점수: CLIP Full 0.04/0.96, G1-cross Encoder Only 0.04/0.90, MaskEmbed Full 0.06/0.99, CLIP C0 crop 0.49/0.16; ROI side@336 pos 19px / neg 73px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/03_obj_material_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/03_obj_material_neg.jpg)
5. part_pattern — **basket:rim / plain** — pos/neg 점수: CLIP Full 0.33/0.84, G1-cross Encoder Only 0.19/0.90, MaskEmbed Full 0.28/0.93, CLIP C0 crop 0.94/0.25; ROI side@336 pos 15px / neg 12px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/04_part_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/04_part_pattern_neg.jpg)
6. part_pattern — **shoe:quarter / striped** — pos/neg 점수: CLIP Full 0.01/0.46, G1-cross Encoder Only 0.00/0.50, MaskEmbed Full 0.01/0.71, CLIP C0 crop 0.91/0.13; ROI side@336 pos 10px / neg 52px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/05_part_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/05_part_pattern_neg.jpg)
7. part_pattern — **mouse_(computer_equipment):logo / plain** — pos/neg 점수: CLIP Full 0.26/0.89, G1-cross Encoder Only 0.46/0.91, MaskEmbed Full 0.59/0.90, CLIP C0 crop 0.96/0.09; ROI side@336 pos 9px / neg 6px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/06_part_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/06_part_pattern_neg.jpg)
8. part_pattern — **scarf:fringes / plain** — pos/neg 점수: CLIP Full 0.79/0.59, G1-cross Encoder Only 0.39/0.62, MaskEmbed Full 0.42/0.63, CLIP C0 crop 0.94/0.03; ROI side@336 pos 24px / neg 40px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/07_part_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/07_part_pattern_neg.jpg)
9. obj_pattern — **napkin / plain** — pos/neg 점수: CLIP Full 0.83/0.93, G1-cross Encoder Only 0.60/0.98, MaskEmbed Full 0.56/0.96, CLIP C0 crop 0.98/0.08; ROI side@336 pos 33px / neg 46px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/08_obj_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/08_obj_pattern_neg.jpg)
10. obj_pattern — **remote_control / text** — pos/neg 점수: CLIP Full 0.08/0.85, G1-cross Encoder Only 0.22/0.88, MaskEmbed Full 0.15/0.70, CLIP C0 crop 0.71/0.02; ROI side@336 pos 48px / neg 45px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/09_obj_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/09_obj_pattern_neg.jpg)
11. obj_pattern — **newspaper / plain** — pos/neg 점수: CLIP Full 0.61/0.97, G1-cross Encoder Only 0.48/0.83, MaskEmbed Full 0.22/0.88, CLIP C0 crop 0.90/0.05; ROI side@336 pos 35px / neg 35px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/10_obj_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/10_obj_pattern_neg.jpg)
12. obj_pattern — **towel / text** — pos/neg 점수: CLIP Full 0.02/0.07, G1-cross Encoder Only 0.02/0.18, MaskEmbed Full 0.11/0.35, CLIP C0 crop 0.97/0.01; ROI side@336 pos 68px / neg 184px
   ![pos](../output/2026-11-07/decoder_study/failure/viz/11_obj_pattern_pos.jpg) ![neg](../output/2026-11-07/decoder_study/failure/viz/11_obj_pattern_neg.jpg)

## Sanity checks

| 항목 | 결과 | 근거 |
|---|---|---|
| 동일 Encoder Checkpoint (sha256, 6개 경로) | OK | output/2026-11-07/decoder_study/preds/*/infer_val.json |
| Encoder weight hash 학습 전/후 + freeze | before 5b23f44c660c… after 5b23f44c660c… unchanged=True, grads none=True | dec/*/done.json |
| Teacher 동일 (원본 CLIP-L/336, hash 불변) | 64a83d7d70f8… unchanged=True | dec/config.json |
| 세 decoder 동일 초기 상태 | init sha256 b17e47d718fc… (assert 동일) | dec/config.json |
| Crop letterbox ↔ Bilinear grid 정렬 | 96 ROI 최대 오차 3.7e-05px; 실제 픽셀 상관 median 0.978 (min 0.860); 합성 좌표맵 오차 6.1e-05px | sanity/checks_cpu.json |
| Decoder Query/Output shape | full [24, 576, 1024] → B, Δ, Z [96, 576, 1024] (R×576×1024), teacher [96, 576, 1024] | sanity |
| Residual zero-init | Δ max|·| = 0.0 (fp32), 0.0 (bf16); Z==B True; 학습 step 0에서도 assert | sanity + dec/*/grad_check.jsonl |
| Padding token 제외 | padding cell(32%)을 noise로 바꿔도 L_detail/L_cov/L_rel 변화 0.0; R2는 cov=0 cell noise에 logit 변화 0 (아래) | sanity |
| Loss 구현 검증 | L_cov vs torch.cov 상대오차 3.8e-06; L_rel vs naive 7.4e-07; self=0 | sanity |
| Loss별 gradient 전달 | D1: step1 body 0.0e+00 / out 9.1e-02; step11 body orig 4.77e-04; L1: step1 body 0.0e+00 / out 9.1e-02; step11 body orig 7.75e-04, cov 1.68e-03; L2: step1 body 0.0e+00 / out 9.1e-02; step11 body orig 4.03e-04, rel 4.56e-04 | dec/*/grad_check.jsonl |
| R2는 cov>0 cell만 읽음 | g1cross_enc 0.0e+00, g1cross_dec 0.0e+00, b0 0.0e+00, d1 0.0e+00, l1 0.0e+00, l2 0.0e+00 | infer_val.json checks |
| Decoder 출력/Δ 실제 사용 | g1cross_dec: zero→|Δlogit| 1.81, b0: zero→|Δlogit| 2.54, d1: zero→|Δlogit| 2.54, Δ 제거→ 0.896, l1: zero→|Δlogit| 1.80, Δ 제거→ 0.906, l2: zero→|Δlogit| 1.68, Δ 제거→ 0.734 | infer_val.json |
| Full image 인코딩 횟수 / ROI RGB crop 재인코딩 | g1cross_enc 1회/0, g1cross_dec 1회/0, b0 1회/0, d1 1회/0, l1 1회/0, l2 1회/0  (C0 oracle만 ROI당 1회) | infer_val.json |
| Decoder가 Full image 문맥 참조 | D1: 0.169, L1: 0.147, L2: 0.143, existing_g1cross_decoder: 0.119 (out-of-box key 교체 시 출력 상대 변화) | featdiag |
| R2 학습 예산 동일 | b0: schedule 15594 steps, ep [1560], rows 200088, 사용 epoch 6; d1: schedule 15594 steps, ep [1560], rows 200088, 사용 epoch 6; l1: schedule 15594 steps, ep [1560], rows 200088, 사용 epoch 6; l2: schedule 15594 steps, ep [1560], rows 200088, 사용 epoch 6; c0: schedule 15594 steps, ep [1560], rows 200088, 사용 epoch 6 (기존 g1cross probes: 동일 yaml, ep6 ckpt) | probes/*/config.json |
| Probe 중 frozen 경로 불변 | b0 True, d1 True, l1 True, l2 True, c0 True | frozen_check.json |
| 공식 Evaluator 동일 | src.pacoattr.official_eval (PACO @cdefd31, lvis-api @7d7f07d, 수정 없음) — 모든 val 행; dev: src.paco_devap.official_dev |  |
| 학습 이미지 / ROI | COCO stream 12,000 steps × 16 images × 4 ROIs (PACO val/test 일치 2장 제외, 고유 이미지 50489), world 1, 64 ROI/step; PACO attribute label 미사용 | dec/config.json |
| Decoder 학습 GPU/Step/Runtime/Peak VRAM | NVIDIA H200 1장(물리 GPU 2), 12,000 step, 세 decoder 동시 2.86 h, peak 9.9 GB | dec/*/done.json |
| Probe runtime | b0 26 min, d1 27 min, l1 27 min, l2 27 min, c0 190 min | probes/*/done.json |
| 기존 작업 충돌 | GPU 0,1 미사용. QAT(GPU 2,3) 사용자 지시로 즉시 중단: step 49749 → checkpoint-45000에서 재개 (손실 4749 step). 재개 상태: output/2026-11-07/decoder_study/qat_resume_status.md | qat_pause/ |

추론 시 GT ROI는 Oracle 진단에만 쓴다. 이 보고서의 어떤 경로도 GT box를 필수 입력으로 요구하는 최종 방법이라고 주장하지 않는다.

## Queue status

```
item	status	updated	note
qat_pause	DONE	2026-11-07 17:08:37	stop 49749 ckpt 45000 lost 4749
checks_cpu	DONE	2026-11-07 17:09:09	
infer_existing_val	DONE	2026-11-07 17:09:59	
failure_prelim	DONE	2026-11-07 17:10:01	
smoke_train	DONE	2026-11-07 17:10:56	25 s/20 steps (3 decoders), peak 9.9 GB, enc unchanged True
smoke_probe	DONE	2026-11-07 17:18:23	
calib	DONE	2026-11-07 17:19:04	
probe_b0	DONE	2026-11-07 17:45:10	ep 6 devM0 49.31
infer_b0	DONE	2026-11-07 17:45:49	
train_dec	DONE	2026-11-07 20:37:52	10288 s peak 9.9 GB enc_unchanged True
probe_c0	DONE	2026-11-07 20:57:02	ep 6 devM0 49.73
devinfer_existing	DONE	2026-11-07 17:46:43	
probe_d1	DONE	2026-11-07 21:24:33	ep 6 devM0 48.83
probe_l1	DONE	2026-11-07 21:52:10	ep 6 devM0 49.42
probe_l2	DONE	2026-11-07 22:19:43	ep 6 devM0 49.29
infer_new	DONE	2026-11-07 22:25:32	
featdiag	DONE	2026-11-07 22:27:02	
devinfer_new	DONE	2026-11-07 22:26:57	
qat_resume	DONE	2026-11-07 22:30:04	
eval_val	DONE	2026-11-07 22:33:23	b0 c0 clip_full d1 g1cross_dec g1cross_enc l1 l2 maskembed_full 
eval_dev	DONE	2026-11-07 22:34:42	clip_full,maskembed_full,g1cross_enc,g1cross_dec,b0,d1,l1,l2
boot	DONE	2026-11-07 22:38:41	val: clip_full,maskembed_full,g1cross_enc,g1cross_dec,b0,d1,l1,l2,c0 | dev: clip_full,maskembed_full,g1cross_enc,g1cross_dec,b0,d1,l1,l2
failure_final	DONE	2026-11-07 22:38:42	
report	RUNNING	2026-11-07 22:38:43
```
