# P-C768 / M1: Crop Feature 감독 방식 검증 (DINOv3-B/16, A0 Coarse-Fine, Full RGB 384, seed 0)

| Method | Teacher | Crop Loss | Pixel Flow | Val Overall | Test Overall | Test Obj Pattern | Test Part Pattern | Baseline 대비 변화 (Test Overall, 95% CI) |
|---|---|---|---|---|---|---|---|---|
| **P-C768** | Native Crop768 | 1 − cos | 있음 (λ 1.0) | 56.37 | 41.41 | 47.80 | 42.75 | PixelDiT Hybrid 대비 **−0.66 [−1.38, −0.01]\*** |
| **M1** | Native Crop384 (C0) | Raw MSE (λ 3.0) | 없음 | 56.74 | 41.25 | 46.27 | 42.33 | A0 대비 −0.36 [−1.19, +0.36] |
| *PixelDiT Hybrid (기준)* | Native Crop384 | 1 − cos | 있음 (λ 1.0) | 56.71 | 42.07 | 48.37 | 43.49 | |
| *A0 Crop Cosine (기준)* | Native Crop384 | 1 − cos | 없음 | 56.53 | 41.61 | 47.61 | 43.30 | |
| *E1a (참고)* | Native Crop768 | 1 − cos | 없음 | 57.11 | 41.57 | 47.84 | 42.91 | |

평가는 PACO 공식 evaluator의 GT Box + GT Category Oracle Attribute AP입니다. Overall = (Object + Part) / 2입니다. Epoch은 공식 Val에서 고정했고(P-C768 8, M1 8, 기준선은 기존 고정값 7), Test는 그 epoch으로 1회만 평가했습니다. CI는 image-level paired bootstrap(B=1,000)이며, \*는 CI가 0을 포함하지 않는 경우입니다. **Seed 0 단일 실험이고 다중비교는 보정하지 않았습니다.**

## 결론

**두 방향 모두 Pattern을 개선하지 못했습니다.**
- **P-C768:** PixelDiT Hybrid보다 Test Overall이 −0.66으로 근소하게 유의하게 낮고, Part Color도 −1.54 [−2.11, −0.57]\*로 낮습니다.
- **M1:** A0와의 Test 차이가 모든 항목에서 유의하지 않습니다.

## 질문별 답

### 1. PixelDiT에서 Crop768 Teacher가 Crop384보다 도움이 됐는가 → 아니요

- **Test, P-C768 − Hybrid:**
  - Overall −0.66 [−1.38, −0.01]\*
  - Object Pattern −0.58 [−3.29, +2.10], Part Pattern −0.73 [−2.46, +0.89]
  - Object Color −0.74 [−1.32, +0.21], **Part Color −1.54 [−2.11, −0.57]\***
  - Val Overall은 −0.34 [−1.75, +0.65]입니다.
- **Flow 유무 비교 (탐색, 같은 Crop768 Teacher):** P-C768 − E1a는 Test Overall −0.17 [−0.82, +0.49]로 유의하지 않습니다. Crop768 Teacher에 Flow를 더해도 이득이 없어, 둘이 상보적으로 작용한다는 증거는 보이지 않습니다.

### 2. MSE가 Cosine보다 ROI Pattern 판별력을 개선했는가 → 아니요

- Test M1 − A0: Object Pattern −1.34 [−4.76, +1.56], Part Pattern −0.98 [−3.05, +0.81]로 모두 유의하지 않습니다.
- Val에서는 Part Pattern +1.82, Object Material +3.87 [+0.61, +6.85]\*였지만 Test에서 재현되지 않았습니다. Test는 Part Pattern −0.98, Object Material +0.70 [−1.60, +2.33]입니다.

### 3. 개선이 있다면 Object/Part 중 어디인가 → 유의한 개선 없음

- **P-C768:** Object와 Part 모두 Test 점추정이 낮습니다(−0.61 / −0.71). 유의한 악화는 Part Color에서만 나왔습니다.
- **M1:** Object −0.20, Part −0.52로 모두 유의하지 않습니다.
- **ROI 크기별(Test 점추정, CI 없음):**
  - P-C768은 Tiny Object Pattern에서 Hybrid보다 높습니다(55.17 vs 53.12). 반면 Medium/Large Part Pattern은 낮습니다(61.06 vs 63.28, 69.57 vs 72.61).
  - M1은 Medium Object Pattern이 63.10으로 A0(66.52)보다 낮습니다.

### 4. Teacher Feature Norm이나 Gradient 변화가 AP 변화와 일치하는가

- **P-C768 Teacher:** Crop768 Teacher의 Norm은 Crop384보다 평균 4% 작고, 두 Teacher의 cosine은 0.92~0.94입니다. Teacher는 실제로 달라졌지만 AP는 오르지 않았습니다.
- **P-C768 Student:** Student Feature Norm이 학습 중 8.65에서 11.0~11.3으로 커져 Teacher(8.5~8.6)와 멀어졌습니다. Cosine Loss는 Norm을 감독하지 않으므로 예상되는 현상입니다. 이것이 AP 악화와 관련 있는지는 검증하지 않았습니다.
- **P-C768 Gradient:** Coarse의 Flow Gradient가 step 1의 0.66에서 3,000 step 이후 0.01~0.02로 줄어 Crop Gradient(0.25~0.36)보다 한 자릿수 작습니다. Crop–Flow Gradient cosine도 0 근처(−0.03 ~ +0.13)입니다. 학습 대부분 구간에서 Flow가 Encoder에 주는 신호는 약합니다.
- **M1 Gradient:** λ_mse=3.0으로 맞춘 MSE Gradient는 Cosine Gradient와 방향이 거의 같습니다(cosine Coarse 0.92~0.99, Fine Transformer 0.93~0.97). 학습 중 크기는 Cosine의 약 60~70%입니다. M1은 사실상 A0와 거의 같은 방향의 약간 약한 감독이었고, AP 차이가 유의하지 않은 것과 일치합니다.
- **M1 Norm:** MSE는 Norm도 감독하지만 Student–Teacher Norm 차이는 학습 중 0.45에서 약 1.0으로 오히려 커졌습니다(Student 7.9, Teacher 8.9). Raw MSE가 Norm을 Teacher에 맞추지는 못했습니다.
- 요약하면 Teacher Cosine, Loss, Norm의 변화가 Pattern AP 개선으로 이어진 경우는 없습니다. **Loss나 Teacher Cosine의 변화만으로 성공을 판단하지 않았습니다.**

### 5. 다음 Method 개발에 더 유망한 방향 → 둘 다 권하지 않음

- P-C768은 Overall과 Part Color가 유의하게 낮아졌으므로 진행하지 않는 것이 좋습니다.
- M1은 해가 없지만 이득도 없습니다. Gradient 방향이 Cosine과 거의 같아서, λ나 정규화만 바꿔서는 다른 결과를 기대하기 어렵습니다.
- 더 눈에 띄는 것은 탐색 비교인 **PixelDiT Hybrid − A0의 Color 이득**입니다(Test Obj Color +0.88 [+0.10, +1.39]\*, Part Color +1.30 [+0.59, +1.86]\*). Pixel Flow의 효과는 Pattern보다 Color 쪽에 있는 것으로 보입니다. 현재 진행 중인 J1(PixelDiT + Grid)이 이 Color 이득을 유지하는지 확인할 만합니다.

## 실행 상태와 알려둘 점

| Experiment | Encoder 학습 | R2 | Val Best Epoch | Test |
|---|---|---|---|---|
| P-C768 | 완료 (12k, 1,479초) | 완료 | 8 | 완료 |
| M1 | 완료 (12k, 1,110초) | 완료 | 8 | 완료 |

- **Crop768 캐시 재사용:** PixelDiT 학습 데이터와 ROI 목록 파일이 같고, object/nested ROI 354개를 다시 인코딩해 cosine 0.9999 이상으로 확인해서 재생성하지 않았습니다. 같은 ROI를 384로 인코딩하면 기존 Hybrid Teacher도 재현됩니다.
- **Step 1 재현:** P-C768의 Flow·Global·Crop384 손실이 기존 Hybrid 로그와 같고 Crop768 손실은 E1a와 같습니다. 즉 Teacher만 바뀌었습니다. M1의 Cosine·Global 손실은 A0 로그와 같습니다.
- **λ_mse = 3.0:** 초기 가중치에서 8개 배치의 ‖g_cos‖/‖g_mse‖ 중앙값으로 정했습니다. Coarse만 보면 2.95, Fine만 보면 4.17입니다. 12k 동안 고정했습니다.
- **M1 검증 다섯 가지:**
  - Feature가 Unit-Norm이 아닙니다(약 8~9).
  - MSE와 Cosine의 Gradient가 서로 다릅니다(cosine 0.93~0.97).
  - Feature Norm 분포가 정상입니다.
  - NaN/Inf가 없습니다.
  - Encoder 파라미터가 실제로 갱신됩니다.
- **R2 실행 위치:** CPU에서는 약 4 img/s로 너무 느려 GPU에서 돌렸습니다. Official 평가와 Bootstrap은 CPU에서 실행했습니다.
- **비교 구분:** P-C768 − Hybrid와 M1 − A0는 사전 지정 비교이고, P-C768 − E1a와 Hybrid − A0는 탐색 비교입니다.
- **R2 경로 차이:** 기준선 A0와 Hybrid의 R2는 11-08 스냅샷의 `src.ivc.train_bb`로, P-C768과 M1의 R2는 `src.g1up.train_bb`로 학습했습니다. 추론과 evaluator 경로는 같으며, 기존 Reader를 현재 경로로 재채점해 AP가 정확히 재현됨을 확인했습니다(results/joint_pixeldit_g1/verification/baseline_audit_*.json). R2 학습 코드 차이의 영향은 J1 실험에서 R2를 같은 경로로 다시 학습한 기준선으로 확인합니다.

## 파일

| 내용 | 위치 |
|---|---|
| 전체 표 (Val/Test, Delta CI, ROI 크기별, 학습 요약) | [eval/results/tables.md](eval/results/tables.md) |
| Official AP JSON / CSV, Delta | [official_ap.json](eval/results/official_ap.json), [val_ap.csv](eval/results/val_ap.csv), [test_ap.csv](eval/results/test_ap.csv), [delta_ap.csv](eval/results/delta_ap.csv) |
| R2 Epoch별 Val, 선택된 Epoch | [eval/selection/](eval/selection/) |
| Config | [configs/](configs/), [lambda_mse.json](lambda_mse.json) |
| 학습 로그, Checkpoint, 진단 | `PC768/`, `M1/` (logs/train.jsonl, logs/diagnostics.jsonl, checkpoints/, smoke/smoke.json) |
| Gradient / Feature Norm 진단 요약 | [verification/diagnostics_summary.json](verification/diagnostics_summary.json) |
| Teacher 캐시 정합성 | [verification/teacher768_vs_pixeldit_data.json](verification/teacher768_vs_pixeldit_data.json) |
