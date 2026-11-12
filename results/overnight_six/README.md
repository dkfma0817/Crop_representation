# Overnight Six Pilots (DINOv3-B/16, A0 Coarse-Fine, Full RGB 384, seed 0)

| Experiment | Best Epoch | Val Overall | Test Overall | Test Obj Pattern | Test Part Pattern | Baseline 대비 변화 (Test Overall, 95% CI) |
|---|---|---|---|---|---|---|
| E2 Gradient Routing | 8 | 55.42 | 39.65 | 47.19 | 40.72 | G1U 대비 **−2.35 [−3.38, −1.52]\*** |
| E2-L Lower Grid Weight | 8 | 57.21 | 42.42 | 50.38 | 44.17 | G1U 대비 +0.41 [−0.51, +0.93] |
| E1a Native Crop768 Pooled | 7 | 57.11 | 41.57 | 47.84 | 42.91 | A0 대비 −0.04 [−0.75, +0.57] |
| E1b Native Crop768 Spatial | 9 | 56.63 | 41.54 | 47.57 | 43.87 | E1a 대비 −0.03 [−0.94, +0.76], A0 대비 −0.07 [−1.01, +0.63] |
| F1 Fine-focused Crop | 6 | 56.77 | 42.01 | 47.25 | 43.38 | A0 대비 +0.40 [−0.61, +0.88] |
| S1 Small-ROI Balanced | 8 | 56.64 | 41.75 | 45.95 | 42.87 | A0 대비 +0.14 [−0.66, +0.65] |
| *A0 Crop384 (기준)* | 7 | 56.53 | 41.61 | 47.61 | 43.30 | |
| *G1U (기준)* | 7 | 56.18 | 42.01 | 49.70 | 44.44 | |

평가는 PACO 공식 evaluator의 GT Box + GT Category Oracle Attribute AP입니다. Overall = (Object + Part) / 2입니다. Epoch은 공식 Val에서 고정했고 Test는 그 epoch으로 1회만 평가했습니다. CI는 image-level paired bootstrap(B=1,000, seed 0)입니다. \*는 CI가 0을 포함하지 않는 경우입니다. **Seed 0 단일 실험이고 다중비교는 보정하지 않았습니다.** 표의 Reflectance는 공식 evaluator의 `transparency` 항목입니다.

## 결론

**6개 중 기준선을 유의하게 개선한 실험은 없습니다.** E2만 G1U보다 유의하게 나빠졌고, 나머지 5개는 Overall CI가 모두 0을 포함합니다.

Crop Oracle과의 격차도 남아 있습니다. 가장 높은 E2-L의 Test Overall이 42.42이고, 이전 실행의 C0 Crop Oracle은 44.91입니다(별도 실행의 참고값).

다음 Method에 쓸 후보를 하나만 고른다면 **낮은 가중치의 Grid Distillation(E2-L)**입니다. Val과 Test Overall, Test Object Pattern에서 모두 가장 높습니다. 다만 유의하지 않고 seed가 1개뿐이어서, 후보 수준으로만 봐야 합니다.

## 항목별 해석

### 1. Gradient Routing이 G1U를 개선했는가 → 아니요, 악화됨

- Test Overall −2.35 [−3.38, −1.52]\*, Part Pattern −3.72 [−5.79, −1.38]\*, Object Material −2.99 [−4.73, −0.86]\*입니다. Object Pattern −2.51 [−5.43, +1.00]은 유의하지 않습니다. Val Overall은 −0.75 [−2.84, +0.40]입니다.
- 구현은 의도대로 동작했습니다. 15회 점검 모두 Grid Gradient가 차단 파라미터(Coarse Backbone, Fine→Coarse)에 적용되지 않았고(변화 norm 0), Fine 쪽에는 전달됐습니다.
- 원인 후보는 충돌이 Fine으로 옮겨 간 것입니다. Fine Transformer의 crop/grid gradient cosine이 step 1의 +0.03에서 step 12,000의 −0.84로 벌어졌습니다(같은 시점 E2-L은 −0.34). 학습 후반 Global Retention 손실도 0.060으로 E2-L의 0.0098보다 약 6배 큽니다. 인과를 검증하지는 않았습니다.
- 차단 대상이던 Coarse의 Grid Gradient는 Crop Gradient의 1.5~2.1배로 컸습니다. cos(crop, grid)는 −0.43 ~ −0.73입니다.

### 2. Grid Weight 감소만으로 개선되는가 → 유의한 개선은 아님

- E2-L은 G1U 대비 Test Overall +0.41 [−0.51, +0.93], Val Overall +1.03 [−0.76, +1.74]입니다.
- E2가 E2-L보다 유의하게 낮으므로(Test Overall −2.76 [−3.52, −1.68]\*), Routing의 결과는 단순한 Grid 약화로 설명되지 않습니다.
- 사전 지정 외 탐색 비교로 E2-L − A0를 계산했습니다. Test Overall +0.81 [−0.24, +1.37], Object Pattern +2.77 [−0.22, +5.06]입니다. 둘 다 0을 포함합니다.

### 3. Crop768 Teacher가 Crop384보다 나은가 → 아니요

- E1a − A0는 Test Overall −0.04 [−0.75, +0.57], Object Pattern +0.23, Part Pattern −0.40, Object Material +1.80 [−0.49, +2.42]로 모두 유의하지 않습니다. Val Overall은 +0.59 [−0.62, +1.52]입니다.
- 768 pooled Teacher와 384 Teacher의 cosine은 평균 0.944로 서로 구분되는 Target이었습니다. Target을 바꾸었는데도 AP가 변하지 않았다는 뜻입니다.
- Full-image 768 Teacher(G1-Native) 때와 같은 방향입니다. Teacher 해상도만으로는 Crop 격차가 설명되지 않습니다.

### 4. Spatial KD가 Pooled KD보다 나은가 → 아니요

- E1b − E1a는 Test Overall −0.03 [−0.94, +0.76], Part Pattern +0.96 [−0.83, +2.64], Object Pattern −0.27로 모두 유의하지 않습니다.
- 학습 손실만 보면 Spatial Loss는 줄었습니다(step 1의 0.296 → 마지막 구간 평균 0.197). 하지만 PACO AP가 오르지 않았으므로 **표현 학습 성공으로 보지 않습니다.**
- λ_spatial = 1.5(초기 가중치에서 ‖g_crop768‖/‖g_spatial‖ 중앙값, 2 유효숫자)로 학습 전에 고정했습니다. Spatial Loss는 ROI의 78.9%에 적용됐고, ROI당 유효 셀 수는 중앙값 60(p25 27, p75 120, p99 144)입니다.

### 5. Fine-focused Learning이 Pattern에 도움이 되는가 → 아니요

- F1 − A0는 Test Object Pattern −0.37 [−3.74, +1.48], Part Pattern +0.08 [−2.01, +1.72]입니다. Overall +0.40 [−0.61, +0.88]과 Object Material +1.41 [−0.65, +2.38]도 유의하지 않습니다.
- Fine 파라미터에 Crop Gradient 추가 적용과 Coarse·Fine→Coarse 미적용은 15회 점검에서 확인했습니다.

### 6. Small-ROI Weighting이 작은 영역에 도움이 되는가 → 근거 부족

- 대상 구간(4~16 token)의 Test 점추정, S1 − A0: Object Pattern +0.15, Object Material +2.17, Object Color +0.10, Part Pattern −0.64, Part Material +1.11, Part Color −0.27입니다. Pattern 개선은 없고, 이 표에는 CI가 없습니다.
- 전체 Test에서 Object Material +2.06 [+0.33, +3.48]\*로 CI가 0을 제외했지만, Val에서는 +0.40 [−2.61, +2.14]로 재현되지 않았습니다. 비교 88개에 대한 다중비교를 보정하지 않았으므로 우연일 가능성이 있습니다.
- Test Object Pattern은 −1.67 [−4.20, +0.81]로 낮은 쪽입니다. 가중치 대상 ROI는 전체의 33.2%입니다.

## 다음 Method에 도움이 될 가능성

PACO Pattern과 Overall만 기준으로 정리했습니다.

1. **낮은 가중치의 Grid Distillation을 후보로 유지합니다.** E2-L은 Val과 Test Overall, Test Object Pattern이 가장 높습니다. 다음 단계는 seed를 늘려 G1U·A0와 비교하는 것입니다. 지금 단계에서 효과가 있다고 주장할 수는 없습니다.
2. **Gradient Routing은 이 형태로는 중단합니다.** 유의하게 나빠진 유일한 실험입니다. 충돌을 Coarse에서 막으면 Fine으로 옮겨 가는 패턴이 관찰됐습니다.
3. **Teacher 해상도 강화(768 Crop, Spatial 정렬)는 우선순위를 낮춥니다.** E1a, E1b, 이전 G1-Native 모두 AP 변화가 없었습니다.
4. **Fine 쪽 Crop Gradient 추가와 Small-ROI 가중은 Pattern 개선 근거가 없습니다.** 약하게 Material 방향 신호만 있고, 재현되지 않았거나 유의하지 않습니다.
5. 공통적으로 Crop Oracle과의 격차(약 2.5 이상)를 줄이지 못했습니다. 이번 6개 원리 중 Full-image Encode-once Feature의 Pattern 판별력을 뚜렷하게 올린 것은 없습니다.

## 실행 상태

| Experiment | Encoder 학습 | R2 평가 | Val Best Epoch | Test |
|---|---|---|---|---|
| E2 | 완료 (step 1,087에서 재개) | 완료 | 8 | 완료 |
| E2-L | 완료 | 완료 | 8 | 완료 |
| E1a | 완료 | 완료 | 7 | 완료 |
| E1b | 완료 | 완료 | 9 | 완료 |
| F1 | 완료 | 완료 | 6 | 완료 |
| S1 | 완료 | 완료 | 8 | 완료 |

## 사양 대비 달라진 점과 알려둘 점

- **R2는 GPU에서 실행했습니다.** CPU 인코더 forward가 32스레드에서 약 4 img/s(측정값)여서, 10 epoch(약 37만 장)가 실험당 수 시간 이상 걸립니다. GPU는 실행 가능한 인코더 학습이 없을 때만 썼습니다. Official 평가와 Bootstrap은 CPU에서 실행했습니다.
- **E2 재개:** 큐 재시작 때문에 step 1,087에서 `recovery.pt`로 이어서 학습했습니다. 모델, EMA, optimizer가 복원되고 데이터 순서는 step으로 고정되어 있습니다. E2를 중단 없이 다시 돌린 결과와 비교하지는 않았습니다.
- **Teacher 캐시 재생성:** 메인 스레드 병목을 고치느라 두 번 재시작했고, 저장 내용은 전부 새로 썼습니다. 검증을 통과했습니다.
- **컨테이너 재시작:** 11-10 08:52(KST)에 Test ROI 크기별 평가 중 발생해 해당 단계만 다시 실행했습니다.
- **ROI 크기별 표에는 CI가 없습니다.** 점추정만 있습니다.
- E2-L − A0와 G1U − A0는 사전 지정 외 탐색 비교입니다.
- E2-L과 G1U는 초기화와 데이터가 같아 독립 반복이 아닙니다.

## 파일

| 내용 | 위치 |
|---|---|
| 전체 표 (Val/Test, Delta CI, ROI 크기별, 학습 요약) | [eval/results/tables.md](eval/results/tables.md) |
| Official AP JSON / CSV | [official_ap.json](eval/results/official_ap.json), [val_ap.csv](eval/results/val_ap.csv), [test_ap.csv](eval/results/test_ap.csv) |
| Baseline 대비 Delta | [delta_ap.csv](eval/results/delta_ap.csv) |
| ROI 크기별 평가 | [test_roi_size.csv](eval/results/test_roi_size.csv), [val_roi_size.csv](eval/results/val_roi_size.csv) |
| R2 Epoch별 Val, 선택된 Epoch | [eval/selection/](eval/selection/) |
| 실험 Config | [configs/](configs/) |
| 학습 로그, Checkpoint, R2 로그 | `<실험>/logs`, `<실험>/checkpoints`, `<실험>/probe`, [logs/](logs/) |
| Gradient 검증 | [verification/gradient_summary.json](verification/gradient_summary.json), `<실험>/logs/diagnostics.jsonl`, `<실험>/smoke/smoke.json` |
| Teacher 전처리 검증 | [verification/teacher768_preprocessing.json](verification/teacher768_preprocessing.json) |
| λ_spatial 산정 | [lambda_spatial.json](lambda_spatial.json) |
| 코드 스냅샷 | [source_snapshot/](source_snapshot/) |
| Crop768 Teacher 캐시 (68 GB) | `output/2026-11-10/overnight_six_teacher768` |
