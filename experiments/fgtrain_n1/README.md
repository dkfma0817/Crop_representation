# N1: B0 + NeCo-style ROI Neighbor Ordering

사용자가 고정한 ROI-level 변형이며 원래 NeCo의 공식 재현이 아니다. 학습 설정은 `training_protocol.json`, `N1/config.json`에 보존한다. 추가 Fine/Pixel Decoder, Adapter는 없다.

B0는 Original pretrained 초기화, COCO GT+random/fallback,12k, RGB384 DINOv3-B/16, 표준 encoder-only export를 사용했다. 기존 B0의 Official Val 및10epoch R2 설정/초기화/제외 split/forward 검증을 재확인하여 재사용한다.

Test Gate: N1 Official Val Overall > 55.450408318307 (B0 표시값55.450). bootstrap은 추가 진입 조건이 아니다. Test에는 Val 선택을 그대로 고정하며 한 번만 평가한다. ADE20K는 자동 실행하지 않는다.

Reader selection은 같은 Official Val2410 이미지에서 10epoch 중 Object/Part 평균 최대, 동점이면 앞 epoch다. 최종 EMA encoder는12k 고정이다. Test로 loss·method·epoch·B2 여부를 선택하지 않는다.

| Split / Model | Overall | Obj AP | Obj col | Obj pat | Obj mat | Obj ref | Part AP | Part col | Part pat | Part mat | Part ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Val Frozen | 54.213 | 52.266 | 42.736 | 51.944 | 52.403 | 61.983 | 56.159 | 50.396 | 52.787 | 57.031 | 64.422 |
| Val B0 | 55.450 | 54.511 | 45.183 | 59.644 | 51.225 | 61.993 | 56.390 | 52.671 | 55.243 | 56.587 | 61.057 |
| Val B1 | 55.779 | 54.327 | 46.525 | 58.186 | 50.875 | 61.722 | 57.231 | 53.480 | 56.376 | 56.483 | 62.584 |
| Val N1 | 55.961 | 54.443 | 45.846 | 56.232 | 52.546 | 63.151 | 57.478 | 51.459 | 55.567 | 58.653 | 64.234 |
| Val C0 | 56.395 | 55.699 | 48.148 | 61.978 | 51.797 | 60.873 | 57.091 | 55.068 | 61.072 | 54.029 | 58.196 |
| Val N1−B0 | 0.511 | -0.068 | 0.663 | -3.413 | 1.320 | 1.158 | 1.089 | -1.212 | 0.324 | 2.066 | 3.177 |
| Test B0 | 41.005 | 41.556 | 31.756 | 49.176 | 34.354 | 50.937 | 40.454 | 35.029 | 44.296 | 36.163 | 46.326 |
| Test N1 | 40.097 | 40.449 | 31.189 | 44.522 | 33.124 | 52.959 | 39.746 | 34.220 | 42.039 | 35.793 | 46.932 |
| Test N1−B0 | -0.908 | -1.107 | -0.568 | -4.654 | -1.230 | 2.022 | -0.708 | -0.810 | -2.257 | -0.370 | 0.606 |

`ref`는 기존 공식 evaluator의 `transparency` 출력이다. Val/Test 수치를 섞지 않는다.
C0는 native ROI RGB를 재인코딩하는 Crop Oracle이며 표준 encoder-only 방법과 정보 조건이 다르다. Frozen은 학습0step, B0/B1/N1은 동일 Original 초기화에서12k다.

검증: `baseline_audit.json`, `evaluation_protocol.json`. 결과: `test_gate.json`, `paired_bootstrap.json`, `test/fixed_protocol.json`, `results.json`.

## 고정 Loss와 실제 검증

Student full RGB384 → 기존 coverage-weighted pooled ROI feature를 사용한다. Teacher는 원본 RGB ROI crop→letterbox768→Frozen Original DINOv3의 B0 pooled cache이며 detach한다. 여러 이미지의 ROI를 함께 L2 normalize하여 cosine similarity를 구한다. PACO 속성 label은 encoder 학습에 사용하지 않는다.
Self 및 같은 이미지 IoU>0.5를 먼저 제외하고 Teacher 순서의 상위4개를 positive, 1-based16~32위(양 끝 포함)를 negative로 고른다. Teacher cosine 차이>0.02인 조합만 비교한다. 같은 이미지라도 IoU≤0.5면 후보로 남긴다.
`L_rank = mean softplus((S_neg − S_pos)/0.07)`; `L_N1 = L_crop + 0.5 L_global + lambda_rank L_rank`. Crop/Global 경로는 B0와 동일한 함수를 그대로 사용한다. lambda는 초기 동일 stream0 미니배치에서 encoder 전체 gradient norm 비율0.25로 한 번 보정한 뒤 고정한다. Teacher/Student similarity 행렬과 이웃 선택은 학습에만 사용하며 추론 경로에는 없다.

실측 lambda_rank=0.074547304493; crop encoder gradient=0.85315037, rank=2.86110401; weighted ratio=0.25000000.
초기 60 ROI: valid pairs 4078/4080 (99.951%), anchor coverage 100.0%. Teacher gap 유효 분위수=[0.022932350635528564, 0.07408158481121063, 0.09577101469039917, 0.1321530044078827, 0.18368755280971527, 0.2453308254480362, 0.40022337436676025]. 첫 batch에서 B0 Crop/Global bit equality, Teacher detach, initial weight 불변성, patch embedding 및12block 각각의 nonzero gradient를 확인했다.
`ranking_unit_verification.json`은 독립 NumPy reference와 ranks16~32, self/overlap 제외, IoU=0.5 포함, 빈 pair의 differentiable zero 및 Teacher gradient 차단을 검증한다. `lambda_rank.json`은 실제 GPU preflight다. 지원이0 또는 극소이면 threshold를 완화하지 않고 중단한다.

## 학습 추이

| Step | Total | Crop | Global | Rank | Valid pair ratio | Anchor coverage |
|---|---:|---:|---:|---:|---:|---:|
| 50 | 0.325635 | 0.294730 | 0.000215 | 0.413118 | 0.999962 | 1.000000 |
| 1000 | 0.170772 | 0.131335 | 0.035715 | 0.289479 | 1.000000 | 1.000000 |
| 3000 | 0.151778 | 0.117284 | 0.028790 | 0.269618 | 0.999982 | 1.000000 |
| 6000 | 0.140528 | 0.108538 | 0.024036 | 0.267909 | 0.999948 | 1.000000 |
| 9000 | 0.133974 | 0.104329 | 0.023074 | 0.242906 | 0.999991 | 1.000000 |
| 12000 | 0.133185 | 0.104657 | 0.021501 | 0.238468 | 0.999991 | 1.000000 |

전체 raw trace는 `N1/train_log.jsonl`, `N1/neighbor_diagnostics.jsonl`, `N1/per_loss_gradients.jsonl`, `N1/block_gradients.jsonl`에 보존한다.

## 표준 Encoder Export

Strict loading=True; parameter count=85641216; patch shape=[2, 576, 768]; forward bit equality=True; additional modules=[]. Final12k EMA를 export하고 같은표준 encoder를 R2에 사용한다. 원본 RGB는 이미지당 한 번만 인코딩한다. 실제 PACO canvas 계수는 `N1/preds/inference_verification.json`에 저장한다.

## Val paired-image bootstrap (1000회)

선택된 Reader epoch에 조건부인95% CI이며 seed 변동을 대신하지 않는다. Test Gate에는 사용하지 않는다.

| N1−B0 | Difference (pp) | 95% CI |
|---|---:|---:|
| Object | -0.068 | [-2.029, +2.185] |
| Part | +1.089 | [-0.522, +2.289] |
| Object Pattern | -3.413 | [-6.488, +1.614] |
| Part Pattern | +0.324 | [-3.509, +3.571] |
| Overall | +0.511 | [-0.976, +1.904] |
