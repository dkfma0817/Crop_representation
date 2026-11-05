# 공유 Feature Readout 큐: 사전 등록 프로토콜 (2026-11-04)

목적: 단순한 구조 변경이 아니라, 각 실험이 노린 **특정 능력이 실제로 학습되는지**를 확인한다. 가설 검증(직접 검사)과 Overall 개선(downstream)은 따로 기록한다.

## 공통

| 항목 | 설정 |
|---|---|
| Encoder | Frozen DINOv2-B/14 (`vit_base_patch14_dinov2.lvd142m`), 336 letterbox, 24×24 token |
| Baseline | **Multi-layer + R2**: block 4/8/12 출력 (12 = final LN), 층별 LN → concat → Linear(3·768→768), 그 위에 R2 (2 cross block, cov>0 mask). 기존 11-04 checkpoint와 코드가 이 서버에 없으므로 **이번 큐의 공통 Baseline으로 새로 정의**한다. 이전 실험을 정확히 재현했다고 주장하지 않는다. |
| 학습 데이터 | PACO train에서 중복 6장(exact 1 + near-dup 후보 5) 제외: 38,472장, 205,499 ROI |
| 학습 | 10 epoch, 이미지 pack ≈128 ROI/step (`full_epoch_batches`, data seed 0), hflip 0.5, masked BCE / #known labels, AdamW (reader 3e-4, head 1e-3, wd 0.05), warmup 1 epoch + cosine, clip 1.0, reader init seed `manual_seed(7)` (seed 0) |
| 실행 | GPU 2+3 DDP (각 step을 이미지 단위로 반씩 나누고, gradient 합산 = 단일 GPU와 같은 업데이트) |
| 평가 | **PACO Validation만.** Test는 큐에서 불러오지 않는다 (`load_data`가 test를 막음). |
| Primary | Val **M2 Overall**, 마지막 epoch(10) 기준 (val에서 epoch 선택 안 함). best-val epoch는 참고로만 기록 |
| Secondary | Object, Part, Color, Material, Pattern, Transparency (M2). Val Object 5–16은 공식 M2에서 NaN이므로 보고하지 않고, 0으로 바꾸거나 임의 계산하지 않는다 |
| 통계 | Val image 단위 paired bootstrap, 1,000회, seed 0 (모든 조건이 같은 draw 사용) |

### Downstream 판정 (모든 실험 공통)
- **Overall 개선**: ΔM2 Overall (method − Baseline)의 95% CI 하한 > 0
- **특정 속성 집중**: 양수인 type별 개선량의 합 중 한 type이 70% 이상
- **주요 악화**: Secondary 지표 중 Δ ≤ −1.0 이면서 CI 상한 < 0
- **Downstream 성공** = Overall 개선 AND 집중 아님 AND 주요 악화 없음. **Overall 개선이 없고 Color만 개선되면 성공이 아니다.**
- **실행 유효**: done.json 존재, 10 epoch 전부 기록, train/val loss 모두 유한. Train-subset(2,000장 고정) loss/M2로 과적합/미학습 확인

## 대조군 학습 (기존 결과 재사용 불가 → 새로 학습)
- `ctrl_f336`: `ml_r2` (Baseline), `r2_final` (final layer만 쓰는 R2, 기존 "F336 R2" 대응)
- `ctrl_c0`: `c0_r2` (원본 GT crop → 336 → final layer R2, crop-grid geometry). D0 전용이며 teacher로 쓰지 않는다
- `ctrl_f672`: `f672_ml_r2` (같은 336 canvas를 GPU에서 ×2 bicubic → 672 → DINOv2, Baseline과 같은 reader). E12 teacher

## D0: 학습 없는 ROI Binding 진단
- 마지막 epoch Val 예측 사용: `ml_r2`, `r2_final`, `c0_r2`, `f672_ml_r2`
- Pair: 같은 이미지, 같은 level, i≠j, Y[i,a]=1 & Y[j,a]=0 (explicit). Matched cross-image pair: j와 level·category·크기 층이 같은 다른 이미지의 explicit negative 최대 5개
- PA_w (같은 이미지 pair 정확도), PA_x (cross-image 대응 정확도 = 점수·분류 수준), **BD = PA_x − PA_w** (같은 이미지에 있어서 생기는 추가 어려움 = ROI 귀속 문제), dP = σ(s_i) − σ(s_j)
- 층: all / obj_small (SigLIP2 n_in 1–16) / obj_large / part / same·diff category / overlap·disjoint
- 같은 pair에서 F336과 C0 비교: 점수 차이(PA_x diff)와 binding 차이(BD diff)를 따로 보고
- **결정 규칙**: all pair에서 BD(ml_r2) ≥ 1.0pp (CI 하한 > 0) 이고 BD(ml_r2) − BD(c0_r2)의 CI 하한 > 0 → **E05**, 아니면 **E08**. D0가 실패하면 E08

## 실험

### E05 — ROI Binding Supervision (D0에서 문제가 확인될 때만 실행)
- 능력: 같은 이미지 안의 이웃 ROI와 속성을 구분해 올바른 ROI에 귀속
- 학습: BCE + λ0.5 · softplus(s_j,a − s_i,a) (같은 이미지, 같은 level, explicit pos/neg만)
- 대조군: Baseline `ml_r2` (같은 batch, 즉 같은 ROI pair를 보지만 BCE만)
- 직접 검사: paired accuracy PA_w, BD, ROI 교체 시 예측 변화 dP (D0와 같은 pair)
- 능력 확인: PA_w 차이 CI 하한 > 0 AND BD 차이 CI 상한 < 0

### E08 — Weight-sharing Repeated Readout (D0에서 binding 문제가 뚜렷하지 않으면 실행)
- 능력: 같은 공유 feature를 다시 읽어 첫 read에서 놓친 정보를 찾기
- 학습: cross block 1개. `x1` 1회 read, `x3` 같은 block을 3회 반복 read, `ffn3` 1회 read 후 같은 block의 FFN만 2회 추가 (같은 파라미터·추가 계산, 재read 없음). **Loss는 최종 출력에만.** 단계별 예측은 따로 기록
- 직접 검사: x3 − x1, x3 − ffn3 (M2), 단계별 M2
- 능력 확인: x3 − x1 CI 하한 > 0 AND x3 − ffn3 CI 하한 > 0

### E09 — Query-conditioned Adaptive Sampling
- 능력: 같은 36개 feature sample로 어디를 볼지 고르기
- 학습: Baseline + block마다 fused F336 map에서 36개 bilinear sample (box 좌표 [−0.25, 1.25]). `ads`는 query로 위치 예측 (grid에서 시작), `fix`는 grid 고정. 같은 feature·reader·데이터·loss, 새 픽셀·crop encoding 없음. 파라미터 차이: offset predictor 0.11M (reader의 약 1%)
- 직접 검사: ads − fix M2, sample 위치 이동량 / box 밖 비율 / 중심 거리 / instance mask 위 비율 (val 1,500 ROI)
- 능력 확인: ads − fix CI 하한 > 0 (위치 이동은 함께 보고)

### E11 — Local/Context Gated Readout
- 능력: local 근거와 context를 따로 읽고 상황에 맞게 섞기
- 주 대조군 `concat`: 하나의 stream, 2 block으로 [ROI token ; 전체 576 token] + token-type embedding. 실험 `gate`: local stream (2 block, ROI token) + context stream (2 block, 전체 token) + 학습 gate. Ablation `ctx`: context만. 입력 token과 출력 차원은 같음 (gate 쪽에 gate MLP와 block 2개만큼 파라미터가 더 있음, 보고서에 기록)
- 직접 검사: Part M2, context 제거(ROI token만) / 교체(ROI 밖 token을 다른 이미지 것으로) 시 M2 변화와 |Δp|, 크기별 gate 값
- 능력 확인: gate − concat Part CI 하한 > 0 AND context 제거 시 gate Part M2 하락 CI 하한 > 0

### E12 — Scale Decision Consistency
- 능력: 원본 추가 픽셀 없이, 인코딩 스케일(336 vs 672)에 따른 판별 격차를 학습으로 줄이기
- Teacher: 같은 336 canvas를 672로 확대해 DINOv2로 인코딩한 `f672_ml_r2` (train 예측). C0 crop teacher는 쓰지 않음
- Student: F336 Baseline 구조, BCE + 1.0 · BCE(student, σ(teacher)) (같은 실제 ROI, 59 속성). 추론은 F336만
- 대조군: Baseline `ml_r2` (같은 BCE, consistency 없음)
- 직접 검사: teacher의 실제 판별 성능 (val M2, Baseline 대비), student의 정답 성능 (val M2), teacher와의 불일치(BCE) 변화, gap closure = Δstudent / Δteacher
- 능력 확인: teacher와의 불일치 감소 CI 상한 < 0 AND gap closure > 0

## 기존 실험과의 중복
- 이 서버에는 F336/C0/F672 DINOv2 reader의 checkpoint나 val 예측이 없다 (`third_party/research`에는 코드만, `output/2026-11-04/probe_smoke_B`는 RAEv2 latent). 그래서 재사용할 수 있는 것이 없어 대조군을 새로 학습한다.
- 기존과 겹치는 R3(Relational KD), R4(Encoder FT 반복)는 제외. A/B/S/C(feature 복원)·D(Detail CNN)와 달리 이번 실험은 모두 F336 공유 feature 위의 readout 학습이며, E12의 teacher는 feature가 아닌 **결정(prediction)** 수준이다.
