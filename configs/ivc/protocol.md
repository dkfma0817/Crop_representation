# PACO Bilinear / Bicubic / C0 Crop: 성능·Feature 차이 분석 — 실행 전 프로토콜 (2026-11-05)

새 decoder 설계·학습 없음. Encoder와 feature는 모두 frozen이고, 조건별로 R0/R2 reader만 학습한다. **PACO Test는 불러오지 않는다**(`load_data`가 test를 막음). 모든 산출물은 `output/2026-11-05/ivc/`에 저장하고, 코드는 새 파일 `src/ivc/*`, 설정은 `configs/ivc/*`만 쓴다. 기존 파일·checkpoint·결과는 수정하지 않는다.

## 0. 기존 코드·자산 확인 결과

| 항목 | 이 서버 상태 | 이번 실험에서 |
|---|---|---|
| paco_token_upsample | `third_party/research/2026-10-30/paco_token_upsample/code`만 있음. perception_cache(feature, ckpt, 결과)는 없음 | R0 정의와 F672 정의만 참고 |
| paco_c0_multilayer, multilevel_crop_native, crop_res_context | 코드·cache·결과 모두 없음(서버 전체 검색) | 재사용 불가. 수치도 비교하지 않음 |
| R0 / R2 구현 | `paco_region_reader/code/common_rr.py`(R0, AttnReader), `src/readout/models.py`(R2 = `Reader('r2', layers=[12])`) | R2는 그대로 쓰고, R0는 같은 정의로 crop grid용으로 다시 구현(`src/ivc/common.py: R0`) |
| PACO 평가 | `paco_crop_finetune/code/eval_tables.py` + `eval_mlr.build_units` + `aggregate` (`src/readout/evaluate.Boot`) | 그대로 |
| Bilinear crop-grid sampling | `dinov2_oracle_region_decoder/models/roi_reader.bilinear` + `data/preprocessing.roi_geometry()['bil_grid']` (11-05 MSD `probe_bil`과 같은 호출) | 그대로. Bicubic은 같은 호출에 `mode='bicubic'`만 바꿈 |
| F336 / C0 token feature cache | 없음 (`/dev/shm/readout_cache`는 uint8 canvas cache이고 dev split 이전 train 순서라서 쓰지 않음) | 새로 추출. 학습은 on-the-fly encoding, 분석은 streaming |
| 11-05 MSD `probe_bil`, `probe_c0` (R2) | 같은 프로토콜의 val/dev score가 있음 | **재현 확인용으로만** 쓴다. 본 결과는 이번에 새로 학습한 reader로 낸다(세 조건 모두 같은 코드 경로) |
| 디스크 | `/home` 3.2 TB 남음(97% 사용), `/dev/shm` 1 TB | token 전체 저장 안 함(val만 해도 조건당 약 9.8 GB fp16). ROI별 scalar, 576-cell map(fp16), pooled vector만 저장(합계 < 0.5 GB) |

## 1. Feature 추출 조건 (세 조건 공통)

- Encoder: timm `vit_base_patch14_dinov2.lvd142m`, `img_size=336, dynamic_img_size=True`, 모든 파라미터 frozen. **최종층 = final LayerNorm 적용 patch token**(= `forward_features`, CLS 제외) 576×768. bf16 autocast로 forward하고 fp32로 꺼낸다. 이후 sampling과 모든 지표 계산은 fp32.
- 입력: letterbox 336. `s = 336 / max(w, h)`, `nw = round(w s)`, `nh = round(h s)`, 가운데 정렬 `(ox, oy) = ((336 − nw)//2, (336 − nh)//2)`. PIL BICUBIC resize, ImageNet mean/std 정규화. Padding은 정규화 공간에서 0(= ImageNet 평균색)이고 **encoder는 이 padding pixel을 입력으로 본다**.
- **A. F336 + Bilinear ROI**: 전체 이미지 → 24×24×768. C0 crop grid의 각 cell 중심 `((j + 0.5)·14, (i + 0.5)·14)`을 crop letterbox → 원본 px → full canvas로 옮긴 좌표(`roi_geometry` `bil_grid`)에서 `F.grid_sample(mode='bilinear', padding_mode='border', align_corners=False)` → 24×24×768.
- **B. F336 + Bicubic ROI**: A와 같은 full-image token, 같은 grid와 설정에 `mode='bicubic'`(PyTorch cubic convolution, A = −0.75, 4×4 support). F336 forward 1회를 A/B가 공유한다.
- **C. C0 GT crop**: 정수 GT box(실수 box를 바깥으로 올림, clamp, 최소 1 px; manifest `l, t, r, b`)로 원본 이미지를 crop → 같은 336 letterbox → 같은 encoder → 24×24×768.
- ROI / 이미지 / label / split: 세 조건이 같은 manifest(`C.load_split`)와 같은 행 순서를 쓴다.
- 선택 참고 조건(C1, F672): **생략**. C1/F672 reader는 이 서버에 없어 새로 학습해야 하고, F672는 grid가 달라 token 단위 비교 대상이 아니다.

## 2. ROI 좌표 정렬 (실행 결과 포함: `output/2026-11-05/ivc/align/`)

- GT box: PACO `bbox`(x, y, w, h, 원본 px)를 바깥으로 올림한 정수 box(manifest `l, t, r, b`).
- F336 좌표: `X = ox_f + x · nw_f / W` (축별 scale에 rounding까지 반영). Full cell k의 중심은 canvas `(k + 0.5)·14` px이고, `align_corners=False`에서 grid 좌표 `X/168 − 1`이 이 중심과 맞는다.
- C0 cell (i, j)의 원본 좌표: `x = l + ((j + 0.5)·14 − ox_c) · (r − l) / nw_c`.
- 유효 token mask: C0 letterbox coverage(cell 면적 중 content 비율). **주 mask M = coverage > 0**(가중치 = coverage, R0 pooling·R2 attention mask와 같음). **구조 지표 mask M1 = coverage = 1인 직사각형**(가장자리의 부분 padding cell 제외).
- Padding cell(coverage 0): C0에서는 mean-color padding pixel을 encoder가 본 token이고, A/B에서는 같은 좌표가 box 밖(context, canvas 밖이면 border clamp)을 sample한다. 비교 대상이 아니므로 주 지표에서 빼고, "전체 576 cell"과 "padding cell만" 지표로 따로 보고한다.
- 검증 (CPU, 실행 완료):
  1. **해석적 검증(val 전 ROI 11,109개, 유효 cell 3,890,448개)**: 독립 수식으로 계산한 좌표와 sampler가 쓰는 `bil_grid`의 최대 오차 9.8e-6 canvas px(float32 저장 오차). 유효 cell 중 2.06%는 sample 위치가 full-grid 중심 hull [7, 329] px 밖이라 border clamp가 걸리고, 0.31%는 이미지 content 밖이다(C0 가장자리의 부분 cell).
  2. **합성 좌표 map**: 채널 = 각 full cell 중심의 canvas X/Y. Bilinear는 hull 안에서 최대 오차 7.7e-5 px다. **Bicubic은 hull 안에서도 평균 0.55 px, 최대 0.67 px의 오차가 있는데**, 이는 grid 문제가 아니라 PyTorch bicubic kernel(A = −0.75)이 선형 함수를 정확히 재현하지 않기 때문이다. 해석적으로 계산하면 cell 간격의 1/4 지점에서 0.047 cell = 0.66 px이고 측정값과 일치한다. 가장자리 1 cell 이내에서는 border clamp 때문에 최대 1.35 px다. Sampling 좌표는 A와 같으므로 수정하지 않고 B 조건의 정의(kernel 특성)로 기록한다.
  3. **Feature shift scan(GPU)**: 고정 600 ROI(D5–16 300개, D17+ 300개)에서 bicubic grid를 (dx, dy) ∈ {−14, −7, 0, 7, 14} px만큼 옮겨 C0와의 token cosine을 잰다. 정렬이 맞으면 (0, 0)에서 최대여야 한다.
  4. 그림: 고정 ROI 12개(종횡비 wide/square/tall × D-bin에서 1개씩 + 최소 n_in, 가장 넓은 것, 가장 긴 것; `configs/ivc/sample_rois.json`, seed 0, feature 계산 전에 고정). 원본 + GT box + C0 cell 중심(유효/padding), C0 입력 crop + 24×24 grid, F336 canvas + sample 위치, coverage mask.

## 3. Reader 학습 (조건별, seed 0, 11-05 MSD probe와 같은 프로토콜)

| 항목 | 설정 |
|---|---|
| 데이터 | PACO train에서 중복 6장(`data_check`) 제외 → PACO dev 1,000장을 이미지 단위로 고정해 뺌(rng 0) → **학습 37,472장 / 200,088 ROI**. Dev는 epoch 선택과 threshold 보정에만 쓴다 |
| Batch | 이미지 pack ≈128 ROI/step(`C.full_epoch_batches`, data seed 0), epoch당 약 1,560 step, GPU 2+3 DDP(step을 이미지 단위로 반씩 나누고 gradient 합산 = 단일 GPU와 같은 update) |
| Augmentation | 이미지 hflip p = 0.5 (box도 flip, C0는 flip된 이미지에서 crop) |
| Loss | masked BCE / step의 known label 수(59 attribute, level별 head) |
| Optimizer | AdamW β (0.9, 0.999), wd 0.05(1-D 파라미터·query 제외). R2 lr 3e-4, head lr 1e-3. Warmup 1 epoch + cosine, grad clip 1.0, **10 epoch**. Reader 연산은 bf16 autocast |
| R2 | learned query + box embedding → 2 cross-attention block(8 head, MLP ×2), kv = LN(token + geo_mlp(geo)), attention mask = coverage > 0. Geometry = C0 crop-grid 규칙(`crop_canvas_geo`: rel = cell 중심의 content-rect 기준 좌표, cov = crop coverage, box = content rect). **세 조건이 완전히 같은 geometry를 쓰고 token만 다르다** |
| R0 | coverage 가중 ROI 평균 → level별 z-score → Linear head(59 × 2). 학습 파라미터는 head뿐(lr 1e-3). z-score 통계: **각 조건 자기 train feature**(학습 train 중 4,000장, rng 0, flip 없음, level별 평균과 비편향 std + 1e-6, `common_rr.mean_stats`와 같은 규칙) |
| 초기화 | reader마다 `torch.manual_seed(7)`. R0와 R2는 같은 process에서 같은 encoder forward를 공유하지만 optimizer는 각각이다. R0 통계 계산은 data 순서·flip·R2 init의 RNG를 건드리지 않는다 |
| Checkpoint 선택 | epoch 1–10 중 **PACO dev M0 Overall 최대**(MSD 규칙). 모든 epoch의 reader state를 저장한다(`probe_<c>/<reader>/ep<k>.pt`) |

기존 코드와의 차이: (1) R0가 원래는 full-image grid의 box coverage cell을 평균했지만, 여기서는 사용자 지정대로 crop-grid 24×24 출력(C0 coverage 가중)을 평균한다. (2) Bicubic sampling은 새로 추가했다. (3) `src/ivc/train.py`는 `src/readout/train.py`를 복사해 위 두 가지와 epoch별 checkpoint 저장만 추가했다.

## 4. 평가 (PACO **Validation**: 2,051장, 11,109 ROI; object 3,140 / part 7,969)

- **M0** (primary) = 기존 `paco_lp` pooled 정의: (level, attribute) unit마다 val ROI 전체에서 sklearn AP를 계산한다. Unit 조건은 val positive ≥ 10, negative ≥ 10, train positive ≥ 10. Group 값은 해당 unit들의 단순 평균. Overall은 모든 unit 평균, Object/Part는 level별, Color/Material/Pattern은 type별 평균이다.
- **M2** (보조, 기존 프로젝트 정의이고 **PACO 공식 detection AP가 아님**) = category-macro AP: (level, category, attribute) unit(positive ≥ 1, negative ≥ 40, 배경 attribute 38/54 제외)마다 COCO 101-point AP → category 평균 → attribute → type → level → overall. M0와 섞지 않는다. Bin에 eligible unit이 없으면 NaN이고 대체하지 않는다.
- 크기 bin = manifest `n_in`(F336 24×24 grid에서 box와 겹치는 cell 수, `eval_fu.py`의 D-bin): **D1–4 / D5–16 / D17+**. 열: Overall, Object, Part, Obj D5–16, Part D1–4, Obj D17+(`eval_fu`의 "17+ obj" 관례), 그리고 참고로 D17+ 전체, Color, Material, Pattern. Bin 값은 해당 ROI만으로 unit을 다시 만든 M0다. 11-05 MSD 보고서의 SigLIP2 n_in bin과는 다르다.
- **Paired bootstrap**: val 이미지 단위 cluster bootstrap, B = 1,000, seed 0. 모든 조건과 reader가 같은 draw를 쓴다. 대비는 Bicubic − Bilinear, C0 − Bicubic, 참고로 C0 − Bilinear. 이는 평가 데이터 표본의 불확실성이고 학습 seed 분산이 아니다(seed는 0 하나).
- 같은 ROI/label 확인: 모든 score 파일이 (11,109, 59)이고 manifest 행 순서이며, label 배열 하나를 공유하고, probe config의 train/val 행·이미지 수와 step 수가 같은지 확인한다.
- 누수: train/val 이미지 id 교집합 없음(assert). Train에서 val/test와 exact 또는 near-duplicate인 6장을 제외했다. PACO dev는 train의 부분집합이다. DINOv2 사전학습 데이터(LVD-142M)와의 중복은 통제할 수 없다.

## 5. Feature 지표 (raw final-layer feature, fp32, `src/ivc/common.py: roi_metrics`)

a = Bilinear 또는 Bicubic의 crop-grid token, c = C0 token, w = coverage(M 위에서 합 1로 정규화).

| 지표 | 정의 | Mask |
|---|---|---|
| Token cosine | Σ w_i cos(a_i, c_i). 비가중 평균과 ROI 내 p10/p50도 기록 | M |
| Token MSE | Σ w_i ‖a_i − c_i‖² / 768 | M |
| Pooled | ā = Σ w_i a_i (= R0 입력). Pooled cosine cos(ā, c̄), Pooled L2 ‖ā − c̄‖ | M |
| 분해 | MSE = E_mean + E_cent (정확히 성립). E_mean = ‖ā − c̄‖²/768, E_cent = Σ w_i ‖(a_i − ā) − (c_i − c̄)‖²/768. 공간 패턴 차이의 비율 E_cent/MSE, centred token cosine Σ w_i cos(a_i − ā, c_i − c̄) | M |
| Spatial variance | Σ w_i ‖a_i − ā‖²/768 (a, c 각각), 비율 a/c | M |
| Self-similarity | S = token 간 cosine 행렬. 차이 = 비대각 원소의 평균 \|S_a − S_c\|(RMS도 기록), 비대각 평균 유사도 | M1 |
| Effective rank | 중심화한 token 행렬 고유값의 exp(entropy)(Roy & Vetterli). a/c 비율 | M1 |
| 인접 token cosine | 가로·세로 이웃 쌍 평균 | M1 |
| 공간 주파수 | M1 직사각형(h × w)에서 채널별 orthonormal 2-D DCT-II, DC 제외. ρ = max(u/h, v/w)로 low < 1/3 ≤ mid < 2/3 ≤ high. Band별 energy(Parseval 정규화), non-DC 비율, a/c 비율, 차이(a − c)의 band 분포. **M1만 쓰므로 letterbox padding 경계가 DCT에 들어가지 않는다.** 높은 energy가 더 좋은 feature라는 뜻은 아니다 | M1 (\|M1\| ≥ 4) |
| 전체 영역(보조) | 576 cell 전체의 token cosine/MSE, padding cell만의 cosine | 전체 / padding |

정규화한 feature 분석은 하지 않는다(raw만). Spatial variance, rank, frequency는 관찰 지표로만 쓴다(이전 연구에서 이 지표들의 개선이 PACO 개선으로 이어지지 않은 사례가 있다).

## 6. ROI 크기 / Mask 분석
- D1–4 / D5–16 / D17+ × object/part, 그리고 n_in 연속값과의 Spearman과 산점도.
- **PACO mask**: val annotation(`paco_lvis_v1_val.json`, `id` = manifest `ann_id`, polygon)을 pycocotools로 decode → 정수 box로 crop → C0 letterbox content 크기로 PIL BOX(면적 평균) resize → canvas에 C0와 같은 위치로 붙임 → cell별 mask 비율 = cell 안 mask 면적 / cell 안 content 면적. **영역 규칙(결과 보기 전 고정)**: coverage > 0 cell 중 interior = 비율 ≥ 0.9, boundary = 0.1–0.9, background(box 안, mask 밖) = 비율 ≤ 0.1. 11,109 ROI 모두 mask가 있다(실행 완료, `features/mask_frac.npz`).

## 7. 예측과 연결한 분석 (Bicubic vs C0, 같은 reader 종류끼리; R2가 주, R0는 보조)
- **정오답 정의**: (조건, reader, level, attribute)마다 **PACO dev**(선택한 epoch의 dev score)에서 F1을 최대로 하는 logit threshold τ를 쓴다. Val은 보지 않는다. M0 unit에 속하고 dev positive ≥ 5, negative ≥ 5인 (level, attribute)만 쓰고, 나머지 pair와 unknown label(−1)은 제외한다. 정답 = (s ≥ τ) == (y = 1).
- G1 Bic 오답/C0 정답, G2 둘 다 정답, G3 둘 다 오답, G4 Bic 정답/C0 오답. **Positive label과 negative label을 따로** 집계한다(negative가 압도적으로 많아서).
- 그룹별 feature: 각 pair에 그 ROI의 Bicubic-vs-C0 지표를 붙여 pair 가중 평균을 낸다. G1 − G2는 그대로의 차이와 **level × D-bin 층화 차이**(층 가중치 = 층의 G1 pair 수) 두 가지로, ROI-level SD 단위로 표준화하고 이미지 bootstrap 95% CI를 붙인다. G1이 작은 ROI에 몰리면 크기 효과와 feature 효과가 섞이므로 층화 값을 주로 본다. 차이가 있어도 인과를 뜻하지는 않는다.
- **점수 변화**: reader 간 logit scale이 같다는 보장이 없으므로 주 지표는 scale에 무관한 **rank margin**이다. Positive label은 같은 unit의 val negative 중 이 점수보다 낮은 비율(AP와 직접 관련), negative label은 같은 unit의 positive 중 이 점수보다 높은 비율이다. 보조 지표는 dev 보정 logit margin (2y − 1)(s − τ). 조건 간 logit SD 비율도 보고한다.
- **상관**: 한 ROI에 attribute가 여러 개 붙어 있으므로 **ROI당 하나의 값**(그 ROI positive label의 C0 − Bic rank margin 평균)을 쓴다. ROI 단위 Bicubic-vs-C0 지표와의 Spearman과, rank n_in과 level을 회귀로 제거한 **partial Spearman**을 계산하고, 이미지 bootstrap 95% CI(300회, seed 0)를 붙인다. 하위 집합은 object/part, D-bin, attribute type(type별 positive만). 지표 11개를 함께 보므로 다중 비교에 주의한다.
- 보조: Bicubic − Bilinear의 ROI별 token cosine 변화와 ROI별 rank margin 변화의 partial 상관.

## 8. 시각화 (측정값만 그림)
- 고정 ROI 12개(object/part × D-bin에서 2개씩, seed 0, `configs/ivc/sample_rois.json`, feature 계산 전에 고정): 원본+GT, C0 입력, Bilinear/Bicubic/C0 token PCA-RGB(ROI마다 세 조건을 함께 fit해 같은 투영), token cosine heatmap(척도 [−0.2, 1] 고정), Bilinear−C0와 Bicubic−C0 token MSE map(ROI 안에서 두 map이 같은 척도), Bicubic/C0 self-similarity와 그 차이(척도 [−1, 1]). Padding cell은 회색이고 수치에서 제외한다.
- G1–G4 예시: R2, positive label pair 중 그룹마다 3개를 `default_rng(0)`으로 (row, attr) 정렬 순서에서 균등 추출한다. Feature 거리나 점수로 고르지 않는다.
- ROI size vs feature difference 산점도, feature difference vs 점수 변화 산점도, token cosine 분포, 분해, DCT band.

## 9. 실행
- GPU 2,3만 쓴다. 사용자 QAT 작업(`/home/irteam/low-bit-training-v2`, 800m ste)은 checkpoint-15000 저장 직후 중지하고(`output/2026-11-05/ivc/qat_pause/`, 명령·환경 저장), 큐가 끝나면 같은 명령(이미 `--resume` 포함)으로 재개한다.
- 큐: `scripts/ivc/queue.sh output/2026-11-05/ivc` (tmux `ivc_q`). 기록은 `queue_status.tsv`, `logs/`.
