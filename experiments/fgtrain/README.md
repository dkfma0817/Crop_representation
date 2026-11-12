# DINOv3 Training-only Fine-grained Supervision Pilot (Seed 0)

표준 DINOv3-B/16 자체의 지역 속성 표현을 평가한다. Fine/Pixel Decoder는 학습 시에만 존재한다. 모든 B 조건은 같은 Original pretrained `vit_base_patch16_dinov3.lvd1689m`에서 독립적으로 시작한다. G1-pool, G1-cross 또는 A0에서 초기화하지 않는다.

## 구조와 손실

Student 입력은 Full RGB384 letterbox, Patch16, 12개 원래 Transformer Block, 최종 24×24×768 patch feature다. 표준 timm DINOv3의 `forward_features`를 그대로 호출한다. Encoder Forward는 ROI와 독립적이다. Fine→Coarse feedback, RGB 입력 Fine branch, 추론용 Adapter/Projection/Calibration은 없다.

Native Teacher는 원본 RGB에서 먼저 ROI를 crop하고 기존 Crop Oracle의 aspect-preserving Bicubic letterbox로 RGB768을 만든 Original frozen DINOv3다. 출력은 48×48×768이다. 768 resize가 원본에 없는 새 픽셀 정보를 생성하는 것은 아니며, Encoder를 더 세밀한 공간 단위에서 계산한다. 셀과 letterbox content의 교차 면적 비율 c_q로 mask/weight한다. 완전 padding 셀은 제거한다. 부분 content 셀은 기존 coverage 규칙으로 가중한다. 압축 캐시는 **유효 원래 token을 그대로** float16 저장하며 48×48 좌표를 보존한다. 기존 E1b의 최대 12×12 재집계 target은 사용하지 않는다. Pooled 및 Full384 global target은 동일 체크포인트와 ROI의 기존 캐시를 실제 재인코딩으로 검증 후 재사용한다.

Fine Decoder는 G1의 Fourier ROI-relative/absolute 좌표 처리 및 XBlock을 재사용한다. Coarse token 768→256 학습용 입력층, 48×48 위치별 query, 2 cross-attention/FFN blocks (8 heads, MLP ratio4), 출력 256→768이다. 모든 Decoder 입력은 Coarse Feature, ROI geometry, Full-image valid-key mask다. RGB를 받지 않는다. Encoder Feature는 detach하지 않고 Teacher는 detach한다.

정의 d(a,b)=1−cos(a,b), weighted_pool(F,w)=ΣwF/Σw:

- L_crop = mean_ROI d(weighted_pool(Student(full384), ROI∩patch area), pooled Native Crop768 Teacher). Encoder 최종 feature에 직접 적용한다.
- L_spatial = mean_ROI Σ_q c_q d(Decoder(F, ROI)_q, Teacher_q)/Σ_q c_q. 같은 teacher token index에 대응하는 ROI-relative query를 쓴다. 별도 relational/pooled decoder loss는 없다.
- L_global = mean_image 0.5[d(CLS_s,CLS_0)+d(valid_patch_mean_s,valid_patch_mean_0)]. Full384 frozen Original teacher의 기존 G1 규칙이다.
- B0: L_crop + 0.5 L_global. B1: B0 + λ_spatial L_spatial.
- B2 (gate 통과 시에만): B1 + 1.0 L_pixel. 학습용 LN→Linear768→GELU→Linear768 head가 각 token에서 RGB16×16을 복원한다. Decoder 입력은 Student feature뿐이며, Student가 실제 관측한 Full384 RGB의 patch-level mean L1을 쓴다. Edge는 진단만 기록한다. Denoising/Flow는 없다.

λ_spatial은 초기 동일 첫 stream mini-batch에서 Encoder 전체 gradient를 각각 구해 `0.5 ||∇E L_crop|| / ||∇E L_spatial||`로 한 번 고정한다. Calibration에서 가중치를 학습하거나 Val/Test를 보지 않는다. 초기 spatial gradient가 비영(非零)이므로 decoder warmup은 사용하지 않는다. 측정 값과 patch embedding/12 block별 norm은 `lambda_spatial.json`에 저장한다.

```mermaid
flowchart LR
  X["Full RGB384"] --> E["표준 trainable DINOv3 12 blocks"]
  E --> F["24×24×768 patch features + CLS"]
  F --> C["직접 ROI pool / L_crop"]
  F --> D["학습용 Fine Decoder + ROI geometry"]
  D --> S["48×48×768 / L_spatial"]
  T["Frozen native Crop768 Teacher"] --> C
  T --> S
  G["Frozen Full384 Teacher"] --> L["L_global"]
  F --> L
  F --> P["B2 학습용 Pixel Decoder / RGB L1"]
```

Spatial/Pixel loss의 backward gradient만 Encoder에 전달된다. Forward에서 Decoder 출력이 Encoder로 되돌아가는 경로는 없다. 추론은 `Full RGB384 → 동일 표준 DINOv3 → 24×24×768`뿐이며 평가용 R2/Linear Probe를 따로 붙인다.

## 데이터와 학습

사용자의 후속 지시에 따라 **기존 GT + Random + fallback ROI를 모두 포함**한다. 기존 COCO train2017 stream 12,000행, 행당 16 images×4 ROI, augmentation 없음, stream 생성 seed2027 및 모델 seed0을 유지한다. 축소 Probe manifest의 기존 제외 함수만으로는 전체 Official Split을 보호하지 못한다. **Official Val2410/Test9443의 모든 image ID(속성 미주석 포함)**, 기존 MD5/pixel-MD5/dHash exact audit, 확인된 Val/Test 근접 중복 train 이미지를 모두 제외한다. 원래 stream50491장에서 총641장을 제외한 **49,850 images/199,400 GT+random+fallback ROI**로 학습한다. 매 step의 원래16개 image slot은 유지하되 제외 slot은 건너뛴다. 같은 규칙을 B0/B1/FrozenDecoder에 적용한다. 첫 batch는15 images/60 ROI다. 검증 결과 train/dev의 Official Val/Test image overlap은 모두0이다. COCO dev는 추가 중복 제외 후126 images/504 ROI이며 모델 선택에 쓰지 않는다. 전체 ID와 검증은 `teacher/manifest.json`, `official_split_exclusion.json`에 기록한다. 원래 G1의 unsupervised COCO 학습에는 full official Val146/Test475 exact 이미지가 포함됐으므로 matched data ablation으로 취급하지 않는다. Test score나 Test method/epoch selection을 사용하지 않는다.

G1의 Encoder AdamW top lr5e-5, layer decay0.75, wd0.05 (1-D/token/pos 제외), betas(0.9,0.999), warmup600+cosine, module별 clip1.0, BF16, EMA0.999→1을 유지한다. Decoder lr3e-4. 원래 64 ROI global step을 GPU1개에서 처리하며 조건별로 GPU를 배정한다. 최대 물리 GPU2개만 사용한다. Encoder 모델 선택은 **12k 최종 EMA**로 사전 고정한다. Student raw도 별도 export한다. 500step마다 optimizer/모델/EMA/RNG recovery를 저장한다.

FrozenDecoder 대조군은 동일 Original Encoder를 고정하고 동일 spatial 가중치·Decoder·12k budget에서 Decoder만 학습한다. 이 Encoder는 원본과 bit-identical임을 state hash로 검증한다. 별도 PACO Probe 재학습은 원본 중복이므로 하지 않는다. 동일 COCO dev ROI에서 B1의 step별 reconstruction과 비교한다. COCO dev126 images는 진단용이며 Encoder/Probe epoch 선택에 사용하지 않는다. ROI별 Full384 patch 면적-equivalent token 수, 교차하는 cell 수, spatial cosine을 함께 저장한다. Tiny ROI에서 48×48 출력이 관측하지 못한 세부를 복원한다고 해석하지 않는다.

## PACO Official Validation

기존 `src.g1up.train_bb`, R2 Reader, `src.pdit.bbdata`의 bilinear ROIAlign24×24, `src.pacoattr.official_eval`을 그대로 사용한다. 새 AP/evaluator를 만들지 않는다. 새 `fgtrain.infer`는 exported 표준 Encoder와 기존 ROI sampling/Reader를 연결해 기존 공식 manifest 형식의 확률을 저장하는 adapter다. 본 모델의 이미지당 Encoder canvas 수가 1인지 실제 계수하며 RGB ROI 재인코딩이 없음을 검증한다. Fine/Pixel/Teacher는 instantiate하지 않는다.

Reader seed0 (기존 init seed7), 10epochs, PACO train 내부 dev1000 이미지 분리, 기존 excluded train split, ~128 ROI packs, hflip0.5, masked BCE, reader lr3e-4/head lr1e-3, AdamW wd0.05, 1epoch warmup+cosine, clip1.0을 유지한다. 전 조건 R2 selection은 **10 epoch 중 Official Val (Object AP+Part AP)/2 최대, 동점이면 앞 epoch**다. 공식 Val2,410 images/31,919 ROI를 사용하며 기존 probe의 축소 Val2,051 images 점수와 혼동하지 않는다. GT Box+Category Oracle, max_dets300, 기존 공식 Attribute AP를 쓴다. Overall은 Object/Part 평균이다. Pattern/Color/Material은 공식 type별 출력만 보고한다. 새 크기별 AP나 custom metric은 만들지 않는다.

Frozen Full384(bilinear R2) 및 Native C0 Crop384는 이미 완료된 동일 10epoch probe/공식 Val 전 epoch 결과를 검증 후 재사용한다. G1-cross Encoder-only는 과거 full-canvas R2와 ROI 처리 방식이 달라 이번에는 같은 bilinear R2를 10epoch 새로 학습한다. G1-cross 가중치는 기존 12k Crop384+Detail+Global final EMA에서 encoder만 추출한다. C0는 표준 Original Encoder이지만 **ROI별 원본 RGB 재인코딩**을 하는 정보 oracle이므로 배포 가능한 B 조건과 추론 비용이 다르다.

B2는 B1−B0의 Overall 및 두 Pattern AP 차이가 양수이고 paired-image bootstrap1000의 95% CI 하한도 모두 양수이며, 고정 ep10의 두 Pattern 차이도 양수일 때만 진행한다. 이는 seed0에 대한 보수적 진행 규칙이며 seed 안정성의 증명은 아니다. Bootstrap CI는 선택된 Reader epoch에 조건부이며 epoch 선택 변동을 포함하지 않는다. Reconstruction loss는 gate로 사용하지 않는다. 모든 선택은 Val이며 개발 중 Test는 사용하지 않는다. 사용자 추가 요청에 따른 최종 Test는 Val-selected Reader epoch와 Encoder hash를 먼저 고정한 후 별도로 보고한다. Test 결과로 선택·가중치·B2 gate를 변경하지 않는다.

## ADE20K 동일 Frozen Linear Probe

공식 ADEChallengeData2016의 train20,210/Val2,000 이미지와 150 classes가 실제 존재함을 확인했다. 기존 **`src.g1eval.ade`를 수정하지 않고** B0/B1 Export loader만 연결한다. Frozen Original 및 G1-pool/G1-cross의 기존 동일 프로토콜 결과는 checksum/설정을 확인해 재사용한다. B0와 B1은 둘 다 평가하고, B2를 실행했다면 함께 평가한다.

Head는 최종 LN patch feature만 읽는 Linear(768,150), BN/fusion/decoder 없음이다. Encoder eval/frozen/no_grad, head만 AdamW lr1e-3, betas(0.9,0.999), weight wd1e-4/bias wd0, 40,000 iterations, global batch16 (2GPU×8), seed0, warmup1500 (ratio1e-6), poly1.0로 학습한다. 동일 sample order, head initialization, worker12, 동일 augmented tensor를 모든 B encoder에 공급한다. 최종 iteration을 평가하며 Val epoch selection은 없다.

기존 표준 프로젝트 전처리를 우선하므로 **train은 384×384, validation은 short-side384를 유지한 가변 크기**다. 모든 입력을 정사각384로 찌그러뜨리는 새 프로토콜은 만들지 않는다. Train: aspect-preserving short-side384×U(0.5,2), random crop384, hflip0.5, ImageNet norm. 작은 이미지는 normalized RGB0/mask255로 padding. Val: short-side384, Patch16 배수 padding, padding crop 제거 후 logits를 원래 GT 좌표로 bilinear resize한다. Val patch grid는 원래 이미지 aspect에 따라 24×N일 수 있으며 이는 기존 baseline에도 동일하다. ADE label0→ignore255, 1..150→0..149, mask nearest resize, ignore/padding 제외, 전체 Val class별 intersection/union으로 mIoU를 계산한다. 기존 256 BRIXEL 실험의 B0/B1과는 다른 실험이다.

## 기존 실험과의 비교 범위

| 실험 | Encoder / 초기화 / Teacher | 이번 실험과 다른 점 |
|---|---|---|
| G1-pool / G1-cross (`src/g1x`) | 표준 DINOv3-B/16, Original init, Crop384,12k (Frozen은0step) | G1 detail은 24×24 및 decoder pooled term; 이번 target은 Native768 48×48, 단순 spatial cosine, λ gradient0.5; 기존 full-grid R2와 새 bilinear R2 구분 |
| A0 / P (`src/hcf`) | DINOv3 + RGB Patch8 Fine path 및 양방향 exchange; G1-pool EMA init, 추가12k (총24k) | Fine path가 추론에 남음. A0 계열은 object4+추가 nested ROI 감독도 사용해 본 G1 image16×ROI4 stream과 차이가 있음 |
| E1a / E1b (`src/overnight6`) | A0 Coarse–Fine, G1-pool 기반 common initial.pt, Native Crop768 pooled/spatial, 추가12k | **E1a를 B0로 재사용하지 않음**. Spatial target은 최대12×12 집계, Fine-grid 비교 및 작은 ROI 제외 |
| D1 / L2 Spatial·Relational (`src/decstudy`) | CLIP-L/14@336, G1-cross 초기화, Crop384(C1 비교 포함) | Backbone·초기화·ROI 및 inference decoder가 다름. 실패를 이번 plain DINOv3의 실패와 등치하지 않음 |
| PixelDiT Pure / Hybrid / P-C768 (`src/pdit`, `src/pc768mse`) | A0 Coarse–Fine, G1-pool 기반 common initial.pt, Flow/Denoising, Crop384 또는768, 추가12k | 학습/추론 backbone과 pixel objective가 다름. 이번 B2는 단순 patch RGB L1, training-only |
| H0/H1 (`src/hier`) | P384 Coarse–Fine + RGB640 Detail branch, H0=Full384→640 / H1=Native640 | 저장소의 H0/H1는 Shared Hierarchical Patch-Pixel 명칭이다. 추가 Detail branch가 추론에도 남아 본 training-only decoder와 다름 |
| MSD H0/H1 (`src/msd`) | Frozen DINOv2-B/14@336 및 decoder, H0=L12 / H1=L12+0.5L8+0.5L4 | 동일 이름의 별도 실험이며 PixelDiT 또는 DINOv3-only가 아님 |
| G1-Native (`src/g1native`) | G1-pool 기반 A0, 원본 Full image→768 Frozen Teacher; Crop384+Grid+Global | 본 실험의 Native ROI→Crop768 Teacher와 crop 범위가 다르며 A0 추론 Fine path가 남음 |
| J-series (`src/jointpg`) | G1-pool 기반 A0 common initial.pt에서 추가12k Crop+Flow+Grid 및 weight mixing | 기존 공식 Val 결과만 문맥으로 기록. 이번 λ 또는 모델 선택에 기존 Test 결과를 사용하지 않음 |

기존 `results/overnight_six/README.md`, `results/pixeldit_crop768_vs_mse/README.md`, `results/joint_pixeldit_g1/eval/results/fast_tables.md`에서 Spatial/Flow의 loss 개선이 Pattern AP 개선을 보장하지 않았음을 확인했다. Backbone 또는 평가가 다른 점수를 B0/B1의 matched ablation으로 취급하지 않는다.

Global cosine 및 L_global은 진단값이다. ADE20K는 공간 표현 보존을 평가하지만 ImageNet 등의 global classification 성능을 직접 대체하지 않으므로, 이 파일럿만으로 모든 Global 성능이 보존됐다고 주장하지 않는다.

## 재현 및 제어

환경: `/home/irteam/envs/dinov2_ord_py310/bin/python`; 공식 PACO evaluator만 기존 `paco_eval` 환경을 쓴다. 코드와 의존 소스 SHA256, 초기화 hash, 캐시 표본·설정·버전은 run 아래에 보존한다.

```bash
# 최초 준비 (기존 allocation은 재색인하지 않음)
python -m src.fgtrain.setup
# 실제 smoke/calibration (기존 background jobs는 checkpoint pause)
python -m src.fgtrain.preflight
# 재시작 가능한 전체 queue; 완료된 단계는 state marker로 건너뛴다.
python -m src.fgtrain.pipeline
# 읽기 전용 결과 표 갱신
python -m src.fgtrain.report
```

Queue는 물리 GPU0/1만 사용한다. 최초 실행의 전체 pause 정책은 아래의 최신 GPU별 병행 정책으로 대체했다. 원래 사용자의 pause는 보존한다. Queue PID는 `pipeline.lock`, GPU/CPU child PID와 명령은 `active.json`에 있다. 중단된 학습은 500step recovery에서 다시 시작한다. Pipeline 자체의 SIGTERM은 자체 child process group만 종료한다.

2026-11-10 사용자 추가 지시: GPU 활용률80% 이상(가능하면100%)을 목표로 기존 A/B 및 대조군/미완료 평가와 파일럿을 동시 실행한다. 전체 PAUSE_all은 해제했고, bgjobs supervisor의 fgtrain_s0 동시 실행 허용과 GPU별 기존 작업 분산을 적용했다. 기존 체크포인트에서 재개하며 학습 설정·예산은 변경하지 않는다. `background_policy.json`, `resource_update.json`, `gpu_rebalance.json`을 참조한다.

최신 GPU 운영 지시(2026-11-10 13:46 UTC): 두 GPU의 활용률80% 이상을 우선하여 ADE20K 평가 중에도 기존 A/B를 병행한다. `src.fgtrain.resources`의 `background_policy.json`이 FrozenDecoder 및 ADE의 병행을 허용한다. 완료된 A/B의 새 초기화 반복도 사용자가 명시적으로 허용하여, 동일 설정·seed0·100k 예산의 A_r3(GPU0), B_r3(GPU1)를 별도 디렉터리에 등록했다. 기존 결과를 보존하며, 활용률을 위한 동일 seed 반복은 독립적인 과학적 근거로 세지 않는다. `gpu_allocation.json`이 실시간 배정의 기준이다. 병행으로 평가 소요 시간은 늘어날 수 있다.

운영 정책 갱신(2026-11-10 14:21 UTC): 사용자가80% 이상을 유지하면 A/B 실행량을 줄여 평가를 가속해도 된다고 허용했다. A/B를22초 일시 정지해 측정한 결과 ADE 학습만으로도 GPU0/1 각각89~100%,90~100%를 유지했다. 따라서 dense ADE 학습 중 A/B는 checkpoint pause하고, 최종 head 저장 이후의 단일 이미지 validation 및 완료 후에 자동으로 재개한다. 실험의 데이터·optimizer·배치·seed·학습 예산은 그대로이며 실행 동시성만 변경했다. 측정 원본은 `background_throughput_measurement.json`, 현재 정책은 `background_policy.json`이다. 앞의 전체 병행 정책을 이 정책이 대체한다.

ADE 실행 오류 및 복구: 첫 torchrun 실행은 `--run` 인자가 launcher의 `--run-path`와 혼동되어 평가 시작 전에 종료됐다. module 앞에 명시적인 `--` 경계를 넣고 실제 launcher parser로 전달 인자를 검증한 후, 완료된 학습/PACO는 재사용하여 ADE만 재개했다. 기록은 `ade_launch_recovery.json`, `launch_failures/`에 보존했다. 이후 실행은 기존 프로젝트의 CUDA JIT cache를 명시적으로 활성화하여 NCCL의 최초 컴파일 지연을 줄인다. 이 수정은 데이터·모델·평가 프로토콜을 변경하지 않는다.


## 실행 결과 (측정된 항목만 기입)

| Method | PACO Overall AP | Object AP | Part AP | Object Pattern AP | Part Pattern AP | ADE20K mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 54.213 | 52.266 | 56.159 | 51.944 | 52.787 | 49.367 |
| B0 | 55.450 | 54.511 | 56.390 | 59.644 | 55.243 | 44.829 |
| B1 | 55.779 | 54.327 | 57.231 | 58.186 | 56.376 | 45.453 |
| B2 | 조건부 미실행 | 조건부 미실행 | 조건부 미실행 | 조건부 미실행 | 조건부 미실행 | 조건부 미실행 |
| G1cross | 55.993 | 54.670 | 57.315 | 58.130 | 55.796 | 45.703 |
| C0 | 56.395 | 55.699 | 57.091 | 61.978 | 61.072 | — |

Color / Material (공식 evaluator 출력):

| Method | Object Color | Part Color | Object Material | Part Material | Val-selected R2 epoch |
|---|---:|---:|---:|---:|---:|
| Frozen | 42.736 | 50.396 | 52.403 | 57.031 | 5.000 |
| B0 | 45.183 | 52.671 | 51.225 | 56.587 | 7.000 |
| B1 | 46.525 | 53.480 | 50.875 | 56.483 | 7.000 |
| B2 | — | — | — | — | — |
| G1cross | 47.288 | 54.532 | 52.589 | 56.316 | 7.000 |
| C0 | 48.148 | 55.068 | 51.797 | 54.029 | 5.000 |

### Frozen 대비 변화 (percentage points)

| Method | Overall | Object | Part | Obj Pattern | Part Pattern | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| B0 | +1.238 | +2.245 | +0.230 | +7.701 | +2.455 | -4.538 |
| B1 | +1.566 | +2.061 | +1.072 | +6.242 | +3.589 | -3.914 |
| B2 | — | — | — | — | — | — |
| G1cross | +1.780 | +2.404 | +1.156 | +6.186 | +3.009 | -3.664 |
| C0 | +2.182 | +3.432 | +0.932 | +10.034 | +8.285 | — |

### B0 대비 변화 (percentage points)

| Method | Overall | Object | Part | Obj Pattern | Part Pattern | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | -1.238 | -2.245 | -0.230 | -7.701 | -2.455 | +4.538 |
| B0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| B1 | +0.329 | -0.184 | +0.841 | -1.458 | +1.133 | +0.624 |
| B2 | — | — | — | — | — | — |
| G1cross | +0.542 | +0.159 | +0.926 | -1.514 | +0.553 | +0.874 |
| C0 | +0.945 | +1.188 | +0.702 | +2.334 | +5.829 | — |

## 실제 검증

- Smoke: **passed**. 좌표 최대 오차 4.77e-07; Padding target 변경 시 loss 동일.
- Spatial loss → Encoder norm 0.438010; Patch embedding 0.198866. 12개 Block 모두 gradient > 0.
- 실제 optimizer update의 Patch embedding 최대 변화: 5.00679e-05.
- Smoke strict export: 85,641,216 parameters, 표준 클래스 `timm.models.eva.Eva`, 576×768 output, Module tree 동일, Forward bit-equal. 최종 12k export 검증과 구분한다.
- 고정 λ_spatial = **1.06140601248**; 같은 초기 64 ROI batch에서 Crop gradient 0.854029, Spatial gradient 0.402310, 가중 비율 0.500000. Decoder warmup/update 없이 측정.
- B0 최종 strict export: True, parameters 85,641,216, Forward bit-equal True, 추가 module [].
- B1 최종 strict export: True, parameters 85,641,216, Forward bit-equal True, 추가 module [].
- FrozenDecoder 최종 strict export: True, parameters 85,641,216, Forward bit-equal True, 추가 module [].

## Reconstruction / Encoder 진단

| Method | COCO dev Crop cos | CLS cos | Patch-mean cos | Spatial cos (init → final EMA) |
|---|---:|---:|---:|---:|
| B0 (step 12000) | 0.87602 | 0.98186 | 0.97735 | 해당 없음 |
| B1 (step 12000) | 0.87735 | 0.97749 | 0.97379 | 0.00468 → 0.83156 |
| FrozenDecoder (step 12000) | 0.68964 | 0.99997 | 1.00000 | 0.00468 → 0.82522 |

동일 12000step에서 Trainable B1−Frozen Decoder의 spatial cosine 차이는 +0.006344이다. Decoder 학습만으로 줄어드는 loss와 Encoder 변경의 기여를 구분한다.

## 결과 해석

B1−B0: Overall +0.329 pp, Object Pattern -1.458 pp, Part Pattern +1.133 pp.
ADE20K B0−Frozen -4.538 pp; B1−B0 +0.624 pp. 작은 차이는 seed-0 파일럿 변동과 구분하지 못한다.

기존 official matching 기반 paired-image bootstrap (1000회, 95% CI; seed 반복 실험과 다름):

| B1−B0 | Δ AP (pp) | 95% CI |
|---|---:|---:|
| Object | -0.184 | [-1.382, +1.646] |
| Part | +0.841 | [-0.158, +1.929] |
| Object Pattern | -1.458 | [-3.820, +3.668] |
| Part Pattern | +1.133 | [-0.164, +4.219] |
| Overall | +0.329 | [-0.485, +1.524] |

두 평가의 점추정이 함께 개선됐다. 작은 차이는 seed0 파일럿 변동 가능성이 있어 seed 반복 없이 안정적인 개선으로 부르지 않는다.
B2 자동 진행: False. 사전 고정 gate와 bootstrap CI는 `b2_gate.json`, `paired_bootstrap.json`에 저장한다.
B1의 안정적인 Pattern 이득을 확인하지 못해 B2를 실행하지 않았다. Native Crop768 감독의 이 결과는 C1 Crop-from-384 판별 학습의 실패를 뜻하지 않는다. Decoder의 복원 향상과 Encoder의 AP 변화, 기존 PixelDiT/J-series 결과를 함께 검토한다.

## G1 기존 Reader 참고 결과 (matched ablation과 분리)

아래는 같은 Official Val2410이지만 기존 full-canvas R2 및 dev-selected ep6 결과다. 새 24×24 bilinear ROIAlign/Val selection과 epoch 규칙이 다르므로 B0/B1의 직접 ablation으로 사용하지 않는다. G1-pool/cross는 Original 초기화에서 Crop384로12k 학습했고 Frozen은 COCO 학습0step이다.
기존 G1 Encoder 사전학습은 full official Val146/Test475 exact 이미지에 노출됐다(해당 이미지에는 속성 positive 주석이 없음). 새 B 조건은 이 이미지 및 확인된 near duplicate를 전부 제외하므로 사전학습 data scope가 다르다. 새 G1-cross bilinear Reader 결과 역시 Encoder는 이 과거 가중치를 쓰는 참고 결과다.

| Reference | Overall | Object | Part | Obj Pattern | Part Pattern | ADE20K mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen full-grid | 54.389 | 52.204 | 56.574 | 56.583 | 55.470 | 49.367 |
| G1-pool full-grid | 56.534 | 54.846 | 58.222 | 57.123 | 58.509 | 45.416 |
| G1-cross Encoder Only full-grid | 56.301 | 54.993 | 57.609 | 55.764 | 56.385 | 45.703 |

## 기존 결과와 다음 실험의 근거

공식 Val의 기존 A0 계열에서 E1a→E1b는 Overall 57.11→56.63, Object Pattern 57.95→59.23, Part Pattern 59.22→59.44였다. Spatial loss가 줄어도 Overall 이득은 확인되지 않았고, 현재 표준 DINOv3와는 G1-pool 초기화·추론 Fine path·spatial target 집계가 다르다.
기존 PixelDiT Hybrid→P-C768의 Val Overall은 56.71→56.37, Joint J1의 Val Overall은 56.40이었다. 이 결과는 Flow 구조의 참고 결과이며 이번 단순 Pixel L1의 효과를 입증하거나 부정하지 않는다. 기존 Test는 이번 가중치·선택·gate에 사용하지 않았다.
B1의 Pattern 이득이 불충분하면 이 결과를 유지하고 B2를 보류한다. 우선 ROI token 수별 spatial cosine과 FrozenDecoder 대조군을 비교해 decoder capacity에 의한 loss 감소인지, 관측 정보의 제약인지 구분한다. 추가 loss나 λ 탐색으로 본 파일럿의 음성 결과를 덮어쓰지 않는다. Native768와 C1-from-384는 다른 정보 조건이므로 후속 가설도 구분한다.

## 구현 중 수정과 제외 실행

Concurrent teacher refresh changed finite pooled/global targets; freeze target byte hashes, restore references, restart encoder.
제외 실행: `B0_initial_cache_trial`. 표와 모델 선택에 사용하지 않았다.

Legacy exclusion scope omitted unannotated images in full official Val/Test; remove full official IDs and confirmed near duplicates, restart encoder from Original.
제외 실행: `B0_incomplete_official_split_trial`. 표와 모델 선택에 사용하지 않았다.


## 파일과 상태

실행 원본: `/home/irteam/Crop_representation/output/2026-11-10/fgtrain_s0`. `active.json`, `events.jsonl`, `state/`, `logs/`, 각 condition의 `progress.json`을 참조한다.
`initialisation.json`, `teacher/manifest.json`, `smoke_verification.json`, `lambda_spatial.json`이 초기화·데이터·좌표·gradient 근거다.
`<method>/per_loss_gradients.jsonl`, `block_gradients.jsonl`, `diagnostics/`에 patch embedding 및 12개 block gradient와 ROI별 관측 token 수·복원 cosine을 보존한다.
`<method>/export/encoder.pt`는 raw 표준 DINOv3 state dict; `encoder_eval.pt`는 기존 평가기용 enc-only envelope다. Teacher/Decoder 가중치는 포함하지 않는다.
`<method>/probe/frozen_check.json` 및 `preds/inference_verification.json`으로 Probe 중 Encoder 불변성과 이미지당 1회 인코딩을 확인한다.
`ade/config.json`, `ade/done.json`, `ade/val_inter_union.npz`, `ade/head_*.pt`가 ADE20K 재현 artifact다.

## 최종 PACO Test (Val 선택 고정 후 별도 보고)

사용자 추가 요청으로 최종 Test를 실행한다. B0/B1은 각각 Official Val에서 선택한 R2 epoch를 고정하고 Encoder/Reader hash를 추론 전에 저장했다. Test로 method/epoch/가중치 또는 B2 gate를 변경하지 않는다. 위 표는 계속 Val 결과이며 Test와 섞지 않는다.

| Method | Val-selected epoch | Test Overall | Object | Part | Object Pattern | Part Pattern |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 7 | 41.005 | 41.556 | 40.454 | 49.176 | 44.296 |
| B1 | 7 | 41.574 | 41.975 | 41.173 | 49.509 | 44.179 |

원본: `test/fixed_protocol.json`, `test/active.json`, `test/results.json`, `test/<method>/official/official_results.json`.
