# Multi-depth Crop-to-Shared Distillation (MSD) 파일럿: 실행 전 문서 (2026-11-05)

## 연구 질문
Crop teacher의 최종 feature만 복원하는 대신 여러 깊이(Block 4/8/12)의 지역 표현을 학습시키고, 그 감독을 공유 encoder까지 전달하면 encode-once 상태에서 더 유용한 spatial ROI representation을 만들 수 있는가?

## 단계
1. **Decoder 사전학습: COCO train.** PACO attribute label은 쓰지 않는다. 기존 RAEv2/R00 COCO 파이프라인을 재사용한다.
2. **Decoder checkpoint 선택: COCO dev**의 Block-12 feature matching loss(L12)로 고른다. 1,000 step마다 평가한다. PACO 성능으로는 고르지 않는다.
3. **PACO probe.** Encoder와 decoder를 완전히 frozen한 상태에서, PACO train으로 새 R2만 학습한다. R2 epoch은 PACO dev로 고르고, 최종 평가는 PACO **Validation**에서 한다(Primary M0 Overall). Test는 쓰지 않는다.

## Architecture (H0–H3 공통)

| 구성 | 내용 |
|---|---|
| Student encoder | DINOv2-B/14 (timm lvd142m), full image 336 letterbox, 이미지당 1회 forward. Block 4/8 출력(raw residual)과 Block 12 출력(final LN)을 쓴다 |
| Encoder adapter (H2/H3만) | Block 4/8/12 출력의 residual stream에 `h ← h + A(h)`. A = LN → 768×192 → GELU → 192×768이고 마지막 층을 0으로 초기화한다(시작 시 원래 encoder와 출력이 같음). Backbone 가중치는 frozen. Block 4 adapter의 변화는 Block 5–12로 전파된다. 3개 합계 0.89M 파라미터 |
| ROI decoder | 기존 R10 `MLRDecoder` 규칙을 두 층으로 확장했다. ROI(+ring) token에서 `F = F12 + A4(F4) + A8(F8)`을 만들고(A는 zero-init adapter), 기존 `CoordMLP` body(9.86M, 코드를 그대로 사용)를 거쳐 **Z: 24×24×768 (crop grid, Block-12 feature space)**를 출력한다 |
| 보조 projection head | `ẑ_d = Linear_d((Z − μ12)/σ12)` (d = 4, 8). **모든 조건에 같은 구조로 존재**하고, 감독 weight만 다르다. R2 평가에서는 제거하고 Z만 쓴다 |
| Teacher | 원본 COCO 이미지의 GT crop → 336 letterbox (C0 방식, `roi_geometry`) → frozen DINOv2 → Block 4/8/12. 학습 감독에만 쓰고 student 추론 경로에는 없다. Student backbone과 같은 frozen 가중치를 adapter 없이 쓰며, gradient는 없다 |

## Loss (가중치와 정규화)
- **Depth별 정규화(모든 depth 동일):** `x̂ = (x − μ_d)/σ_d`. 채널별 μ, σ는 stream 처음 47 step(3,008 ROI)의 teacher 유효 cell로 구했다(`teacher_stats.npz`). 실제 채널 std 평균은 block 4 0.04, block 8 0.18, block 12 1.59로 스케일이 크게 다르다.
- **L_d** = teacher-cell content coverage `w`(padding cell은 0)로 가중한 평균의 `[mean_c (p̂ − t̂)² + (1 − cos(p̂, t̂))]`. p̂12 = 정규화한 Z, p̂4 / p̂8 = 보조 head 출력.
- **H0:** L12
- **H1:** L12 + 0.5·L8 + 0.5·L4
- **H2:** L12 + 1.0·L_glob
- **H3:** L12 + 0.5·L8 + 0.5·L4 + 1.0·L_glob
- **L_glob(H2/H3 공통):** frozen 원본 full encoder와 비교한 `½[(1 − cos CLS) + (1 − cos 평균 patch token)]`의 이미지 평균.

## 학습 (기존 COCO decoder 레시피)
| 항목 | 설정 |
|---|---|
| 데이터 | 기존 12k stream (`cache/stream/stream_ids_12000.npy`, seed 2027): step당 COCO train 이미지 16장 × ROI 4개 = **global ROI batch 64**. ROI는 COCO 객체 annotation box이고, 객체가 4개 미만인 이미지는 임의 box로 채운다(기존 coco_rois pool 그대로, 전체 ROI의 36%가 임의 box). Flip 없음 |
| 제외 | Stream에서 PACO val/test와 ID·byte md5·pixel md5·dHash가 일치하는 2장(388662, 397819). Dev에서 32장(dev는 468장, 1,872 ROI). PACO val/test ID는 stream을 만들 때 이미 제외됐다 |
| 최적화 | AdamW lr 1e-4 (0.9, 0.95), wd 0.05, warmup 600 + cosine (0까지), clip 1.0, BF16, **12,000 optimizer step**, seed 0 (decoder init seed 0, adapter seed 1. decoder init은 4조건 모두 동일) |
| 실행 | GPU 2+3 DDP. 각 rank가 step의 이미지 8장을 맡고, gradient를 합산한다(64-ROI step 1회와 동일). Checkpoint는 4k/8k/12k와 best_dev를 조건별 폴더에 저장 |

## PACO probe (H0–H3 및 기준선 모두 동일)
- PACO train 전체(중복 6장 제외)에서 **PACO dev 1,000장**을 이미지 단위로 고정해 뺀다(rng 0, R2 epoch 선택 전용). 나머지 37,472장으로 R2를 학습한다. PACO 공식 분할에는 dev가 없어서 train에서 떼어냈다.
- R2: 기존 readout 큐와 같은 레시피다. 10 epoch, 약 128 ROI/step 이미지 pack, hflip, masked BCE, AdamW 3e-4 / head 1e-3, seed 0.
- Feature는 frozen encoder와 decoder가 만든 Z(crop grid)이고, R2 geometry는 C0 crop-grid 규칙을 따른다.
- 기준선도 같은 프로토콜로 다시 probe한다.
  - Bilinear: decoder 없이 원본 Block 12를 crop grid에서 bilinear로 읽음
  - Full Multi-layer+R2: full image token을 직접 읽음
  - GT crop C0
- Frozen 검사: 모든 encoder/decoder 파라미터는 `requires_grad=False`이고, 학습 전후 체크섬이 같아야 한다(`frozen_check.json`). 다르면 실행이 실패한다.
- Primary는 PACO val **M0 Overall**이고, R2 epoch은 dev M0 Overall이 가장 높은 것을 쓴다. Secondary는 Object/Part/Color/Material/Pattern(M0), 보조로 M2. 이미지 단위 paired bootstrap 95% CI.

## Smoke test에서 코드로 확인한 것 (2026-11-05 01:58)
- **H0/H1:** encoder backbone에 gradient가 없고(None) 가중치 변화가 0이다. Decoder body는 첫 step부터 gradient와 갱신이 있다.
- **H2/H3:** encoder adapter fc2(zero-init)는 step 1에서 gradient가 약 1e-16이다. Decoder 마지막 층이 0으로 초기화되어 있어 gradient가 위로 전달되지 않기 때문이다. Step 2부터 gradient 1.8e-3, 갱신 0.011로 학습된다. fc1은 fc2가 0이 아니게 된 뒤부터 gradient를 받는다(step 1–5를 `grad_check.json`에 기록).
- **Teacher:** gradient 없음(no_grad, frozen 가중치).
- **Aux loss(L4 + L8)만 역전파:** decoder body에 gradient norm 0.31이 들어간다. 즉 보조 head만 학습되는 것이 아니다.
- **Probe 단계:** encoder와 decoder는 `requires_grad`가 모두 False이고 gradient가 없으며, 체크섬이 같다.

## 기존 실험과의 차이
| 실험 | 이 서버 | 이번과의 차이 |
|---|---|---|
| R00/R10 (multilevel_relation_decoder) | 코드 있음, checkpoint 없음 | 같은 `CoordMLP` body와 adapter 규칙을 쓴다. R10은 block 6 하나만 입력에 더하고 loss는 raw final-layer `MSE + (1 − cos)` 하나다. 이번 실험은 block 4/8 입력, depth별 정규화 loss, multi-depth 감독, encoder adapter와 global loss를 쓴다. R00/R10의 body는 gen_decoders mlp_s0로 초기화했지만, 그 checkpoint가 이 서버에 없어 이번에는 처음부터 학습한다 |
| Pretrained Region Decoder (dinov2_oracle_region_decoder) | 코드 있음, checkpoint 없음 | 재사용할 checkpoint가 없다 |
| C2 (11-04 슬라이드 "Enc + Decoder") | 코드·설정·checkpoint 없음 | 슬라이드 수치(test, attention reader 41.02)만 있고 encoder를 어떤 형태로 학습했는지 확인할 수 없다. 그래서 직접 비교하지 않는다. 확인 가능한 구조적 차이는 이번 실험의 backbone은 frozen이고 zero-init residual adapter만 학습하며 global preservation loss를 쓴다는 점이다 |
| 4090 D0–D3, Shared Detail Memory | 이 서버에 없음 | 비교하지 않는다 |

**Multi-depth 감독 외에 달라진 변수 (H0–H3 내부):** H2/H3에만 encoder adapter와 L_glob가 있다. 그래서 H2−H0은 "adapter + L_glob" 묶음의 효과다. 이외의 데이터, ROI 순서, decoder 구조와 init, step, 레시피는 모두 같다.

## 판정
- H1−H0 = multi-depth 감독 효과, H2−H0 = encoder adaptation(+ global loss) 효과, H3−H2 = adaptation 위에서 multi-depth 감독의 추가 효과. 모두 PACO val M0 Overall의 paired CI로 본다.
- Feature matching이 좋아져도 PACO가 개선되지 않으면 성공이 아니다.
- 핵심 질문은 공유 encoder와 decoder가 만든 spatial feature 자체의 유용성(frozen probe)이 개선됐는지다.
