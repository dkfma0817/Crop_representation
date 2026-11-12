# Idea Search: 작은 물체를 위한 crop 표현 얻기

작성: 2026-11-04 · 참여: Areum Kim, Claude
근거 자료: `../meeting slide/` (261027, 261029, 261031, 261103, 261104), 교수님 녹취 `../record/261027.txt`, `261029.txt`, `261031.txt`
같은 날 세부 기록: [2026-11-04.md](2026-11-04.md) (미팅 흐름, 교수님 방향), [2026-11-04_direction.md](2026-11-04_direction.md) (방향 결정 과정)
방법 그림: [Foveated Encoding 방법 가이드](https://claude.ai/artifact/3KFam6wmNC3mW9sXzHR2bC) (비공개, 공유 필요)

---

## 0. 한 문단 요약

작은 물체(5–16 token)의 색·형태·무늬는 이미지를 한 번 인코딩한 feature에서 사후에 어떻게 읽거나 복원해도 회복되지 않는다. 10-27 ~ 11-04에 시도한 decoder, 보간, low-level 재사용은 모두 같은 한계(5–16 obj 38–40)에 걸렸다. 반면 encoder **안에서** 그 물체에 token을 더 배정하면 새 픽셀 없이도 45–46까지 오른다. 사람이 주변시로 훑고 중요한 곳을 중심와로 다시 보듯이, 질의 영역에 encoder 안에서 token을 배정하고 전역 문맥은 공유하는 방법(foveated encoding)이 방향이다. 그 token을 어떻게 만들지에 따라 세 가지 아이디어가 나온다: (I1) crop처럼 촘촘한 token + 문맥 참조, (I2) box로 조건화한 소수의 질의 register, (I3) "새"처럼 개념으로 조건화한 register.

---

## 1. 출발점

### 1-1. 연구 가정 (Areum)
- **"crop된 표현을 찾아서 주면 잘 학습한다."** 이득은 표현 자체에 있다.
- 단순히 해상도를 높이는 것(전체 이미지를 고해상도로 인코딩)은 연구 목적이 없어지므로 하지 않는다.
- 사람은 전체를 다 보고 찾지 않는다. 중요한 곳을 보고 그곳의 색·형태·무늬를 본다.

### 1-2. 교수님이 보는 틀 (녹취 10-27 / 10-29 / 10-31)
- **"Feature 단위 zoom"**: 한 번 인코딩한 feature에서, 원하는 영역(박스든 임의 mask든)에 대해 crop만큼 긴 token sequence를 얻는 것. "Feature segmentation"이라는 관점이 새롭다고 봄.
- 생성 문제가 아니므로 결과는 하나로 정해지면 됨 (diffusion noise 불필요).
- 반복 질문
  - Q1. 그냥 crop해서 다시 넣으면 되지 않나? 효율 말고 더 좋은 점은? (힌트: 우리 feature는 전체를 봤고 crop은 바깥을 못 본다)
  - Q2. Bilinear로 늘리면 왜 안 되나?
  - Q3. 기존 detection / segmentation의 low-level feature 재사용과 뭐가 다른가?
  - Q4. (i) low-level 재사용으로 충분한가, (ii) learnable token이 빈자리를 채워야 하나? → 의견을 가져올 것
  - Q5. 위치를 못 찾는 문제인가, 위치를 알아도 표현을 못 하는 문제인가?
- 요구: 결론 먼저, 근거는 뒤에. 주장을 글로 먼저 쓰고 그걸 보여줄 가장 단순한 실험.
- **마감: 10-31 기준 약 2주 뒤 (11월 중순).** 목표는 드래프트 한 번 완성.

---

## 2. 확립된 사실 (frozen DINOv2-B/14, PACO test M2, GT ROI, R2 attention reader)

| 입력 / 방법 | token 배정 | 원본 픽셀 | 5–16 Obj | Overall | 출처 |
|---|---|---|---|---|---|
| F336 (한 번 인코딩) | encoder 밖 | — | 38.10 | 40.69 | 11-03 |
| Bilinear ×2 / AnyUp ×2 | encoder 밖 | — | 37.41 / 37.06 | 40.56 / 40.00 | 11-03 |
| Bilinear ×8 / ×14 | encoder 밖 | — | 38.79 / 38.07 | 41.17 / 41.14 | 11-03 |
| Decoder A/B/S, encoder 공동학습 C | encoder 밖 | — | 37.9–38.4 | 40.6–41.7 | 11-04 |
| Detail CNN (D) | encoder 밖 | 672 CNN | 40.44 | 42.72 | 11-04 |
| F336 multi-layer readout | encoder 밖 | — | 38.50 | 43.27 | 11-04 |
| F672 (확대 후 재인코딩) | encoder 안, 29 | 없음 | 44.95 | 43.67 | 11-03 |
| Native F672 | encoder 안, 29 | 있음 | 46.62 | 45.14 | 11-03 |
| C1 crop (336 이미지에서 자름) | encoder 안, ~380 | 없음 | 46.06 | — | 11-03 |
| C0 crop (원본에서 자름) | encoder 안, 382 | 있음 | **54.75** | 45.44 | 11-03 |

추가 사실
- **F1. Crop 표현은 더 잘 학습된다.** 같은 PACO label로 encoder까지 fine-tune: Full 40.57 / C1 47.27 / C0 49.41 (Mean reader). Full은 train M2 91.7까지 맞추지만 test 격차가 2.09 → 6.69로 커진다. (11-03 App.)
- **F2. 이득의 원인은 encoder 안의 token 배정.** 위 표의 위·아래 블록. 보간으로 token을 1,000개 넘게 늘려도 효과 없음.
- **F3. 원본 픽셀은 token이 충분할 때만 쓸모.** 29 token에서 +1.67 (F672 → Native), 380 token에서 +8.69 (C1 → C0).
- **F4. 문맥은 필요하다.** ROI 밖 attention 차단 −5.0, gray mask −2.8 (Overall, 11-03 App.).
- **F5. 속성은 벡터 하나로도 많이 읽힌다.** C0를 mean pooling + linear(R0)로만 읽어도 5–16 obj 47.51 (F336 R0 31.59). (11-03 App. Reader Robustness)
- **F6. 크기별로 최적이 다르다.** 1–4 token part는 Native F672가 C0보다 +5.7 (C0는 과확대로 흐림). Part는 위치 찾기, object는 표현이 병목 (10-31 p6).
- **F7. 비용** (ROI당 encoder GFLOPs, 10-31): F336 19, F672 91 (이미지당), C1/C0 105 (질의당).

---

## 3. 아이디어

각 항목: 무엇 / 왜 / 위험 / 관련 연구와 차이 / 실험 / 상태

### I0. 한 번 인코딩한 feature + decoder로 crop 복원 — **종료**
- **무엇:** F336 token에서 ROI를 모아 decoder로 C0 token map을 복원 (10-27 ~ 11-04의 주된 시도).
- **결과:** 구조(Transformer, MLP, DiT), 데이터(ImageNet), loss(feature, visual, semantic), encoder 공동학습까지 모두 Bilinear +1 이내. 복원 지표는 좋아져도 PACO는 그대로 (README RAEv2 실험 결론과도 동일).
- **이유:** 물체에 배정된 token이 10개뿐이라 세부가 남아 있지 않다. Decoder 출력은 평균으로 수렴 (공간 분산 teacher의 0.26, 고주파 상관 0.03).
- **남기는 것:** "encoder는 고정, 가벼운 모듈만 학습"이라는 틀은 아래 I1–I3에서 유지.

### I1. 문맥 공유 Foveated Encoding — **1순위**
- **무엇:** 전체 이미지를 한 번 인코딩해 층별 token을 캐시(주변시). 질의 영역은 crop처럼 원본 픽셀로 576 token을 만들어 같은 frozen block을 통과시키되, 마지막 k block에서 캐시 token을 K/V로 참조(중심와 + 주변 유지). 문맥 token은 ROI를 보지 않으므로 질의가 여러 개여도 재사용.
- **왜:** F2 + F3 (token을 encoder 안에서, 원본 픽셀과 함께) + F4 (문맥 필요) + Q1 답 (crop은 바깥을 못 봄).
- **위험:** frozen 모델이 처음 보는 조합(좌표계가 다른 두 token 집합). 질의당 비용은 C0와 비슷하므로 이 단계는 상한 확인용이고, 비용 이득은 I2(token 수 줄이기)에서 나와야 함.
- **관련 연구와 차이**
  - [TEVA (ICCV 2025)](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.pdf): 핵심 영역 고해상도 patch + 저해상도 전역 patch를 VLM에 함께 넣음. 우리는 frozen ViT **내부**에서 문맥을 캐시로 공유하고 질의 시점에 영역을 정함.
  - [Quadformer (CVPRW 2023)](https://arxiv.org/abs/2304.00287): saliency 기반으로 영역별 patch 크기를 다르게. 우리는 질의마다 바뀌는 영역 + 문맥 재사용.
  - [Alpha-CLIP (CVPR 2024)](https://arxiv.org/abs/2312.03818): alpha 채널로 영역을 지정해 문맥을 유지하며 영역에 집중. 단 encoder를 RGBA로 fine-tune하고 물체 token 수는 그대로. 우리는 encoder 고정 + token 배정.
  - [ViCrop (2502.17422)](https://arxiv.org/abs/2502.17422), [HiDe (2510.00054)](https://arxiv.org/abs/2510.00054): MLLM에서도 crop이 핵심이고 전체 확대로는 안 됨. ViCrop은 전역 시야가 필요한 질문에서 실패 → 문맥 공유의 동기.
- **실험:** k ∈ {0, 3, 6, 12} × ROI 입력 {C0 원본, C1 대조군}, 학습 없음. k = 0이 C0(54.75)와 같아야 함(구현 검증). R2 seed 3 + R0.
- **판정:** C0 이상 → 문맥이 도움, I2로 비용 낮추기. C0 미만 → 문맥 attention에만 작은 adapter 학습.

### I2. 질의 register (box 조건) — **I1과 하나의 축으로 묶임**
- **무엇:** ROI 위치로 조건화한 학습 가능한 token N개를 F336 forward sequence에 넣고, encoder가 forward 중에 그 token에 물체 정보를 쓰게 한다. 학습은 token(+작은 adapter)만.
- **왜:** 교수님 Q4 (ii) "learnable token이 빈자리를 채운다"의 직접 구현. 336 이미지 layer-0 patch embedding에는 픽셀 정보가 거의 그대로 있고(14×14×3 = 588 → 768), F672가 회복한 것은 정보가 아니라 처리하는 token 수였다 → C1(46.06)까지는 원리적으로 가능.
- **설계 선택 (결과를 가름)**
  - 한 방향 (질의 → patch만 참조): patch 캐시로 매우 쌈. 하지만 사실상 frozen block을 쓰는 multi-layer reader라서 11-04 multi-layer(5–16 obj 38.50) 한계에 걸릴 위험.
  - 양 방향 (patch도 질의를 봄): 질의당 F336 forward 1회(약 19 GFLOPs, C0 105보다 쌈). Encoder가 그 물체를 위해 다시 처리 → 더 유망.
- **통합된 손잡이:** 질의 token 수 N. N = 576 + 원본 픽셀이면 I1, N을 줄이면 register식 경량 버전. N ∈ {4, 16, 64, 144, 576}이 곧 비용–성능 곡선.
- **관련 연구와 차이**
  - [Vision Transformers Need Registers (ICLR 2024)](https://arxiv.org/abs/2309.16588): 픽셀에 대응하지 않는 token을 넣어 encoder 안에서 역할(전역 정보 저장)을 맡김. 구조적 선례. 우리는 전역이 아니라 **질의 영역 세부**를 맡김.
  - [Vision Transformers Don't Need Trained Registers (NeurIPS 2025)](https://arxiv.org/abs/2506.08010): 학습 없이 high-norm activation을 추가 token으로 옮김.
  - [VPT-deep (ECCV 2022)](https://arxiv.org/abs/2203.12119): frozen ViT 층마다 학습 prompt token. 우리는 prompt가 **질의(ROI)마다 조건화**되고 target이 crop 표현.
  - [T-REN (2604.18573)](https://arxiv.org/abs/2604.18573), [ReVA (2608.28707)](https://arxiv.org/abs/2608.28707), [TextRegion (TMLR 2025)](https://arxiv.org/abs/2505.23769), [BLIP-2 Q-Former](https://arxiv.org/abs/2301.12597): frozen encoder **출력 위에서** learnable query가 cross-attention으로 region token을 만듦 = 한 방향 방식. 우리 데이터(F2, multi-layer 38.50)는 이 방식의 한계를 보여주므로, **양 방향(encoder 내부)** 이 차별점이자 검증 포인트.
- **실험:** 양 방향, N = 16 (ROI pos-embed + 학습 query), target = C0 (token map 또는 pooled), C1 46.06 대비 회복률. 이후 N sweep, 원본 픽셀 초기화.

### I3. 개념 register ("새를 보고 있으면 '새' token을 추가") — **Q5에 바로 답함**
- **무엇:** I2의 register를 box가 아니라 개념(category 이름 text embedding)으로, 또는 개념 + box로 조건화. Encoder가 forward 중에 "새"에 관한 정보를 그 token에 모음.
- **왜:** 사람이 "새"를 찾아 그 새를 보는 것과 같음. 그리고 **F5**: 속성은 C0 pooled 벡터 하나로도 47.51까지 읽힘 → target을 24×24 map 대신 **C0 pooled 벡터**로 잡으면 token 1~16개로 학습 목표가 맞는다.
- **위험**
  - 같은 종류가 여러 개면 어느 것인지 모름 (SteerViT heatmap 실패율 단일 7.7% vs 다중 36%, 10-31 p6).
  - Part 질의는 부모 물체와 같은 곳을 봄 (heatmap cos 0.89).
  - → 개념만으로는 부족할 수 있고 개념 + 위치(box나 점)가 필요할 수 있음.
- **관련 연구와 차이**
  - [SteerViT (2604.02327)](https://arxiv.org/abs/2604.02327): frozen DINOv2 block 사이에 text cross-attention을 넣어 patch token 자체를 text로 조정. 우리 baseline (PACO 32.61, GT box 없음). 차이: SteerViT는 기존 patch token(작은 물체는 여전히 ~10개)을 바꾸고, 우리는 **전용 token을 추가**하고 **crop 재인코딩을 target**으로 학습.
  - [CLIPSelf (ICLR 2024)](https://arxiv.org/abs/2310.01403): crop CLS를 teacher로 dense feature의 region pooling을 맞춤 (encoder fine-tune). 우리는 encoder 고정 + 개념/위치 조건 register.
  - [Red Circle (ICCV 2023)](https://arxiv.org/abs/2304.06712), Alpha-CLIP: 입력에 영역 표시 → 문맥 유지. 우리는 표시가 아니라 token 배정.
- **실험 (Q5 분리):** 같은 register 구조에서 조건만 바꿈: box만 / 개념만 / 개념 + box. box만 ≈ 개념 + box → 표현이 병목. 개념만 ≪ 개념 + box → 위치 찾기가 병목. K ∈ {1, 4, 16}, target = C0 pooled (cos), 평가 R0/R2.

### I4. Register 버전 encoder 점검 — **저비용 확인**
- **무엇:** 슬라이드 실험 reader는 timm DINOv2 lvd142m(register 없음), 레포 RAEv2는 `dinov2_vitb14_reg`(register 4개). 작은 물체 ROI의 token이 high-norm artifact이면 그 자리 정보가 이미 날아가 있을 수 있음.
- **실험:** F336 vs C0 격차를 reg 버전으로 다시 측정 (학습 없음). 격차 중 artifact 몫을 분리.
- **보너스:** I1의 문맥 K/V 576개 대신 register(전역 요약) + ROI 주변 token만 참조 → 질의당 문맥 attention 비용 절감.

### I5. Crop token 수 sweep — **비용 그림의 근거**
- **무엇:** C0 / C1을 112, 168, 224, 336 px(약 64, 144, 256, 576 token)로 재인코딩. Full 쪽 sweep(F336 → F504 → F672)은 있는데 crop 쪽이 없음.
- **왜:** "물체가 몇 token이면 충분한가"의 knee가 I1/I2의 token 예산 N을 정함. F6(작은 part는 과확대로 손해)과 연결해 ROI 크기별 N 규칙도.

### 기각
- **전체 이미지 고해상도 인코딩 (Native 672 이상):** Areum이 기각. 5–16 obj에서 C0와 8pp 차이가 남고 비용이 이미지당 91 GFLOPs.
- **사후 feature upsampler ([FeatUp](https://arxiv.org/abs/2403.10516), [LoftUp](https://arxiv.org/abs/2504.14032), [JAFAR](https://arxiv.org/abs/2506.11136), [AnyUp](https://arxiv.org/abs/2510.12764)):** 11-03에서 AnyUp ×2/×8이 Bilinear보다도 낮음 → Q2의 반례로 related work에만.

---

## 4. 하나로 보기: 두 개의 손잡이

| | 질의 token 수 N | 연결 | 조건 | 질의당 비용 | 기대 |
|---|---|---|---|---|---|
| C0 (oracle) | 576 | 문맥 없음 | 위치 | 105 GFLOPs | 54.75 |
| **I1** | 576 (원본 픽셀) | 마지막 k block에서 문맥 참조 | 위치 | ≈ C0 + 문맥 attention | ≥ C0? |
| **I2** | 4–144 (학습 query) | 양 방향 | 위치 | ≈ 19 GFLOPs + N | → C1 46? |
| **I3** | 1–16 (학습 query) | 양 방향 | 개념 (+ 위치) | ≈ 19 GFLOPs + N | 위치 vs 표현 분리 |
| 한 방향 reader (T-REN류) | 임의 | 출력 위 cross-attn | 위치/개념 | 매우 쌈 | ≈ 38.5 (multi-layer와 같음) |

논문 그림: x = 질의 token 수 N (또는 질의당 GFLOPs), y = 5–16 obj. 그 위에 C0, Native F672, 한 방향 reader를 점으로.

---

## 5. 교수님 질문 → 답

| 질문 | 답 | 근거 |
|---|---|---|
| Q1 crop보다 나은 점 | 문맥 공유(crop은 바깥을 못 봄), 질의 여러 개일 때 비용, 임의 mask에 token 배치 가능 | F4, ViCrop, I1 실험 |
| Q2 bilinear로 안 되는 이유 | token은 encoder 안에서 계산되어야 함 | F2, 11-03 Exp1, upsampler 계열 |
| Q3 det/seg와 차이 | low-level 재사용은 이미 부족. Detection에서도 FPN 위에서 slicing이 크게 이김 | 11-04 D, multi-layer; [SAHI](https://arxiv.org/abs/2202.06934) (+5~7 AP 추론만) |
| Q4 (i) vs (ii) | **(ii)**: 질의 영역에 encoder 안에서 token 배정 (I1–I3) | F2, F3 |
| Q5 위치 vs 표현 | Object는 표현, part는 위치가 병목. I3로 같은 구조에서 분리 | 10-31 p6, I3 실험 |

---

## 6. 실험 순서와 선행 작업

0. **선행:** R2 reader가 import하는 `common.py`(이전 서버 `2026-10-30/paco_full_vs_crop_ft/code`)를 이 서버로 옮김. 없으면 PACO 평가 불가 (`configs/paths.yaml:39`).
1. **I1** training-free k sweep (구현 작음, 학습 없음) → 방향 상한 확인.
2. **I4** reg 버전 점검 (학습 없음, 반나절).
3. **I5** crop token sweep (학습 없음) → N 예산.
4. **I2** 양 방향 질의 register, N sweep (학습 필요).
5. **I3** 개념 / box / 개념 + box 비교 (I2 구조 재사용).

마감(11월 중순)을 고려하면 1–3은 학습이 없어 빠르게 결론을 낼 수 있고, 4–5 중 하나를 Method의 주 결과로.

---

## 7. 열린 질문
- [ ] `common.py` 등 이전 서버 코드 이전 시점
- [ ] I1이 C0보다 낮을 때 adapter 학습을 마감 전에 넣을지
- [ ] I2/I3의 target: C0 token map vs pooled 벡터 (F5는 pooled로 충분할 수 있음을 시사)
- [ ] 비용 축: GFLOPs vs latency, 질의 수 범위
- [ ] 평가 reader 고정 (R0 / R2), seed 수

## Sources
- CLIPSelf: https://arxiv.org/abs/2310.01403
- FeatUp: https://arxiv.org/abs/2403.10516 · LoftUp: https://arxiv.org/abs/2504.14032 · JAFAR: https://arxiv.org/abs/2506.11136 · AnyUp: https://arxiv.org/abs/2510.12764
- Quadformer: https://arxiv.org/abs/2304.00287
- TEVA (ICCV 2025): https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.pdf
- ViCrop: https://arxiv.org/abs/2502.17422 · HiDe: https://arxiv.org/abs/2510.00054 · FOCUS: https://arxiv.org/abs/2506.21710
- SAHI: https://arxiv.org/abs/2202.06934
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588 · Don't Need Trained Registers: https://arxiv.org/abs/2506.08010
- VPT: https://arxiv.org/abs/2203.12119
- SteerViT: https://arxiv.org/abs/2604.02327
- Alpha-CLIP: https://arxiv.org/abs/2312.03818 · Red Circle: https://arxiv.org/abs/2304.06712
- T-REN: https://arxiv.org/abs/2604.18573 · ReVA: https://arxiv.org/abs/2608.28707 · TextRegion: https://arxiv.org/abs/2505.23769 · BLIP-2: https://arxiv.org/abs/2301.12597
- GRF (의료, venue 미확인): https://arxiv.org/abs/2607.06909
