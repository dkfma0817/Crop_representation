# 2026-11-04 Direction Discussion: (ii) 방향과 첫 실험

참여: Areum Kim, Claude
이전 맥락: [2026-11-04.md](2026-11-04.md) (미팅 흐름 10-27 → 11-04, 교수님 녹취 분석 §7)

## 1. 결과 요약 (frozen DINOv2-B/14, PACO test M2, GT ROI 제공, R2 attention reader)

| 입력 | 5–16 obj token | Overall | 5–16 Obj | 출처 |
|---|---|---|---|---|
| F336 | 10 | 40.69 | 38.10 | 11-03 |
| Bilinear ×2 / AnyUp ×2 (feature 보간) | 29 | 40.56 / 40.00 | 37.41 / 37.06 | 11-03 Exp1 |
| Bilinear ×8 / ×14 | 351 / 1,033 | 41.17 / 41.14 | 38.79 / 38.07 | 11-03 App. |
| F672 (336 → 672 확대, 새 픽셀 없음) | 29 | 43.67 | 44.95 | 11-03 |
| Native F672 (원본 → 672) | 29 | 45.14 | 46.62 | 11-03 Exp2 |
| C1 crop (336 이미지에서 자름, 새 픽셀 없음) | ~380 | — | 46.06 | 11-03 Exp3 |
| C0 crop (원본 픽셀) | 382 | 45.44 | 54.75 | 11-03 |
| Detail CNN D (11-04, 원본 672 CNN + residual) | — | 42.72 | 40.44 | 11-04 |
| F336 multi-layer readout | — | 43.27 | 38.50 | 11-04 |

읽는 법:
- Feature 보간은 token을 1,000개로 늘려도 효과 0 → **token은 encoder 안에서 계산되어야 한다.**
- 원본 픽셀 효과: token 29개에서는 +1.67 (F672 → Native), token 380개에서는 +8.69 (C1 → C0) → **원본 픽셀은 질의 영역에 token이 충분히 배정될 때만 쓸모 있다.**
- Low-level reuse(D, multi-layer)는 Bilinear 대비 +2~3이지만 5–16 obj에서 C0와 14pp 이상 차이 → **교수님 (i) "low-level 재사용만으로 충분"은 기각.**
- ROI당 encoder 비용(10-31): F336 19, F672 91 (이미지당), C1/C0 105 (질의당) GFLOPs.

## 2. 문헌에서 얻은 것

| 문헌 | 핵심 | 우리에게 주는 의미 |
|---|---|---|
| [CLIPSelf (ICLR 2024)](https://arxiv.org/abs/2310.01403) | ViT 자신의 crop embedding을 teacher로, dense feature에서 pool한 region 표현을 맞추도록 self-distill. Dense map의 각 위치가 전역 이미지를 인코딩하는 문제를 지적 | 아름이 10-27에 언급한 "crop을 teacher로 쓰는 기존 연구". crop의 **CLS 하나**를 맞추는 것이고, 우리는 **crop token map 전체(full-length sequence)** 를 얻으려는 것이 차이. Related work에서 명확히 구분 필요 |
| [FeatUp](https://liner.com/review/featup-a-modelagnostic-framework-for-features-at-any-resolution), [LoftUp](https://arxiv.org/abs/2504.14032), [JAFAR](https://arxiv.org/pdf/2510.12764), [AnyUp](https://arxiv.org/abs/2510.12764) | Task-agnostic 사후 feature upsampler 계열 | 11-03에서 AnyUp ×2/×8이 Bilinear보다도 못함. **"사후 upsampling은 세부 판별 정보를 만들지 못한다"** 를 이 계열 전체에 대한 반례로 쓸 수 있음 (교수님 Q2 답) |
| [SAHI (ICIP 2022)](https://arxiv.org/abs/2202.06934) | FPN을 이미 쓰는 detector(FCOS, VFNet, TOOD)에 slicing 추론만 해도 small-object AP +5~7, fine-tuning 포함 +12~15 | 교수님 Q3 "detection에서도 crop이 효과 있나?"에 대한 **문헌 근거**: low-level 융합(FPN)이 있어도 잘라서 다시 보는 게 크게 이김. 실험 없이 인용 가능 |
| [ViCrop (2502.17422)](https://arxiv.org/abs/2502.17422) | MLLM의 작은 물체 실패는 위치 찾기가 아니라 **지각(perception) 한계**. Attention으로 위치 찾고 crop하면 training-free로 개선. 단 전역 시야가 필요한 질문에서 실패 | 우리 Motivation(PerceptionBench 26/27 위치는 찾음)과 같은 결론. 그리고 "crop은 전역 문맥을 잃는다"는 약점 = 우리 방법의 차별점 |
| [HiDe (2510.00054)](https://arxiv.org/abs/2510.00054) | **전체 이미지 확대만으로는 MLLM이 나아지지 않고, crop이 핵심**. 배경 간섭을 원인으로 봄 | 아름의 가정("crop 표현 자체가 이득")과 같은 방향의 외부 근거. Native F672가 5–16 obj에서 C0에 못 미치는 우리 결과와도 맞음 |
| [FOCUS (2506.21710)](https://arxiv.org/abs/2506.21710) | Training-free crop 탐색을 효율화 (ZoomEye 대비 3~6배) | 질의 수가 늘면 crop 반복 비용이 문제라는 동기의 근거 |
| [Quadformer (CVPRW 2023)](https://arxiv.org/abs/2304.00287) | Saliency 높은 영역에만 작은 patch, 나머지는 큰 patch. 전역 attention 유지, patch 중심 기반 2D pos-embed | "한 forward 안에서 영역별로 token 밀도를 다르게"의 선행 연구. 우리는 **질의 시점에** 영역을 정하고 **문맥은 재사용**한다는 점이 다름 |
| [TEVA (ICCV 2025)](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.pdf) | VLM에서 핵심 영역만 고해상도 patch로, 전역은 저해상도 patch로 유지 | 가장 가까운 경쟁 설계. 차이를 실험으로 보여야 함 (문맥 token을 encoder 안에서 공유하는지, 임의 mask 지원, frozen encoder 그대로인지) |
| [GRF (2607.06909, 의료)](https://arxiv.org/abs/2607.06909) | 고해상도 patch feature를 cross-attention으로 대응하는 전역 token에 주입 (token 수 고정) | 반대 방향 설계(전역 token을 고침). 우리는 질의 영역을 crop처럼 **길게** 만드는 쪽. venue 미확인 |

## 3. 방향 결정

### 3-1. 질의 영역 token의 입력 = 최근 미팅(11-03 story, 11-04 D) 기준
- **원본 픽셀을 질의 영역에서만 쓰고, 전역 문맥은 공유한다** (11-03 Proposed Method 그대로).
- 다만 "결국 해상도 아니냐"는 질문에 대비해, 같은 구조에서 **질의 영역 입력만 C1식(336 이미지에서 자른 픽셀)으로 바꾼 대조군**을 같이 둔다 (11-03 Exp3의 C1 vs C0 설계를 그대로 재사용). 이 대조군이 "crop 표현(token 배정) 자체의 이득"을 분리해서 보여준다.

### 3-2. 첫 실험: training-free 공동 forward

**질문:** frozen encoder가 질의 영역을 crop처럼 촘촘한 token으로 처리하면서 이미 계산된 전역 token을 문맥으로 참조하면, 학습 없이도 C0만큼(또는 그 이상) 나오는가?

**설정**
- Encoder: frozen DINOv2-B/14 (reader 코드와 같은 timm lvd142m), full image 336 → 24×24 token. 층별 full-image token을 한 번 계산해 저장.
- 질의 영역: 원본 픽셀 crop을 336 letterbox → 576 token (C0와 같은 입력), crop canvas 기준 pos-embed.
- 각 block에서 ROI token은 [ROI token + 같은 층의 full-image token]에 attention. Full-image token은 ROI를 보지 않음 → 문맥은 질의와 무관하게 재사용.
- 문맥 주입 깊이 k ∈ {0, 3, 6, 12} (마지막 k block에서만 문맥 참조). **k = 0은 C0와 정확히 같아야 함** → 구현 검증용.
- 대조군: 같은 설정에서 ROI 입력을 C1 픽셀로.
- Read-out: R2 attention reader (seed 3개) + R0 mean.
- 보고: Overall, 5–16 Obj, 1–4 Part, Color / Pattern, 질의당 GFLOPs.

**판정**
- k > 0이 C0(5–16 obj 54.75) 이상 → 문맥 공유가 도움. Training-free로 (ii)의 상한 확보, 이후 token 수를 줄이는 쪽으로.
- k > 0이 C0보다 낮음 → 학습 안 된 encoder에 두 좌표계 token을 섞은 분포 이탈일 가능성. 문맥 attention에만 작은 adapter/LoRA를 붙여 학습하는 다음 단계로. 이것도 교수님 (ii)의 "learnable" 부분에 해당.

**위험 / 미지수**
- 두 token 집합이 서로 다른 좌표계(crop canvas vs full image)의 pos-embed를 갖는다. Frozen 모델에서 처음 보는 분포라 성능이 오히려 떨어질 수 있다. 11-03 context removal(attention 차단 시 −5)은 full-image 안에서 문맥이 도움이 된다는 것이지, 이 설정에서도 도움이 된다는 보장은 아니다.
- 비용은 질의당 C0와 거의 같다 (576 token × 12 block + 문맥 attention). 비용 이득은 다음 단계(token 수 줄이기)에서 나와야 한다. 이 실험은 **상한 확인용**.
- 차별점: TEVA, Quadformer와 무엇이 다른지 (frozen encoder 그대로, 질의 시점 영역 지정, 문맥 재사용, 임의 mask에 token 배치 가능)를 결과와 함께 정리해야 한다.

**실행 가능성 (이 서버)**
- Reader 코드(`third_party/paco_region_reader/code/common_rr.py`)가 `common.py`(이전 서버 `2026-10-30/paco_full_vs_crop_ft/code`)를 import하는데 이 서버에 없다 (`configs/paths.yaml:39`). **먼저 이 파일을 옮겨와야 R2 reader로 평가 가능.**
- Encoder 수정은 timm ViT block의 attention에 KV 연결만 하면 되므로 구현은 작다.

## 4. 교수님 질문에 대한 답 정리 (다음 미팅용)

| 교수님 질문 | 답 | 근거 |
|---|---|---|
| Q2 bilinear로 늘리면 왜 안 되나 | Token은 encoder 안에서 계산돼야 함 | 11-03 Exp1, App.(×8, ×14, AnyUp); FeatUp 계열 |
| Q3 det/seg와 뭐가 다른가 | Low-level 재사용은 이미 부족하다고 확인됨. Detection에서도 slicing이 FPN 위에서 크게 이김 | 11-04 D, multi-layer; SAHI |
| Q4 (i) vs (ii) | **(ii)**: 질의 영역에 token을 배정 | 위 두 줄 + 원본 픽셀 효과의 token 의존성(+1.67 vs +8.69) |
| Q1 crop보다 나은 게 뭐냐 | 문맥 공유 (crop은 바깥을 못 봄), 다중 질의 비용, 임의 mask | 11-03 context removal; ViCrop의 전역 시야 실패 사례; **3-2 실험으로 확인 예정** |

## 5. 결정 사항
- 질의 영역 입력: 최근 미팅(11-03 story) 기준 원본 픽셀 + 같은 구조의 C1 대조군.
- 첫 실험: training-free 공동 forward (§3-2). 선행 작업: `common.py` 이전.
- 로그 위치: 이 파일 (discussion/2026-11-04/2026-11-04_direction.md).

## 6. 주장 정리 (Q: 사람은 중요한 곳만 본다 → crop 표현을 두고 decoder만 배우면 되지 않나?)

- **사람 비유를 정확히 쓰면 foveation이다.** 주변시(저해상도, 전체)로 문맥과 "어디를 볼지"를 잡고, 눈을 옮겨 중심와(fovea)로 그 지점을 **새로 고해상도로 본다**. 주변시 신호만 가지고 뇌가 무늬를 복원하지는 않는다.
  - 주변시 = full image 한 번 인코딩 (F336)
  - 눈 옮기기(saccade) = 텍스트/attention으로 위치 찾기 (SteerViT, ViCrop)
  - 중심와 = 질의 영역에 촘촘한 token을 배정해 다시 보기 (crop 재인코딩)
  - 사람은 중심와로 볼 때도 주변 문맥을 잃지 않는다 ← **crop 재인코딩에 없는 것**
- **"full-image feature + decoder만"은 주변시에서 중심와를 복원하려는 것**이라 비유와 반대이고, 데이터로도 이미 막혀 있다: 10-27 ~ 11-04 decoder 5종(구조, ImageNet, feature/visual/semantic loss, encoder 공동학습)이 R2 기준 Bilinear +1 이내 (40.66 → 최고 41.39), 5–16 obj 37.3 → 38.4 vs C0 55.1. 같은 픽셀 C1(46.1)에도 못 미치므로, 문제는 픽셀이 아니라 **한 번 인코딩할 때 물체에 token이 10개밖에 배정되지 않는 것**.
- **"encoder는 고정하고 가벼운 모듈만 배운다"는 유지 가능.** 단, 그 모듈의 입력에 질의 영역 token(중심와)이 들어가야 한다. 11-04 D, §3-2 공동 forward + adapter가 이 형태.
- **주장 (초안):** 작은 물체의 세부 속성(색, 형태, 무늬)은 한 번 인코딩한 전역 feature로는 사후에 어떻게 읽거나 복원해도 회복되지 않는다. 필요한 것은 질의 영역에 encoder 안에서 token을 배정하는 것(foveation)이다. 전역 문맥을 재사용하는 foveated encoding으로, 독립 crop보다 정확하고(문맥) 질의가 많을 때 싸다(문맥 공유).
  - Finding 1: crop 표현은 같은 supervision에서도 더 잘 학습/일반화된다 (FT: Full 40.57 / C1 47.27 / C0 49.41)
  - Finding 2: 이득의 원인은 token 배정. 사후 upsampling·decoder·low-level 재사용으로는 안 됨 (11-03 Exp1, 11-04 A–D)
  - Finding 3: 원본 픽셀은 token이 충분할 때만 쓸모 (+1.67 vs +8.69), 문맥 제거는 손해
  - Method: 문맥 공유 foveated encoding (§3-2 → adapter)

## 7. Register token과의 연결 (Q: semantic을 잘 찾는 연구에 registration이 있던데?)

- 해석: [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) (Darcet et al., ICLR 2024) 계열로 이해함. 큰 ViT(DINOv2, CLIP 등)는 정보가 적은 배경 patch 몇 개를 **전역 정보 저장소**로 재활용해서 norm이 매우 큰 artifact token이 생기고, 그 patch의 local 정보는 사라진다. 학습 가능한 register token을 sequence에 추가하면 그 역할을 register가 가져가서 dense feature map과 attention이 깨끗해진다. 후속 [Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010) (NeurIPS 2025)는 high-norm을 만드는 소수 neuron을 찾아 학습 없이 추가 token 하나로 옮기는 test-time register를 제안.
- 우리와 닿는 지점:
  1. **교수님 (ii) "learnable token이 빈자리를 채운다"의 구조적 선례.** Register는 픽셀에 대응하지 않는 token을 sequence에 넣어 encoder 안에서 역할을 맡긴다. 우리 질의 영역 token도 "질의마다 붙는 token"이라는 점에서 같은 메커니즘이다. 차이: register는 전역 정보를 빼내 주는 역할, 우리는 질의 영역의 세부를 담는 역할.
  2. **"full-image token에는 전역 정보가 섞인다"의 근거.** 한 번 인코딩한 token에서 decoder가 세부를 못 꺼내는 이유(A 경로의 천장)를 설명하는 데 인용 가능. CLIPSelf도 같은 지적.
  3. **바로 확인할 것: 우리 encoder가 register 버전인가.** Reader 쪽은 timm DINOv2 lvd142m(register 없음, 루트 README 기준), RAEv2 쪽은 `dinov2_vitb14_reg`(register 4개, `src/rae/common.py:4`). 작은 물체 ROI의 token이 artifact token이면 그 자리 정보가 날아가 있을 수 있다. F336 vs C0 격차를 reg 버전으로 한 번 재보면 "격차 중 artifact 몫"을 분리할 수 있다 (학습 불필요, 저비용).
  4. **비용 절감 아이디어.** 문맥 K/V 576개 대신 register(전역 요약) + 질의 영역 주변 token만 참조하게 하면 질의당 문맥 attention 비용을 줄일 수 있다. Register가 전역 정보를 모은다는 성질을 그대로 이용.

## 8. Register 토큰만 decoder로 배우면 crop을 배울 수 있나?

- **register를 그대로 꺼내 decoder에 넣는 방식: 어렵다.**
  - Register는 전역 요약을 모으는 자리라서, 특정 작은 물체의 색·무늬 같은 local 세부를 담지 않는다 (그래서 patch map이 깨끗해지는 것).
  - 한 번 인코딩한 출력을 사후에 읽는 것 = A 경로. 11-04 F336 multi-layer readout(여러 층 token을 attention으로 읽음)이 5–16 obj 38.50으로 F336 38.10과 거의 같다. register를 더해도 같은 천장일 가능성이 크다.
- **형태를 바꾸면 가능성 있음: "질의 register"를 encoder 안에 넣기.**
  - ROI 위치로 조건화한 학습 가능한 token N개를 full-image forward의 sequence에 추가 (VPT-deep처럼 층마다, 또는 입력에서). Encoder가 forward 중에 그 token에 물체 정보를 써 넣게 한다. 학습 대상은 이 token과 작은 adapter뿐, encoder는 frozen.
  - 근거: 336 이미지의 layer-0 patch embedding에는 픽셀 정보가 거의 그대로 있다(14×14×3 = 588 → 768). F672가 새 픽셀 없이 회복한 것은 정보가 아니라 **처리할 token 수**였다. 따라서 질의 token이 encoder block의 처리를 받으면 C1(46.06)까지는 원리적으로 갈 수 있다.
  - 두 가지 연결 방식:
    - 한 방향 (질의 token → patch만 참조, patch는 질의를 안 봄): patch를 캐시해 질의당 매우 쌈. 단 사실상 "frozen block을 쓰는 multi-layer reader"라 38.5 천장에 걸릴 위험.
    - 양 방향 (patch도 질의 token을 봄): 질의마다 full forward 1회 (F336 기준 약 19 GFLOPs, C0 105보다 쌈). Encoder가 그 물체를 위해 다시 처리하므로 더 유망.
  - **§3-2 방법과 하나로 묶임:** 질의 token 수 N이 손잡이. N = 576이고 원본 픽셀로 초기화하면 §3-2의 C, N을 줄이면 register식 경량 버전. N ∈ {4, 16, 64, 144, 576} sweep이 곧 비용–성능 곡선.
- 실험 순서 제안: (1) 양 방향, 질의 token N=16을 ROI 위치 pos-embed + 학습 query로 → C1 46.06에 얼마나 가는지, (2) N sweep, (3) 질의 token을 ROI 원본 픽셀 patch로 초기화해 C0 방향으로.

## 9. "새를 보고 있으면 '새' register를 추가해서 배우면 되지 않나?"

- **된다고 본다. 단 encoder 안에서 (양 방향) 해야 한다.** §8의 질의 register를 "위치(box)"가 아니라 "개념(새)"으로 조건화한 것. Forward 중에 encoder가 새에 관한 정보를 그 token에 모으고, 학습은 그 token(+작은 adapter)만.
- **의외로 중요한 숫자: 속성 맞추기에는 crop의 map 전체가 아니라 벡터 하나로도 충분할 수 있다.** 11-03 App.(Reader Robustness): C0를 mean pooling + linear(R0)로만 읽어도 5–16 obj 47.51 (F336 R0 31.59). 즉 target을 24×24 map 대신 **C0의 pooled 벡터**로 잡으면 "새 token 1~몇 개"로 학습 목표가 맞는다. (CLIPSelf가 crop CLS를 target으로 쓴 것과 같은 선택. 우리는 frozen encoder + register token으로 한다는 점이 다름.)
- **위험:**
  - 같은 종류가 여러 개일 때 어느 새인지 모름: SteerViT heatmap 실패율 단일 7.7% vs 다중 36% (10-31 p6).
  - Part 질의("새의 부리")는 부모 물체와 거의 같은 곳을 봄 (heatmap cos 0.89).
  - → 개념만이 아니라 개념 + 위치(box나 점)를 같이 주는 조건이 필요할 수 있다.
- **이 실험이 교수님 Q5(위치 찾기 vs 표현)에 바로 답한다:** 같은 register 구조에서 조건만 바꿔 비교
  - box만 / 개념(text)만 / 개념 + box
  - box만 ≈ 개념+box → 표현이 병목, 개념만 ≪ 개념+box → 위치 찾기가 병목.
- **설정 초안:** frozen DINOv2-B/14, F336 forward에 register K개(K ∈ {1, 4, 16}) 추가, 양 방향 attention. 조건 = category 이름의 text embedding (SigLIP2 text) 그리고/또는 box embedding. Target = C0 pooled 벡터 (stop-grad), loss = cos (+ 선택적으로 C0 token map). 평가 = PACO R0/R2. 비용 = 질의당 F336 forward 1회(약 19 GFLOPs) + K token.

## 10. 열린 질문
- [ ] `common.py` 등 이전 서버 코드 이전 시점
- [ ] Training-free 결과가 C0보다 낮을 때, adapter 학습을 마감 전에 넣을지
- [ ] 비용 축 x = 질의 수, 범위(1, 4, 16, 64?)
