# RAEv2 재구성 시각화 분석 (Teacher / Bilinear / A / B / C, 12k)

작성 2026-11-04 19:55 KST. **분석과 시각화만** 했다. 추가 학습, 새 loss, 새 모델은 없다.
- 기존 12k checkpoint(`perception_cache/runs/{A,B,C}/ckpt_12000.pt`)를 그대로 썼다.
- 공식 RAEv2 dinov2b-k1 encoder–decoder를 그대로 썼다.
- 전처리도 동일하다: 외부 256 letterbox, 내부 224, 같은 ROI geometry, stats, padding mask.
- 기존 결과 파일은 수정하지 않았다.

코드:
- 그림: `code/viz_analysis.py`
- hval teacher 정정: `code/hval_teacher_fullpad.py`

산출물:
- `figures/reconstruction_comparison.png`
- `figures/high_frequency_comparison.png`
- `figures/reconstruction_error_maps.png`
- `figures/reconstruction_error_maps_content.png` (content 영역으로 확대한 버전)
- `results/reconstruction_by_size.csv`
- `results/visualization_tiles.csv` (그림 타일별 지표)
- `results/visualization_selection.json` (선정 규칙과 ROI ID)

## 0. 데이터에 관한 사전 확인 (요청과 다른 부분)

- **hval에는 part와 속성 라벨이 없다.** hval은 rae_crop_pilot의 head_val로, COCO 객체 GT box 3,194개뿐이다.
  - 그래서 "작은 Part"와 "Pattern/Material" 그룹은 hval에서 뽑을 수 없다.
  - 이 두 그룹은 기존 재구성 평가에 함께 쓴 **PACO test-primary** ROI(`recon_rae.py`의 pacot 세트와 같은 manifest)에서 뽑았다. test-primary 이미지는 모든 decoder 학습 pool 밖에 있다.
- **크기 bin 정의:**
  - hval의 `n_in`은 기존 F336 grid(336/14, 24×24) 기준임을 확인했다.
  - hval의 **D1-4는 part가 아니라 "아주 작은 객체"**다.
  - 실제 Part D1-4(PACO level = part, F336 n_in 1–4)는 PACO test-primary 행으로 따로 보고한다.
- **기존 hval Teacher 수치 정정:**
  - 결함: 기존 `recon_rae.py`는 hval teacher latent를 content-cell 전용 캐시에서 읽었다. 이 캐시는 padding cell을 0으로 채우므로, letterbox padding이 있는 ROI에서는 실제 teacher와 다른 latent가 decoder에 들어갔다.
  - 영향 없는 부분: Bilinear/A/B/C는 full-image 캐시가 완전해서 영향이 없다(재계산 차이 L1 ≤ 0.0002). PACO 세트와 학습도 영향이 없다.
  - 정정 방법: hval teacher만 crop을 새로 인코딩해서 다시 계산했다(`results/recon/hval_teacher_fullpad.csv`).
  - 정정값: L1 0.082 → **0.076**, dE 7.33 → **5.76**, hp_corr 0.318 → **0.343**. 기존 README의 hval Teacher 행이 이 결함의 영향을 받았다. 결론은 바뀌지 않는다.

## 1. ROI 선정 규칙 (재구성 결과를 보기 전에 고정, 그룹마다 `numpy.default_rng(0)`)

| 그룹 | 소스 | 조건 | 우선순위 목록 → 각 항목에서 무작위 1개, 앞에서부터 4개 | 선정된 ROI |
|---|---|---|---|---|
| G1 작은 Object | hval | F336 n_in 5–16 | clock, cell phone, laptop, bicycle, tv, keyboard, dog, book | clock 417729_1 (n_in 9), cell phone 441488_0 (6), laptop 472581_3 (10), bicycle 139390_1 (9) |
| G2 작은 Part | PACO test-primary | level=part, F336 n_in 1–4 | clock, cellular_telephone, laptop_computer, bicycle, dog, shoe, television_set, mug | clock:hand row38574 (2), cellular_telephone:bezel row33422 (4), laptop_computer:touchpad row31928 (2), bicycle:seat_stay row21729 (1) |
| G3 Pattern 디테일 | PACO test-primary | level=object, F336 n_in 5–16, 해당 pattern 라벨 positive | text, striped, logo, woven, floral, checkered | box/text row10511 (6), chair/striped row4948 (9), mouse/logo row4803 (9), basket/woven·rattan row1592 (16) |

- 특정 방법이 좋아 보이는 사례를 고르지 않았다. 선정에는 카테고리, 크기, 라벨만 썼다.
- G3에서는 앞의 네 라벨로 4개가 채워져 floral, checkered는 쓰지 않았다.
- material 라벨은 `visualization_selection.json`의 attrs에 있다(paper, fabric, plastic, rattan, metal 등).

## 2. 정량 결과 (`results/reconstruction_by_size.csv`, 기존 `per_roi`를 재집계하고 hval Teacher만 정정)

| 세트 / 크기 | 방법 | RGB L1 | LPIPS | 평균 색 dE | hp_corr | teacher latent cos |
|---|---|---|---|---|---|---|
| hval 전체 (3,194) | Teacher | 0.076 | 0.179 | 5.76 | **0.343** | 1 |
| | Bilinear | 0.182 | 0.408 | 15.2 | 0.005 | 0.617 |
| | A | 0.197 | 0.432 | 18.8 | 0.001 | 0.710 |
| | B | **0.149** | **0.362** | **10.2** | 0.006 | 0.227 |
| | C | 0.162 | 0.393 | 11.6 | 0.003 | 0.709 |
| hval D5-16 객체 (1,042) | Teacher | 0.071 | 0.174 | 5.68 | **0.408** | 1 |
| | Bilinear | 0.203 | 0.411 | 17.7 | 0.000 | 0.546 |
| | A | 0.197 | 0.422 | 18.8 | 0.000 | 0.665 |
| | B | 0.160 | 0.363 | 10.5 | 0.003 | 0.231 |
| | C | 0.173 | 0.394 | 12.2 | 0.000 | 0.665 |
| hval D1-4 아주 작은 객체 (565) | Teacher | 0.064 | 0.186 | 6.47 | **0.485** | 1 |
| | Bilinear / A / B / C | 0.233 / 0.202 / 0.178 / 0.182 | 0.482 / 0.442 / 0.396 / 0.418 | 23.4 / 19.7 / 14.3 / 14.9 | 0.001 / −0.001 / 0.001 / 0.000 | 0.424 / 0.668 / 0.283 / 0.667 |
| hval D17+ 객체 (1,587) | Teacher | 0.084 | 0.180 | 5.57 | **0.250** | 1 |
| | Bilinear / A / B / C | 0.151 / 0.194 / 0.131 / 0.149 | 0.379 / 0.435 / 0.350 / 0.383 | 10.7 / 18.5 / 8.6 / 10.1 | 0.010 / 0.003 / 0.011 / 0.006 | 0.732 / 0.754 / 0.204 / 0.753 |
| PACO **D1-4 Part** (1,000) | Teacher | 0.066 | 0.160 | 6.91 | **0.393** | 1 |
| | Bilinear / A / B / C | 0.207 / 0.198 / 0.169 / 0.180 | 0.416 / 0.361 / 0.331 / 0.339 | 22.1 / 20.1 / 16.0 / 17.3 | 0.001 / −0.001 / 0.000 / −0.001 | 0.374 / 0.679 / 0.287 / 0.679 |
| PACO **D5-16 Object** (1,000) | Teacher | 0.067 | 0.180 | 6.01 | **0.443** | 1 |
| | Bilinear / A / B / C | 0.192 / 0.201 / 0.158 / 0.169 | 0.431 / 0.449 / 0.388 / 0.416 | 17.6 / 19.4 / 11.8 / 13.0 | 0.000 / 0.000 / 0.000 / 0.000 | 0.517 / 0.637 / 0.240 / 0.637 |
| PACO D17+ 객체 (1,000) | Teacher | 0.078 | 0.184 | 5.75 | **0.292** | 1 |
| | Bilinear / A / B / C | 0.150 / 0.191 / 0.131 / 0.146 | 0.396 / 0.457 / 0.371 / 0.406 | 11.6 / 18.3 / 9.2 / 10.7 | 0.007 / 0.001 / 0.009 / 0.003 | 0.698 / 0.727 / 0.224 / 0.726 |

- **Teacher hp_corr는 작은 ROI에서 오히려 높다.** 크기가 작아질수록 0.25–0.29 → 0.41–0.44 → 0.39–0.49로 올라간다.
  - 작은 ROI의 GT crop은 몇십 픽셀을 256으로 확대한 흐린 이미지라, 그 "디테일"은 주로 굵은 윤곽이다.
  - 큰 객체의 GT crop은 16×16 latent와 decoder 해상도를 넘는 미세 텍스처를 담고 있다.
- **A/B/C의 hp_corr는 모든 크기에서 −0.001 ~ 0.011이다.** Teacher가 복원하는 GT 대응 디테일 중 사실상 0%에 접근했다. Bilinear도 같다.

## 3. Figure별 관찰

### Figure 1 (reconstruction_comparison.png)
- **Teacher:** 시계 링·숫자판 막대·바늘, 휴대폰 외곽·카메라 렌즈·버튼, 노트북 화면 테두리, 자전거 프레임 윤곽, 상자 라벨 글자 덩어리, 마우스 로고, 바구니 엮음 무늬를 GT와 거의 같게 재현한다.
  - 예외는 아주 흐린 작은 part다. cellular_telephone:bezel(F336 n_in 4)에서 teacher L1 0.179로, A(0.150)보다 나쁘다. 색 이동 때문이다.
- **Bilinear:** 평균색 위에 decoder의 격자형 아티팩트가 깔린다. 구조는 없다.
- **A:** 거의 단색의 평균색 패치다. 휴대폰은 회색 사각형, 시계는 회갈색 텍스처다. content 영역(letterbox 형태) 밖 정보는 없다.
- **B:** 색과 아주 거친 명암 배치가 맞는다. 마우스의 어두운 타원 덩어리, 노트북의 어두운 세로 띠, 상자 중앙의 밝은 영역이 그 예다. 경계, 글자, 로고, 무늬는 없다.
- **C:** A와 B의 중간이다. A보다 색은 GT에 가깝지만 형태 정보는 A와 거의 같다.

### Figure 2 (high_frequency_comparison.png)
- **High-pass 정의:** `rcp.hp_corr`와 동일하게 계산했다.
  - 휘도 Y = 0.299R + 0.587G + 0.114B에서 HP = Y − box7(Y)를 구한다(7×7 평균, stride 1, pad 3, count_include_pad=False).
  - 3 px 침식한 content 영역만 표시한다.
  - 모든 방법에 고정 스케일 ±0.115를 썼다(선정 ROI GT의 |HP| 99 백분위).
- **Teacher:** GT의 링, 외곽선, 글자, 로고 윤곽, 엮음 패턴이 같은 위치에 나타난다(tile hp_corr 0.21–0.63).
- **Bilinear:** 고주파 에너지는 오히려 크다(hf_ratio 최대 3.2). 하지만 decoder 격자 무늬라 GT와 상관이 없다(hp_corr ≈ 0). **에너지가 곧 디테일은 아님**을 보여 준다.
- **A:** 거의 평평하다(hf 0.08–0.5).
- **B:** 약한 주기 텍스처가 있고, 일부는 GT를 닮은 거친 윤곽이다(마우스 외곽 타원, 시계의 둥근 구조). 위치나 크기가 맞지 않아 hp_corr는 ≈ 0 또는 음수다.
- **C:** decoder 고유의 주기 격자만 보인다.
- **hp_corr 지표와 시각 판단이 일치한다.** 눈으로 GT와 같은 경계가 보이는 것은 teacher뿐이고, 나머지는 tile hp_corr가 모두 −0.05 ~ 0.04이다.

### Figure 3 (reconstruction_error_maps*.png; 고정 스케일 [0, 0.5], padding은 회색이며 지표에서 제외)
- **Teacher:** 오차가 낮고 고르게 퍼져 있다. 남은 오차는 미세 텍스처와 하이라이트 위치에 있다.
- **A/B/C/Bilinear:** 오차 지도가 GT의 구조(시계 링, 휴대폰 렌즈, 상자 라벨, 마우스 로고와 하이라이트, 바구니 띠)를 그대로 그린다. 출력이 구조 없는 매끈한 색면이라는 뜻이다.
- **B:** 평균색이 맞아 넓은 균일 영역의 오차는 작다. 예를 들어 마우스 몸통, 노트북 touchpad(L1 0.041로 teacher 0.103보다 낮음)가 그렇다. 하지만 경계와 하이라이트의 오차는 A/C와 같은 수준이다.
- **C:** A 대비 넓은 영역의 색 오차가 줄었다. 예를 들어 바구니는 0.312 → 0.246이다. 구조가 있는 위치의 오차 패턴은 A와 같다.

## 4. 질문별 정리

**1. Teacher는 실제로 어떤 디테일을 복원하는가?**
- 객체 경계, 내부 윤곽, 글자와 로고의 형태, 반복 무늬(엮음, 줄무늬), 작은 부품의 위치를 GT와 같은 위치에 복원한다.
  - hp_corr: 작은 객체 0.41–0.49, part 0.39, 큰 객체 0.25–0.29.
  - 평균 색 오차 dE 5.6–6.9.
- 한계도 있다.
  - 매우 작고 흐린 crop에서는 색 이동이 생긴다.
  - 큰 객체의 미세 텍스처는 16×16 latent 해상도에서 뭉개진다. 그래서 큰 객체일수록 hp_corr가 낮다.
- 결론: **decoder는 "올바른 latent"만 주어지면 작은 객체와 part의 구조를 충분히 그릴 수 있다.**

**2. A/B/C는 어떤 정보를 복원하고 무엇을 놓치는가?**
- **복원하는 것:** ROI의 평균색(B > C > A), content 영역의 범위(letterbox 형태), 아주 거친 명암 배치(주로 B).
- **놓치는 것:** 경계, 윤곽, 글자, 로고, 패턴, 작은 부품. 세 방법 모두 모든 크기 bin에서 hp_corr ≈ 0이다.
- A는 teacher latent 방향(cos 0.71)은 가장 잘 맞추지만, 공간 분산이 teacher의 13%뿐이다(svar 0.128). 그래서 decoder가 평균적인 색면으로 그린다.

**3. B는 왜 RGB 지표는 좋은데 속성 판별이 낮을 수 있는가?**
- B의 latent는 teacher latent와 거의 직교하는 영역에 있다(cos 0.20–0.34). 이 latent는 decoder를 거치면 평균색과 거친 명암을 잘 맞추므로 L1, LPIPS, dE가 가장 좋다.
- L1, LPIPS, dE는 넓은 매끈한 영역의 색 일치에 크게 보상한다. 반면 경계와 패턴의 부재에는 크게 벌하지 않는다. 이는 Figure 3에서 B의 오차가 구조 위치에 집중된 것에서 확인된다.
- PACO reader는 DINOv2 latent 공간의 semantic, material, pattern 방향을 읽는다. B는 그 공간을 벗어나 decoder 전용의 "색 코드"로 latent를 쓴 것으로 보인다. 그래서 color는 유지되지만(PACO color −0.2) material −5.3, pattern −9.5, 큰 객체 −7.8이 떨어진다.
- 결론: **Pixel-only의 RGB 개선은 crop latent 복원이 아니다.** 여러 latent가 비슷한 RGB로 decode되는 decoder의 다대일 성질을 이용한 해일 가능성이 높다. 이 마지막 부분은 해석이다.

**4. C는 A 대비 색뿐 아니라 경계와 패턴도 개선하는가?**
- **색만 개선한다.** hval dE는 18.8 → 11.6, L1은 0.197 → 0.162다.
- 경계와 패턴은 개선하지 않는다. 모든 bin에서 hp_corr가 0.000–0.006으로 A와 같다. Figure 2의 C high-pass는 decoder 격자뿐이고, Figure 3의 구조 위치 오차도 A와 같다.
- latent 지표(cos 0.709, svar 0.128)도 A와 동일하다. pixel 항은 feature 해에 직교하는 "색 보정" 방향으로만 작용했다(calibration에서 gradient cos 0.006).
- PACO에서 C − A가 작게 나온 것(R2 +0.6, R0 +0.1)과도 일치한다.

**5. Information Bottleneck과 Decoder Capacity Bottleneck 중 어느 쪽을 더 지지하는가? (시각화만으로 확정할 수 없음)**
- **현재 증거는 Information Bottleneck 쪽을 더 지지한다.**
  - (a) 같은 frozen decoder가 teacher latent로는 작은 ROI의 구조를 복원한다(hp_corr 0.39–0.49). 그래서 "decoder가 그런 디테일을 그릴 능력이 없다"는 해석은 맞지 않는다.
  - (b) decoder를 통해 RGB를 직접 최적화한 B조차 GT 대응 디테일을 전혀 만들지 못한다(hp_corr ≈ 0). 학습 신호가 decoder를 지나 들어와도, 입력(full-image 16×16 ROI token; D5-16 객체는 RAE 16×16 grid에서 2–16 token, 중앙값 6)에 그 디테일이 없으면 만들 수 없다는 상황과 맞는다.
  - (c) Bilinear도 같은 수준(≈ 0)이다. Crop−Full PACO 격차(D5-16 object +20)는 재인코딩으로만 정보가 들어온다는 것과 맞는다.
- **확정할 수 없는 이유와 남는 대안:**
  - (i) "decoder capacity"가 아니라 **mapping(CoordMLP) capacity** 병목일 수도 있다. CoordMLP는 10M 크기에 k=4 국소 MLP이고, teacher 공간분산의 13%만 재현한다. 다만 256-ROI 과적합에서도 feature cos는 0.83에서 멈췄다.
  - (ii) 작은 ROI의 GT crop 자체가 저해상도를 확대한 이미지다. 그래서 "복원할 디테일"의 상한이 이미 낮다.
  - (iii) decoder가 off-manifold latent에서 디테일을 그리지 못하는 것과 입력에 정보가 없는 것이 함께 작용할 수 있다.
- 이를 구분하려면 입력 정보량을 직접 재는 실험이 필요하다. 예: full-image ROI token만으로 GT crop latent를 예측하는 상한을 훨씬 큰 mapping으로 확인하거나, crop 고주파 성분의 선형 예측 가능성을 측정하는 것. 이번 범위(분석 전용)에서는 하지 않았다.

## 5. 추가 확인: "C가 형태를 전혀 못 그리는 것은 학습 실패인가?" (2026-11-04 20:10 KST)
- **큰 ROI에서는 형태를 그린다.** 같은 파이프라인으로 hval 큰 객체 6개를 다시 그렸다(F336 n_in ≥ 17, rng(0), `code/viz_large.py`, `figures/reconstruction_comparison_large_rois.png`).
  - C는 얼룩말 줄무늬, 전자레인지 문과 패널, 고양이 앞발, 비행기 형태와 색, 사람 실루엣을 알아볼 수 있게 복원한다.
  - 6개 평균: C의 L1은 0.144로 Bilinear 0.149, A 0.177보다 낮고, dE도 8.7로 Bilinear 10.4, A 11.9보다 낮다. B는 형태가 가장 선명하다(전자레인지, 비행기).
- **학습 실패 정황은 없다.** COCO dev(held-out)와 train(11.5–12k)의 손실이 거의 같다.
  - C: feat 0.769 vs 0.765, L1 0.149 vs 0.148.
  - B: L1 0.135 vs 0.122. 과적합도 발산도 없다.
  - 256-ROI 과적합에서는 Pixel loss만으로 L1 0.054, hp_corr 0.17까지 내려갔다.
- **해석:** 같은 모델이 정보가 있는 큰 ROI에서는 형태를 복원하고, ROI token이 RAE grid에서 2–16개뿐인 작은 ROI에서는 평균색 패치를 낸다. Figure 1이 작은 ROI만 보여 줘서 학습 실패처럼 보였던 것이다.
  - 다만 큰 ROI에서도 출력이 teacher보다 흐리고, 공간 분산 비는 0.13–0.18로 Bilinear 0.47보다 낮다. 회귀가 조건부 평균 쪽으로 smoothing되는 효과는 함께 존재한다.
