# RAE Reconstruction 진단과 R00 구조 확인 (PACO val, 2026-11-05)

- 학습 없음. 기존 RAEv2 K=1 encoder/decoder와 이 서버에 있는 기존 checkpoint만 썼다. PACO Test는 쓰지 않았다.
- 실행 전 규칙은 [`protocol.md`](../../configs/raediag/protocol.md), 구조 확인은 [`architecture_audit.md`](../../configs/raediag/architecture_audit.md)에 있다.
- 자동 생성 표는 [`tables.md`](tables.md)(paired CI 포함), ROI별 원자료는 `reconstruction_results.csv`, `latent_cycle_results.csv`다.
- 범위: PACO val **전체 11,109 ROI / 2,051 이미지**(object 3,140, part 7,969). 크기 bin은 manifest n_in(F336 grid) 기준 D1–4 / D5–16 / D17+.
- 모든 조건 간 차이는 같은 ROI 쌍에 대한 이미지 bootstrap 1,000회(seed 0) 95% CI로 확인했다. 여기서 말하는 차이는 특별히 언급하지 않으면 모두 CI가 0을 포함하지 않는다.

## 결론 먼저

| 질문 | 관찰 |
|---|---|
| **Q1** GT crop latent는 잘 복원되는가? | **예.** A: L1 0.072, LPIPS 0.168, SSIM 0.68, hp_corr 0.37. 재인코딩 token cosine 0.86(decoder → encoder 왕복 자체의 상한) |
| **Q2** Full 복원 후 crop하면 얼마나 남는가? | **형태와 색은 남고 세부는 거의 없다.** B: LPIPS 0.322, hp_corr 0.03, 재인코딩 cosine 0.56(D17+ 0.68, D5–16 0.48). 입력 픽셀만 crop한 B_in은 hp_corr 0.38, 재인코딩 0.75. 세부 손실의 대부분은 입력 해상도보다 **full-image autoencoding**에서 생긴다(B_in − B: hp_corr +0.35, 재인코딩 cos +0.19). 다만 작은 ROI는 입력 해상도 한계도 크다(Part D1–4의 B_in 재인코딩 0.59) |
| **Q3** Bilinear/Bicubic ROI latent를 decoder에 넣으면 왜 나빠지는가? | Full latent를 그대로 decode한 B보다도 나쁘다(C: LPIPS 0.386, 재인코딩 0.41, 격자 무늬). 원인 후보를 대조군으로 하나씩 바꿔 봤다. **Token 에너지(Bicubic)나 평균 표현(F)을 C0에 맞춰도 거의 그대로**이고, **C0 latent를 full-grid 해상도로 smoothing한 E가 C와 같은 수준으로 실패**한다. 관찰과 가장 일관된 설명은 "ROI 안 공간 구조가 거의 평평한 latent(인접 token cosine 0.98 vs C0 0.83)를 decoder가 정상 이미지로 그리지 못한다"는 것이다. 인과를 증명하지는 않는다 |
| **Q4** 복원 이미지를 다시 encode하면 C0 feature에 얼마나 가까운가? | **조건 평균 사이에서는 RGB fidelity와 latent fidelity가 같은 방향**이다(LPIPS vs 재인코딩 cos Spearman −0.98, 8개 조건). **ROI 단위로는 약하게만** 같이 움직인다(조건 안 ρ −0.26~−0.43). 그리고 **입력 latent가 C0에 가까워도 복원 결과는 가깝지 않다**: E는 입력 cos 0.86인데 재인코딩 0.42, Pixel-only MB는 입력 cos 0.25인데 재인코딩 0.44 |

---

## 1. 기존 R00의 실제 구조 (상세는 `architecture_audit.md`)
- **R00 = zero-init Adapter(입력 ROI token에 residual) + 기존 CoordMLP body(그대로 복사)**. 초기화 시점 출력은 CoordMLP와 같다. RAE 이식본과 DINOv2 원본은 grid import 한 줄만 다르다.
- 입력과 출력:
  - 입력: student full-image 최종층 token 중 ROI + 1-cell ring token (R, N, 768), box 기하 정보(`tok`, `coords`, `pad`, `inside`, `valid_w`, `bil_grid`), 출력 query 위치 P.
  - 출력: crop grid latent (R, G², 768). RAE는 G = 16, DINOv2는 G = 24.
- **Local token 경로가 실제로 있다.** 출력 위치마다 full-grid 거리로 가장 가까운 **4개** ROI/ring token을 kNN gather한다(보간 아님). LN feature와 상대 offset Fourier feature가 들어간다.
- **ROI mean을 쓴다.** Adapter를 거친 **student full feature**의 box 내부 token 비가중 평균에 box vector(7차원)를 붙인다.
- 위치 정보: crop 기준 query 위치 + full 기준 절대 위치 + 상대 offset.
- **MLP는 출력 위치마다 독립적으로 적용**되고, 위치 사이에서 공유되는 정보는 ROI mean과 box뿐이다.
- **Teacher feature는 decoder 입력에 들어가지 않는다.** Teacher가 쓰이는 곳은 loss target과 출력 scale 상수(`mean_T`, `std_T` = teacher 채널별 dataset 통계)뿐이다.
- Gradient 경로:
  - Feature loss: raw latent MSE + (1 − cos) → CoordMLP.
  - Pixel loss: Z → frozen RAE decoder → L1 + 0.2·LPIPS → decoder를 통과해 CoordMLP로. Encoder는 cache라 gradient가 없다.
- 이 서버의 기존 checkpoint: Feature-only(A)와 Feature+Pixel(C) CoordMLP는 **없다**(원래 서버에만 있었음). 새로 학습하지 않았다. Pixel-only(B, 12k)만 있어서 MB 조건으로 넣었다. DINOv2 timm 공간의 R00/R10은 RAE decoder와 latent 공간이 달라 연결하지 않았다.

## 2. 조건과 정렬
| | 경로 |
|---|---|
| A | GT crop canvas → enc → T(C0 latent) → dec |
| B | full canvas → enc → Zf → dec → full RGB → GT ROI를 crop canvas pixel grid로 재샘플 |
| B_in (참고) | full **입력** canvas를 같은 방식으로 재샘플(enc/dec 없음): 입력 해상도의 한계 |
| C / D | Zf → crop grid bilinear / bicubic → dec (같은 grid, 같은 padding 규칙) |
| E (진단) | T를 ROI의 full-grid 크기로 area-pool한 뒤 다시 확대(평균 유지, 공간 해상도만 full grid 수준) → dec |
| F (진단) | C의 content 평균만 T 평균으로 교체 → dec |
| MB | 기존 Pixel-only CoordMLP(12k) → dec |

- E와 F는 GT crop latent를 쓰는 **원인 분리용 대조군**이고, 사용 가능한 방법이 아니다.
- **정렬 검증**(`alignment_checks/alignment.json`, 그림 12장):

| 검사 | 결과 |
|---|---|
| Feature grid(독립 수식 vs `bil_grid`, 11,109 ROI) | 최대 오차 7.6e-6 px. 유효 cell의 3.3%는 canvas 가장자리 border clamp |
| 합성 좌표 latent | Bilinear 내부 최대 3.8e-5 px. **Bicubic 0.62 px 평균**(PyTorch kernel A = −0.75가 선형을 재현하지 않는 특성. Grid는 같음) |
| RGB 재샘플(B/B_in, 합성 좌표 canvas) | 최대 4.6e-5 px |
| B_in pixel 정합(±2 px shift) | L1이 (0, 0)에서 최소(0.034, ±1 px에서 0.037) |
| Feature 정합(Bicubic grid ±8/±16 px shift vs T) | cosine이 (0, 0)에서 최대(0.591, ±8 px에서 0.569) |
| Encoder 256→224 | `F.interpolate(bicubic, align_corners=False)`. 256 canvas의 16 px cell이 224 입력의 14 px patch와 정확히 대응 |

- RGB 재샘플은 bilinear이고 antialias가 없다. Crop scale은 항상 full scale 이상이라 확대 방향만 있다. GT crop canvas는 원본에서 PIL bicubic으로 만든다. 두 resize 방식의 차이는 B_in 자체 오차(L1 0.040)에 포함된다.
- Padding 처리: RGB 지표는 content 픽셀만 쓴다(LPIPS는 content 밖을 GT로 합성). Latent 지표는 content cell(coverage 가중)만 쓴다. C/D의 padding cell에는 box 밖 context feature가 들어가고, decoder는 전역 attention이라 이 값이 content 복원에 영향을 줄 수 있다(한계 참고).

## 3. RGB 재구성 (모든 ROI 평균, per-ROI p10 / p50 / p90은 `tables.md`)
| 조건 | L1 ↓ | PSNR ↑ | LPIPS ↓ | SSIM ↑ | hp_corr ↑ | 평균 색 dE ↓ |
|---|---|---|---|---|---|---|
| A GT crop | 0.072 | 20.96 | **0.168** | 0.678 | 0.367 | 6.3 |
| B_in 입력 픽셀 crop (참고) | 0.040 | 26.12 | 0.185 | 0.771 | 0.376 | 0.5 |
| B full 복원 → crop | 0.134 | 15.72 | 0.322 | 0.501 | 0.030 | 8.4 |
| C Bilinear | 0.179 | 13.72 | 0.386 | 0.366 | 0.003 | 16.7 |
| D Bicubic | 0.176 | 13.79 | 0.380 | 0.383 | 0.004 | 16.1 |
| E C0 latent smoothing | 0.152 | 14.75 | 0.357 | 0.494 | 0.009 | 12.6 |
| F Bilinear + C0 평균 | 0.152 | 14.74 | 0.353 | 0.479 | 0.003 | 12.9 |
| MB Pixel-only CoordMLP | 0.150 | 15.01 | 0.342 | 0.486 | 0.003 | 12.1 |

- B_in의 L1/PSNR이 A보다 좋은 것은 같은 원본 픽셀을 resize만 한 것이기 때문이다. 세부(hp_corr)는 비슷하지만, 이것은 지표상 비교일 뿐 B_in은 crop 해상도 정보를 갖고 있지 않다.

**크기별** (LPIPS / hp_corr / 재인코딩 token cos)

| bin | A | B_in | B | C | D | E | F | MB |
|---|---|---|---|---|---|---|---|---|
| Obj D1–4 (102) | .20 / .47 / .86 | .27 / .19 / .64 | .43 / .00 / .46 | .47 / .00 / .35 | .47 / .00 / .35 | .43 / −.01 / .37 | .43 / .00 / .37 | .41 / .00 / .39 |
| Obj D5–16 (1,384) | .18 / .43 / .88 | .23 / .28 / .73 | .39 / .01 / .47 | .43 / .00 / .34 | .42 / .00 / .34 | .42 / .00 / .35 | .41 / .00 / .35 | .39 / .00 / .37 |
| Obj D17+ (1,654) | .18 / .30 / .90 | .14 / .69 / .91 | .31 / .08 / .69 | .38 / .01 / .48 | .37 / .01 / .49 | .39 / .02 / .43 | .38 / .01 / .48 | .36 / .01 / .46 |
| Part D1–4 (2,871) | .16 / .39 / .82 | .21 / .12 / .59 | .32 / .00 / .50 | .41 / .00 / .43 | .41 / .00 / .43 | .32 / .00 / .49 | .32 / .00 / .49 | .33 / .00 / .49 |
| Part D5–16 (2,889) | .17 / .41 / .85 | .20 / .28 / .72 | .33 / .01 / .49 | .37 / .00 / .36 | .36 / .00 / .36 | .35 / .00 / .37 | .35 / .00 / .37 | .33 / .00 / .40 |
| Part D17+ (2,209) | .16 / .29 / .89 | .13 / .68 / .90 | .28 / .08 / .68 | .35 / .01 / .46 | .34 / .01 / .47 | .36 / .02 / .41 | .35 / .01 / .46 | .33 / .01 / .45 |

**주요 조건 간 차이** (전체, [95% CI])
- **B − C** (같은 full latent를 full 크기로 decode vs ROI sampling 후 decode): LPIPS −0.064 [−0.066, −0.061], 재인코딩 cos **+0.147** [+0.144, +0.151]. D17+에서 +0.217, D1–4에서 +0.079.
- **D − C** (Bicubic − Bilinear): LPIPS −0.006, SSIM +0.017, 재인코딩 cos +0.007 [+0.006, +0.008]. 통계적으로는 차이가 있지만 크기가 매우 작다.
- **B_in − B**: hp_corr **+0.346** [+0.338, +0.355], 재인코딩 cos +0.194.
- **E − C**: 재인코딩 cos +0.004 [+0.002, +0.007], LPIPS −0.029. **F − C**: +0.021, −0.033. **MB − C**: +0.024, −0.044.
- E와 F는 D1–4에서만 이득이 있다(재인코딩 +0.06, LPIPS −0.09). D17+에서는 E가 C보다 나쁘다(재인코딩 −0.045).

## 4. Latent 재인코딩과 decoder 입력 분포 (전체 평균)
| 조건 | 입력 latent vs T: token cos / pooled cos | 복원 → 재인코딩 vs T: token cos / pooled cos / spatial var 비 | cycle: 재인코딩 vs 입력 cos | 입력 token norm | 입력 z² (공식 stats) | 입력 인접 token cos |
|---|---|---|---|---|---|---|
| A | 1 / 1 | **0.861** / 0.925 / 1.08 | 0.861 | 27.7 | 0.93 | **0.83** |
| B | – | 0.559 / 0.689 / 0.82 | – | – | – | – |
| B_in | – | 0.753 / 0.831 / 0.91 | – | – | – | – |
| C | 0.538 / 0.646 | 0.412 / 0.521 / 0.64 | 0.377 | 25.4 | 0.77 | 0.98 |
| D | 0.525 / 0.638 | 0.418 / 0.528 / 0.66 | 0.366 | 27.1 | 0.93 | 0.98 |
| E | **0.861** / 0.999 | 0.416 / 0.526 / 0.59 | 0.480 | 23.4 | 0.55 | 0.99 |
| F | 0.826 / **1.000** | 0.432 / 0.546 / 0.63 | 0.477 | 24.3 | 0.63 | 0.98 |
| MB | 0.248 / 0.301 | 0.436 / 0.563 / 0.72 | 0.323 | 24.8 | 0.89 | 0.99 |

- z² = 공식 per-position stats로 정규화한 latent의 평균 제곱. 학습 분포에서는 1에 가깝다.

## 5. 관찰된 사실과 가능한 해석

**관찰**
1. **Decoder 자체는 crop latent를 잘 복원한다**(A). 다만 decode → encode 왕복으로 token cosine은 0.86까지만 회복된다. 이번 재인코딩 지표들의 실질적 상한이다.
2. **Full image를 한 번에 autoencode하고 crop하면(B) 객체 형태와 색은 남지만 GT에 정렬된 세부는 거의 없다**(hp_corr 0.03). 같은 crop을 입력 픽셀에서 하면(B_in) 세부가 남는다(0.38). Full 복원 손실은 큰 ROI에서도 크다(D17+ hp_corr 0.08 vs B_in 0.69).
3. **ROI-sampled latent(C/D)는 같은 full latent를 full 크기로 decode한 B보다 일관되게 나쁘다.** 모든 bin에서 그렇고, 결과는 객체 없는 격자 무늬다(`figures/examples_*.png`). 큰 ROI(D17+)에서는 입력 픽셀(B_in 0.91)과 full decode(B 0.69)에는 정보가 남아 있는데 C는 0.48이다.
4. **C/D의 입력 latent에서 C0와 다른 점들**: 너무 매끄럽다(인접 cosine 0.98 vs 0.83), effective spatial variance가 낮다(0.27), Bilinear는 token 에너지도 낮다(z² 0.77).
   - Bicubic(D)은 에너지를 C0 수준(z² 0.93)으로 되돌리지만 복원은 거의 그대로다(재인코딩 +0.007).
   - 평균만 C0로 바꾼 F는 pooled cosine 1.0이 되지만 재인코딩 +0.021에 그친다.
   - **C0 latent 자체를 full-grid 해상도로 smoothing한 E**(입력 token cos 0.86, pooled 0.999)도 C와 같은 수준으로 실패한다(재인코딩 0.416 vs 0.412, 그림에서도 같은 격자 무늬).
5. **입력 latent가 C0에 가까운 정도와 복원 결과가 C0에 가까운 정도는 따로 움직인다.** E는 입력 0.86인데 출력 0.42이고, Pixel-only MB는 입력 0.25인데 출력 0.44다. MB는 RGB가 C/D보다 낫지만(LPIPS 0.342) latent는 C0와 가장 멀다.
6. RGB fidelity와 재인코딩 fidelity는 조건 평균 사이에서는 거의 같은 순서다(ρ −0.98). 같은 조건 안의 ROI 사이에서는 약하게만 같이 움직인다(ρ −0.26~−0.43, A는 +0.12).

**해석(추정, 검증되지 않은 부분 포함)**
- Q3에 대해: 관찰 4는 "C/D 실패의 주원인이 token 에너지 차이나 평균 표현 차이"라는 설명과 맞지 않는다. 반대로 "ROI 안에서 공간적으로 평평한 latent(full-grid 대역폭)는, 내용이 C0에서 왔더라도 이 decoder가 정상 이미지로 그리지 못한다"는 설명과는 맞는다.
  - 다만 E는 smoothing 때문에 에너지도 함께 낮아진다(z² 0.55). 그래서 공간 대역폭과 에너지를 E 하나로 완전히 분리하지는 못했다. 에너지만 맞춘 D가 실패했다는 점이 보조 근거다.
- Q2와 연결: B > C는, 같은 full latent에 담긴 정보가 full canvas 문맥에서는 decoder가 어느 정도 그릴 수 있는 형태로 남아 있다는 뜻이다. 이를 crop grid로 늘린 latent는 decoder 입력으로 적합하지 않다. 즉 **"정보가 없다"와 "ROI sampling feature가 decoder 입력에 맞지 않는다"가 둘 다 일부 작동**한다.
  - 작은 ROI: 입력 해상도와 full encoding의 한계가 크다(B_in과 B도 낮음).
  - 큰 ROI: sampling한 latent의 형식 문제가 두드러진다(B_in과 B는 높은데 C만 낮음).
- 이 결과를 latent manifold 이탈의 증거로 단정하지 않는다. 관찰된 것은 특정 형태(평평한 공간 구조)의 latent에서 이 decoder의 출력이 무너진다는 것까지다.

## 6. 구분할 수 없는 원인과 한계
- **Encoder에 정보가 있는지 자체는 판정할 수 없다.** B의 세부 손실은 encoder 정보 손실, decoder의 full-image 복원 한계, 256 해상도 중 어느 것인지 이 실험으로 나눌 수 없다. Decoder 출력으로 정보량을 재는 한계다.
- E는 공간 대역폭과 token 에너지를 함께 바꾼다. 둘을 따로 통제한 조건은 없다.
- C/D/F의 padding cell에는 box 밖 context latent가 들어가고, decoder는 전역 attention이다. Padding 내용이 content 복원에 미치는 영향은 따로 떼어 보지 않았다.
- 재인코딩 지표는 같은 encoder 공간이라 A도 0.86에서 멈춘다. 절대 수준보다 조건 간 상대 비교로 읽어야 한다.
- Feature-only / Feature+Pixel CoordMLP checkpoint가 이 서버에 없어, 기존 11-04 A/C 모델의 cycle은 측정하지 못했다(Pixel-only만 측정).
- RAE 쌍(dinov2_vitb14_reg, 224, LN affine 없음, 16×16)은 IVC/PACO 실험의 timm DINOv2 F336(24×24)과 다른 공간이다. 이번 수치를 IVC feature 차이와 직접 비교하지 않는다.
- RGB 지표는 decoder 특성의 영향을 받는다. hp_corr은 세부 위치 일치만 본다.
- 학습 seed 개념이 없는 결정적 평가다. CI는 ROI/이미지 표본 불확실성만 반영한다.

## 그림과 파일
- ROI 크기 × level별 A/B/C/D 비교(행 = full 입력 | GT | A | B_in | B | C | D | E | F | MB, seed 0으로 사전 고정한 3개씩): `figures/examples_{object,part}_{D1-4,D5-16,D17+}.png`
- 분포: `figures/dist_lpips_recos_by_bin.png`. RGB vs latent fidelity: `figures/rgb_vs_latent_fidelity.png`.
- 정렬: `alignment_checks/align_*.png`, `alignment.json`.
- 코드: `src/raediag/{run,align,report}.py`. 고정 샘플: `configs/raediag/sample_rois.json`.
- 실행: `python -m src.raediag.run --shard {0,1} --nshard 2`(GPU 2,3, shard당 약 5분), `src.raediag.align`, `src.raediag.report`. Smoke 결과는 `smoke/`.
- GPU 2,3의 QAT는 강제 중단 후(step 약 21,898) checkpoint-20000부터 `--resume`으로 재개했다(`qat_pause/`).
