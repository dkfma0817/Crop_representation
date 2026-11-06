# Architecture Audit (실행 전, 2026-11-05)

E1–E7 구현 전에 이 서버의 기존 코드와 결과를 확인한 기록이다. 결론부터 적는다.

- **기존 S-series(A0/A1/B0/B1)는 E1–E7과 같은 실험이 아니다.** 입력(Block 4/8/12 multi-depth 융합), loss(teacher 정규화 공간의 multi-depth L12 + 0.5·L8 + 0.5·L4), 출력(모든 조건이 Bilinear + residual), memory(ROI+ring cache token) 모두 다르다. 그래서 결과를 재사용하지 않고 직접 비교하지도 않는다(참고 수치로만 쓴다).
- **R00 checkpoint는 이 서버에 없다.** 그래서 R00(CoordMLP) 구조 그대로 이번 프로토콜에서 다시 학습한다(B2, 새 init). 가장 가까운 기존 결과 H0는 입력과 loss가 달라서 참고로만 쓴다.
- **재사용 가능한 baseline:** IVC Bicubic(B1, R0 + R2), IVC C0(B4, R0 + R2), MSD Full ML+R2(B3, R2만). 세 결과 모두 같은 probe 코드(`src.ivc.train` / `src.readout.train`)와 같은 PACO dev/val 분할을 쓴다. **Direct Full final-layer**(B0)는 같은 프로토콜의 결과가 없어서 새로 probe한다.

## 1. 구조별 확인

| 구조 | Input | Query | Memory (K/V) | Attention | Output / Residual | Teacher | Loss | Evaluation | 결과 / 비고 |
|---|---|---|---|---|---|---|---|---|---|
| **R00 / CoordMLP** (`multilevel_relation_decoder/models_mlr.py`, `MLRDecoder('final')`) | F336 final-LN, ROI+ring token(ROI cell + 1-cell ring, 가변 길이) | 없음. 출력 cell마다 MLP를 쓴다 | 출력 cell마다 가장 가까운 ROI+ring token **k = 4개**(full-grid 거리) + Fourier offset, ROI 평균 + box vector, query 위치 | 없음 | `mlp(h)·std_T + mean_T`. 마지막 층이 zero-init이라 시작 출력은 teacher 평균. 입력에 zero-init adapter `F + A(F)` | C0 GT crop → 336 letterbox → 같은 frozen DINOv2 final-LN. 입력에 들어가지 않고, loss와 mean_T/std_T 상수에만 쓴다 | raw token `MSE_w + 1.0(1−cos)_w`, w = C0 coverage, ROI 평균(`losses_mlr` = `src/losses/feature.py`) | 옛 서버 PACO M2, test-primary | **checkpoint 없음**(rae_diag에서 확인). 옛 body init(gen_decoders mlp_s0)도 없음 |
| **RAEv2 K=1 Feature-only (A)** (`src/engine/train.py`, `configs/raev2_crop_recon`) | RAE 인코더 `dinov2_vitb14_reg` 256→224, **16×16** latent | R00와 같음(G=16) | R00와 같음 | 없음 | R00와 같음 | 256 letterbox GT crop | **`MSE_w + 1.0(1−cos)_w` raw latent, mean over ROI** | RAE latent PACO | **A checkpoint 이 서버에 없음.** 백본·해상도가 달라 이번 F336과 수치 비교 불가. **Loss 규칙만 그대로 가져온다** |
| **MSD H0/H1** (`src/msd`) | F336 Block 4/8/12, `F12 + A4(F4) + A8(F8)`(zero-init) | R00와 같음 | ROI+ring | 없음 | CoordMLP(`·σ12 + μ12`) | C0, Block 4/8/12 | **teacher 정규화 공간** `(x−μ)/σ`, coverage 가중 cell 평균 L12(H0), + 0.5·L8 + 0.5·L4(H1) | COCO dev L12로 ckpt 선택 → frozen R2 probe(PACO dev 1,000장으로 epoch 선택, val) | H0 val M0 R2: MSD report 참고. H1 43.67 |
| **S-series A0/A1** (`src/msd/sdec.py`) | H1과 같은 multi-depth 융합 | 24×24 grid feature(**Bilinear** sampling) + 2-d 위치 | ROI+ring cache token(4-d 위치) | **SA + CA + FFN** × 4 block, d 384 / 6 heads. A1은 4→8→16→24 계층 | `Bilinear(F12) + σ12·Linear(LN h)`, zero-init | C0 4/8/12 | H1 loss(정규화, multi-depth) | H1 probe | A0 43.21, A1 43.53 (val M0 R2) |
| **S-series B0/B1** | 같음 | **16개 box query**(학습 token + box MLP) | ROI+ring cache, 그리고 working map W(bilinear grid feature) | q: CA(cache), CA(W), FFN, +SA/FFN2(B0) 또는 W ← CA(q)(B1). 3 stage | `W + CA(W, q)` → `Bilinear + σ·Linear`, zero-init | 같음 | 같음 | 같음 | B0 / B1: sdec REPORT 참고 |
| **기존 Working Map / Attention decoder** | = S-series B (위). 이 저장소에 다른 구현은 없다 | | | | | | | | |
| **R2 reader** (`src/readout/models.py`, `kind r2, layers [12]`) | feature map(조건마다: full 576 token 또는 crop grid 576 token) | 학습 query + box embedding | token + geo MLP(rel x/y, cov, log size), **cov > 0 token만** | 2 cross-attention block, 8 heads | Linear head 59 × 2 level | – | masked BCE | PACO train(중복 6장 제외)에서 PACO dev 1,000장을 떼어 epoch 선택(dev M0 최대), val 최종. 10 epoch, ≈128 ROI/step, hflip, AdamW 3e-4 / 1e-3, seed 7 | |
| **R0 reader** (`src/ivc/common.py`) | 같은 feature map | – | coverage 가중 ROI 평균 → level별 z-score(train 4,000장) | 없음 | Linear head | – | masked BCE | R2와 같은 프로토콜(같은 process에서 R2와 동시에 학습) | |
| **PACO evaluation** (`src/readout/evaluate.py`, `src/ivc/analyze.py`) | – | – | – | – | – | – | – | M0 = (level, attr) pooled AP 평균, M2 = category-macro AP. D-bin = manifest `n_in`(F336 24×24 grid, D1–4 / D5–16 / D17+). Paired image bootstrap 1,000, seed 0 | IVC와 같은 bin 정의 |

## 2. S-series와 E1–E7의 실제 차이 (같은 실험이 아님)

| 항목 | S-series A/B | 이번 E1–E7 |
|---|---|---|
| Encoder 입력 | Block 4/8/12, zero-init adapter로 융합 | **Final L12만** |
| Loss | teacher 정규화 공간, cell 가중 평균, multi-depth(L12 + 0.5·L8 + 0.5·L4), aux head | **RAEv2/R00 raw token** `mean_ROI[MSE_w + (1−cos)_w]`, L12만, aux head 없음 |
| Query 초기값 | Bilinear grid feature(A), 학습 box query(B) | Mask token + ROI-relative PE(E2/E3/E5/E7), **Bicubic** feature(E1/E4/E6) |
| Memory | ROI + 1-cell ring (가변) | **ROI token만**(E2, ring 제외), **전체 576 token**(E3–E7) |
| Self-attention | A: 매 block, B0: query SA | E1(B 위 SA만, CA 없음), **E7만**(CA 뒤) |
| 출력 | 모든 조건이 Bilinear + zero-init residual | **Direct**(E2/E3/E4/E7: μ + σ·head) vs **Bicubic skip**(E1/E5/E6) |
| 크기 | d 384, 6 heads, 4 block / 3 stage, 10.7–11.5M | d 256, 8 heads, 2 block, **1.98–2.51M** |
| 위치 부호화 | crop-frame Fourier(2-d grid, 4-d token) | **ROI-relative (u, v)**: 하나의 PE를 query와 memory에 공유, sym-log 압축(no clamp), cell 크기 |

그래서 S-series 결과는 "다른 프로토콜의 attention decoder" 참고 수치로만 쓴다. E와 직접 차이를 계산하지 않는다.

## 3. 공통 Feature / 좌표 확인

- **Source = Target layer:** timm `vit_base_patch14_dinov2.lvd142m`, 336 letterbox(PIL bicubic, ImageNet norm, padding 0), Block 12 출력에 final LayerNorm 적용, prefix(CLS, register 없음 — lvd142m은 register가 없다) 제외 → 24×24×768. Student(full)와 Teacher(C0 crop)는 같은 frozen 가중치, 같은 layer, 같은 LN이다. bf16 autocast forward 후 fp32로 바꾼다(기존과 같음). Feature 추가 정규화 없음(raw).
- **Bicubic B:** `src.ivc.common.sample_roi(feats[12], geom, 'bicubic')`를 그대로 호출한다(= IVC B 조건). `grid_sample(fp32, bil_grid, bicubic, border, align_corners=False)`, clamp 없음, autocast 끔. E1/E4/E5/E6과 B1 baseline이 이 함수 하나를 쓴다.
- **좌표 chain:** 기존 `roi_geometry`(C0 cell 중심 → crop letterbox 역변환 → 원본 → F336 letterbox → grid 좌표). `roi_frame`(F336 canvas 위 GT box, C0 content rect)만 추가했다. Query와 memory의 ROI-relative (u, v)는 같은 affine chain이다.
- **Smoke 결과**(`smoke_cpu/alignment.json`): (u, v)에서 다시 계산한 query canvas 좌표와 `bil_grid`의 최대 오차 1.0e-5 px. Memory token (u, v)와 `roi_geometry` coords의 최대 오차 3.8e-6(ROI 단위). 합성 선형 ramp에서 bilinear 6e-5 px(float32). Bicubic은 interior 0.67 px(IVC와 같음)이고, canvas 경계 1–2 cell 안쪽(border clamp + 4×4 kernel)에서는 최대 1.35 px다. **kernel 성질이고 grid 구현 오류가 아니다.**
- **Teacher 누수:** decoder forward 인자는 `(feats, geom)`뿐이다. Teacher crop을 바꿔도 출력이 같다(smoke). `geom`의 C0 coverage(`valid_w`)는 box geometry에서 나온 값이고 teacher feature가 아니다. μ12/σ12(teacher 채널 통계, stream 3,008 ROI)는 R00의 mean_T/std_T와 같은 고정 상수다.

## 4. Mask 정책 (실행 전 고정)

1. **ROI token selection (E2):** `roi_geometry.inside` = F336 canvas에서 GT box와 겹치는 모든 cell(floor/ceil 규칙)이다. 정의상 ROI마다 최소 1개 token이 있다(없으면 `roi_geometry`가 ValueError). 그래서 **nearest-token fallback은 쓰지 않고, 적용 건수는 0**이다. Ring token은 memory에서 뺀다. Bicubic resampling은 쓰지 않는다(원본 token, 가변 길이 n_in).
2. **Batch padding mask:** E2 cross-attention key mask = `inside & ~pad`. E3–E7은 ROI 이미지의 576 token을 모두 쓴다(full letterbox padding 영역 token 포함, encoder 출력 그대로).
3. **C0 valid-coverage mask:** loss weight `w = valid_w`(padding cell 0). E1/E7 self-attention key mask = `valid_w > 0`. Invalid query는 누구에게도 key/value가 되지 않는다(smoke: invalid query에 noise를 넣어도 valid 출력 변화 0.0). Invalid query도 출력은 만들지만 loss weight가 0이고, R0/R2는 cov > 0만 읽는다.
- 평가에서 ROI를 크기 때문에 빼는 일은 없다.

## 5. 재사용 / 신규 판단

| Baseline | 판단 | 근거 |
|---|---|---|
| B0 Direct Full (final, R0 + R2) | **신규 probe** | 같은 프로토콜(PACO dev 1,000장, `src.ivc.train`)의 final-layer-only full 결과가 없다(readout_queue ctrl_f336은 dev split이 없는 이전 규칙) |
| B1 Bicubic-only | **IVC `probe_bic` 재사용** | 같은 `sample_roi` 호출, 같은 student forward, 같은 probe 코드·config(R0 + R2, 10 ep, dev 1,000, seed 0) |
| B2 R00 | **신규 학습 + probe** | ckpt 없음. 같은 데이터·loss·recipe·probe로 다시 학습한다. 옛 R00 대비 차이는 fresh init뿐이다 |
| B3 Full Multi-layer + R2 | **MSD `probe_mlr2` 재사용 (R2만)** | 같은 probe 프로토콜. 입력 정보가 더 많아서 실용 baseline으로만 쓴다 |
| B4 C0 Oracle | **IVC `probe_c0` 재사용** | 같은 프로토콜, R0 + R2 |
| 참고(직접 비교 안 함) | H0/H1, S-series | 입력과 loss가 다르다 |

## 6. Decoder capacity 주의

Hidden 256 / 2 block은 파일럿 고정값이다. 모든 조건이 baseline을 넘지 못해도 decoder capacity의 한계는 배제할 수 없다. Parameter 수: E1 1.98M, E2/E3/E5 1.99M, E4/E6 2.18M, E7 2.51M, R00 10.16M.
