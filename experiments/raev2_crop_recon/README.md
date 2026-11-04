# RAEv2 ROI Crop-Latent Reconstruction 파일럿: Pixel Supervision(A/B/C) 효과

작성 2026-11-04 19:40 KST. 학습 16:30–19:10 KST (A GPU 0, B GPU 1, C GPU 7), probe·평가 ~19:30 KST.

> 2026-11-04 리팩토링 이후 코드는 repo 루트의 `src/`, 실행 스크립트는 `scripts/raev2_crop_recon/`, 캐시는 `cache/`로 옮겼습니다. 아래 본문의 `code/…`, `perception_cache/…` 경로는 원래 서버 기준 기록입니다 (대응표: 루트 README).

사전 등록 `configs/protocol.md`, seed 0 비교표 `results/s0_table.md`(자동 생성), 재구성 `results/recon/`, 그림 `results/viz/`.

## 0. 결론

**Pretrained RAEv2 Pixel Decoder를 통한 reconstruction supervision은 작은 ROI feature를 crop-like representation으로 만드는 데 실질적인 도움이 되지 않았다** (seed 0 파일럿 기준).
- Pixel-only(B)는 RGB 지표가 가장 좋지만 teacher latent에서 멀어지고(cos 0.23) PACO가 크게 떨어진다 (R2 −5.2, R0 −6.3 vs A).
- Feature+Pixel(C)은 latent 품질을 A와 똑같이 유지하며 색 재구성만 개선했다. PACO C−A는 R2 +0.62 / R0 +0.11 (overall)로 seed 편차(R2 0.49, R0 0.21) 수준이며,
  C는 여전히 Full 직접 read-out(R2 −0.66)을 넘지 못하고 GT crop 상한과의 격차(overall −9.3, Obj D5-16 −19.6)는 그대로다.
- A·B·C 모두 GT 대응 디테일(hp_corr ≈ 0.00–0.01, teacher 0.29–0.44)을 복원하지 못한다. 개선은 저주파 색뿐이다.
- 사전 등록한 종료 규칙(복원만 개선되고 AP 이득 없음 → 추가 Loss/구조 탐색 없이 종료)에 해당. 유일하게 남는 신호는 R2의 C−A Obj D5-16 +1.62
  (seed 편차 1.18)로, 다중 seed 재검증 대상이 될 수 있으나 R0에서는 −0.45로 재현되지 않는다.

## 1. RAEv2 checkpoint와 실제 shape (results/verify_pair.json, sanity.json S1)
| 항목 | 값 |
|---|---|
| Repo / HF | nanovisionx/RAEv2 @ 8a0d238 / nyu-visionx/RAEv2-models @ 9770b7b |
| Decoder | `stage1/imagenet/dinov2b-k1/decoder.pt` sha256 aabfee19… (= HF etag), ViT-XL(1152×28, 16 heads) 415.35M, pos-embed (1,257,1152), 출력 256×256 RGB. Missing/unexpected key 없음 |
| Stats | mean, var (768,16,16) per-position, 축약 없이 사용 |
| Encoder | torch.hub `dinov2_vitb14_reg` (register 4, final LN affine 제거, pos-embed 16×16 resample) — 기존 timm lvd142m과 다름 |
| 경로 | RAE resolution 256 → 공식 preprocess(ImageNet norm, bicubic 224) → raw latent (B,256,768) / normalised (B,768,16,16) → decoder → (B,3,256,256) |
| Pair 검증 (ImageNet val 256, 공식 sample 전처리) | 기본 PSNR 18.72 / LPIPS 0.256; [woreg] 8.37 / 0.878; [norm] 16.21 / 0.334 → 기본만 맞는 쌍. 경험 stats vs stats.pt: mean corr 0.993, var 비 0.995 |
| Wrapper vs 공식 `stage1.rae.RAE` (noise_tau 0, eval) | encode max diff 4.8e-5, decode 2.9e-5 |

## 2. 기존 CoordMLP 구조 분석 (models_mlr.py = gen_decoders models_g.CoordMLP + R00 adapter)
1. **가변 ROI → 고정 grid**: ROI 안 token + 1-cell ring(가변 N)을 모으고, 출력 query = teacher crop canvas의 G×G cell. 각 query의 full-image 위치(bil_grid)에서
   가장 가까운 k=4 ROI token의 LN feature + 상대 offset Fourier → l_local, ROI 평균 + box vector → l_glob, query 위치 → l_pos, 합 → MLP → 768.
2. **Bilinear/skip 없음**: kNN gather(가중 보간 아님). 출력 skip 없음 — 마지막 층 zero-init이라 초기 출력 = teacher 평균(mean_T). R00 adapter만 입력 쪽 residual.
3. **위치 정보는 둘 다**: P = crop canvas 기준 정규화 좌표(ROI-relative), u/G = full-image 절대 위치, rel = token−query offset(full grid 단위), box vector(full grid). 올바르게 처리되어 보완 불필요 (S5: 합성 좌표 이미지로 오차 ≤0.27% W, 8-bit 양자화 0.39% 이하).
4. **R00 Adapter** = LN→768→192→768 zero-init bottleneck을 입력 ROI token에 residual로 더함(0.30M). **CoordMLP body**(9.86M)가 grid 변환과 예측 담당.
5. **정규화 차이**: 기존 = timm lvd142m(register 없음) final LN **affine 있음**, loss는 raw token. RAE = reg 모델, LN **affine 없음**(token norm √768 = 27.7), 공식 latent는 per-position stats로 정규화.
   이번에는 CoordMLP 출력·feature loss·decoder 입력 모두 raw 공간(decoder(raw) ≡ RAE.decode(normalise(raw)), 이중 역정규화 없음). CoordMLP의 mean_T/std_T는 기존 구조의 채널별 출력 스케일(RAE stats 아님).
- 변경은 grid 크기(G=16, S=256, PATCH=16) import 한 줄. 세 run 모두 같은 새 init 파일(seed 0)에서 시작, R00/mlp_s0 가중치 미사용.

## 3. Sanity Check — 모두 통과
| | 결과 |
|---|---|
| S1 공식 클래스 일치 | 위 표 |
| S2 GT crop → enc → dec (hval) | L1 .074, PSNR 20.3, LPIPS .183, hp_corr .35 — 정상 (results/viz/sanity_teacher_bilinear.png) |
| S3 Bilinear → dec | L1 .187, LPIPS .428, latent cos .61, svar .32 — 큰 ROI만 형태, 작은 ROI는 격자 아티팩트 |
| S4 shape/공간 | teacher·full·CoordMLP 출력 (R,256,768) raw; crop content 정규화 latent var 0.92 (ImageNet 분포와 유사), padding cell은 이탈 |
| S5 좌표 | 위 2-3 |
| S6 Pixel gradient | 첫 step 2 tensor(zero-init 마지막 층) → 2 step 후 20/20 tensor; decoder grad 0, 가중치 불변(fingerprint), padding cell latent도 gradient 받음(decoder 전역 attention) |
| S7 메모리 (4090, BF16, ckpt 없음) | micro 1/2/4/8/16 ROI → peak 4.1/4.6/5.7/8.0/12.6 GB, 32 OOM → micro 16 × 4 누적 = 64 ROI/step (A/B/C 동일 분할). activation checkpointing 불필요 |
| S8 256 ROI 과적합 (1k step) | Feature-only: feat loss 1.23→0.48, cos .51→.83. Pixel-only: L1 .249→.054, LPIPS .53→.27, dE 26.7→1.3, **latent cos .51→.40** |
- Loss 가중치 (calib.py, 고정 train batch 8개): ‖∇LPIPS‖/‖∇L1‖ = 3.21 → β = 0.2; ‖∇L_pix‖/‖∇L_feat‖ = 5.04 → λ = 0.1 (λq = 0.50). 두 gradient cos 0.006 (거의 직교). 학습 중 변경 없음.
- 학습 무결성: 세 run 모두 step-0 dev 동일(feat 1.239), decoder/LPIPS fingerprint 불변, NaN/OOM 0건. 시간 A 9분, B ~2.5 h, C ~2.7 h (초반 GPU 7을 baseline probe와 공유, 이후 GPU 0으로 이동).

## 4. A/B/C 재구성 및 PACO 절댓값

### 재구성 (12k, hval 3,194 GT ROI, 학습 pool 밖 COCO 1,000장; PACO test-primary bin별은 results/recon/summary.csv)
| | RGB L1 | LPIPS | 평균 색 dE | hp_corr (GT 디테일) | latent cos vs teacher | spatial var 비 |
|---|---|---|---|---|---|---|
| Teacher (상한) | .082 | .181 | 7.3 | .318 | 1 | 1 |
| Bilinear | .182 | .408 | 15.2 | .005 | .617 | .322 |
| A Feature-only | .197 | .432 | 18.8 | .001 | .710 | .128 |
| B Pixel-only | **.149** | **.362** | **10.2** | .006 | .227 | .094 |
| C Feature+Pixel | .162 | .393 | 11.6 | .003 | .709 | .128 |
PACO D1-4 part / D5-16 object / D17+ object에서도 순위 동일 (B < C < Bilinear ≈ A in L1, hp_corr 전부 ≈ 0, teacher .29–.44).

### PACO test-primary M2 (12k, reader seed 0; 사용자 지시로 seed 0 단일 비교, 통계 검정 없음)
| 조건 | R2 Overall | Obj D5-16 | Part D1-4 | Color | Material | Pattern | R0 Overall | R0 Obj D5-16 | R0 Part D1-4 |
|---|---|---|---|---|---|---|---|---|---|
| Full (GT-ROI 직접 read-out) | 35.38 | 31.42 | 27.72 | 28.84 | 31.08 | 37.46 | 29.70 | 29.39 | 28.47 |
| Bilinear | 35.33 | 31.00 | 28.50 | 27.84 | 30.88 | 37.79 | 29.85 | 28.77 | 29.23 |
| GT crop oracle | **44.01** | **51.57** | **39.16** | **36.23** | **36.25** | **53.37** | **37.53** | **43.88** | **36.08** |
| A Feature-only | 34.10 | 30.36 | 28.25 | 25.51 | 30.74 | 36.48 | 30.58 | 28.52 | 28.14 |
| B Pixel-only | 28.95 | 27.76 | 26.73 | 25.28 | 25.47 | 27.00 | 24.31 | 23.21 | 27.09 |
| C Feature+Pixel | 34.72 | 31.99 | 29.38 | 26.14 | 30.94 | 35.93 | 30.70 | 28.07 | 28.25 |

| 차이 (seed 0) | R2 Overall | Obj D5-16 | Part D1-4 | Color | Material | Pattern | R0 Overall | R0 Obj D5-16 |
|---|---|---|---|---|---|---|---|---|
| **C − A (주 비교)** | **+0.62** | **+1.62** | +1.13 | +0.63 | +0.20 | −0.56 | **+0.11** | **−0.45** |
| B − A | −5.15 | −2.60 | −1.52 | −0.23 | −5.27 | −9.48 | −6.27 | −5.31 |
| C − B | +5.77 | +4.23 | +2.65 | +0.85 | +5.47 | +8.93 | +6.38 | +4.86 |
| C − Full | −0.66 | +0.57 | +1.65 | −2.70 | −0.14 | −1.54 | +0.99 | −1.32 |
| A − Full | −1.28 | −1.05 | +0.52 | −3.33 | −0.34 | −0.98 | +0.88 | −0.87 |
| Crop − Full | +8.63 | +20.16 | +11.43 | +7.39 | +5.17 | +15.91 | +7.83 | +14.49 |
| **Seed 노이즈 (A, seed 0/1/2 max−min)** | 0.49 | 1.18 | 1.28 | 0.14 | 0.46 | 1.89 | 0.21 | 0.34 |
- 보조 bin: SigLIP2 5-16 object C−A R2 +1.67 / R0 −0.47; RAE-grid bin은 results/s0_table.md.
- 참고: 기존 DINOv2 F336/C0 수치(40.7/45.4)와 직접 비교하지 않는다(backbone, 224 입력, register, LN affine이 다름).

## 5. 시각화
`results/viz/recon_paco_12000.png` (PACO test-primary, bin별 4개; 행 = GT crop / RAE teacher / Bilinear / A / B / C; 예시 id는 `_examples.json`),
`results/viz/recon_hval_12000.png`, `results/viz/sanity_teacher_bilinear.png`, `results/viz/verify_pair_imagenet.png`.
- Teacher recon은 GT와 거의 같다. Bilinear는 격자 아티팩트. A는 거의 단색 평균색 패치. B는 색과 큰 윤곽(노트북, 베개)이 흐릿하게 나타남. C는 A와 B 사이. 작은 ROI의 실제 무늬·글자·질감을 복원한 경우는 없다.

## 6. Pixel-only vs Feature-only
- B는 RGB 공간에서 A보다 확실히 좋다 (L1 −24%, LPIPS −16%, dE 18.8→10.2). 그러나 teacher latent와의 cos가 0.71 → 0.23으로 무너지고, PACO는 R2 −5.2 / R0 −6.3.
- 손실은 material(−5.3), pattern(−9.5), 큰 객체(D17+ −7.8)에 집중되고 color는 −0.2로 거의 같다 → decoder가 그럴듯한 저주파 색을 그리는 latent는 attribute 정보를 담은 crop latent가 아니다.
- 256 ROI 과적합에서도 같은 현상(RGB 거의 맞춤, cos .51→.40). **Pixel-only의 RGB 개선은 crop latent 복원 성공이 아니다.**

## 7. Feature+Pixel의 추가 이득
- C는 A의 latent 지표(cos .709, svar .128)를 그대로 유지하면서 RGB 색만 개선 (dE 18.8→11.6, L1 −18%). 두 loss gradient가 거의 직교(cos 0.006)해 서로 간섭하지 않은 결과.
- PACO에서는 R2 +0.62(노이즈 0.49), Obj D5-16 +1.62(노이즈 1.18)의 작은 양의 신호가 있으나 R0에서는 +0.11 / −0.45로 사라진다. 색 개선이 color AP로도 이어지지 않는다(R2 +0.63, Full 대비 −2.7).
- C도 Full 직접 read-out을 R2에서 넘지 못하고(−0.66), GT crop과의 격차(Obj D5-16 −19.6)는 사실상 줄지 않았다.

## 8. 실패 원인과 근거
1. **정보 병목은 decoder 입력 쪽**: 작은 ROI의 full-image 16×16 latent(5–16 token 객체도 RAE grid에서는 훨씬 적은 token)에는 crop 수준 디테일이 없다. hp_corr이 A/B/C/Bilinear 모두 ≈0이고 teacher만 .3–.4 → 어떤 loss로도 없는 디테일은 만들 수 없음. Crop−Full +20(D5-16)은 재인코딩이 새 정보를 가져오기 때문.
2. **Pixel loss의 해가 crop latent가 아님**: frozen decoder는 많은 latent를 비슷한 RGB로 보낸다(B의 cos .23인데 L1 최저). RGB 감독은 latent를 crop latent 쪽으로 묶지 못하고, 저주파 색에 해당하는 방향만 준다.
3. **이전 실험과 일관**: visual_recon_loss(DINOv2 + 자체 RGB head, B−A −0.44 n.s.), shared_detail_memory·detail_guided_decoder(색만 개선) 등 decoder 쪽 loss·디테일 추가가 색 외 attribute를 못 올린 패턴과 같다.
4. 한계: reader seed 0 단일·decoder seed 0 단일, 통계 검정 없음(사용자 지시). λ=0.1 하나만 사용(사전 등록 규칙). PACO는 원 해상도에서 crop, RAE는 224 입력이라 절대 수준이 기존 DINOv2 실험보다 낮다.

## 9. 절차 기록 (사전 등록 대비 변경)
- **학습 stream 재생성**: 기존 12k stream에 PACO val 935장 / test 2,896장 이미지가 포함 → 사용자 규칙(Dev/Test 이미지 제외)에 따라 PACO val/test 이미지를 뺀 pool(55,861장)에서 같은 sampler·seed 2027로 재생성(중복 0, results/data_overlap.json). 캐시 71 GB (perception_cache/feat).
- **Probe 간소화 (사용자 지시 18:30/18:35 KST)**: 모든 조건 R2 + R0, reader seed 0만 비교. 이미 끝난 A 12k seed 1·2, A 4k(seed 0, val), baseline seed 1·2 결과는 보존(노이즈 크기 계산에만 A seed 0–2 사용). B/C는 R2+R0 seed 0만 실행, 4k probe 미실행. Bootstrap/다중 seed 표(eval_rae.py final)는 실행하지 않음.
- Baseline probe를 GPU 7 → GPU 0으로 epoch 경계에서 이동(--resume, C 속도 1.51 → 0.72 s/step).
- 환경: research/envs/dinov2_ord_py310 (torch 2.7.1, transformers 4.52.1; 공식 pin과 다르나 S1에서 공식 클래스와 수치 일치 확인). TORCH_HOME = perception_cache/torch_hub.

## 10. 파일
code/: rae_common.py(공식 쌍·256 기하), models_rae.py, verify_pair.py, sanity.py, extract_cache.py, make_stream.py, calib.py, train_rae.py, data_rae.py, pixel.py,
probe_rae.py, recon_rae.py, s0_table.py, eval_rae.py(다중 seed bootstrap용, 미실행), run_*.sh.
perception_cache/: runs/{A,B,C}/ckpt_{4000,8000,12000}.pt, probe/, feat/(71 GB, 재생성 가능), init/, stream/, raev2_models/, torch_hub/.
