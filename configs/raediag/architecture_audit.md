# R00 / CoordMLP / RAEv2 K=1 구조 확인 (코드 기준, 2026-11-05)

코드 위치는 repo 루트 기준이다. 원래 DINOv2 R00 코드는 `third_party/research/2026-10-26/multilevel_relation_decoder/code/`(이하 `MLR/`)에 있고, RAEv2용 이식본은 `src/models/coordmlp.py`다. 두 파일은 `from ... import G, PATCH, S` 한 줄만 다르다(`diff`로 확인: grid 24×24 → 16×16).

## 1. R00의 입력·출력 tensor
- 호출: `MLRDecoder.forward(F_fin_r, F_mid_r, b, P)` (`src/models/coordmlp.py:143`). R00 = `MLRDecoder("final")`라서 `F_mid_r`는 쓰지 않는다(RAE 학습에서는 `None`을 넘김, `src/engine/train.py:63-67`).
- 입력:

| tensor | shape | 내용 |
|---|---|---|
| `F_fin_r` | (R, N, 768) | `gather_roi(full_tokens, b)`(`src/rae/common.py:171`). **Student full-image** 최종층 token 중 ROI와 1-cell ring token. N = batch 안 최대 `n_ctx`, 빈자리는 0 + `pad` mask |
| `b` (geometry) | `tok` (R,N), `coords` (R,N,4), `pad` (R,N), `inside` (R,N), `valid_w` (R,G²), `bil_grid` (R,G²,2), `n_in`, `n_ctx` | `roi_geometry`(`src/rae/common.py:111`)의 결과를 `collate_geoms`로 묶은 것. 모두 box 좌표에서 계산하고 **teacher feature는 쓰지 않는다** |
| `P` | (G², 2) | 출력 query 위치 = crop canvas cell 중심의 정규화 좌표(`query_coords`) |

- 출력: (R, G², 768) = crop canvas 격자의 latent. DINOv2 R00은 G = 24(576 token), RAE 이식본은 G = 16(256 token, RAE latent와 같은 shape). Raw 최종층 공간이다(`mlp(h) * std_T + mean_T`, `coordmlp.py:118`).

## 2. ROI token 선택
- `roi_geometry`(`src/rae/common.py:111-141`): GT box를 full canvas로 옮기고, `j0 = floor(X0/PATCH)`, `j1 = ceil(X1/PATCH) − 1`(i도 같은 방식)로 box와 겹치는 full-grid cell 직사각형 `[i0..i1] × [j0..j1]`을 고른다. 여기에 **바깥 1-cell ring**을 더한다(grid 경계에서 clip, `:125-126`).
- `inside`는 ring을 제외한 box 내부 cell을 표시한다(`:128`). `n_in` = 내부 cell 수, `n_ctx` = ring 포함 수.

## 3. Local token: **있다, 출력 위치마다 4개**
- `CoordMLP(k=4)`(`coordmlp.py:76-82`), `neighbours()`(`:100-106`).
- 각 출력 query의 full-grid 위치 `u`(= `bil_grid`를 full-grid 단위로 바꾼 것)에서 **가장 가까운 4개 token**을 ROI+ring token 중에서 고른다(`pad` 제외, ring token도 후보).
- 선택한 token의 LN feature(4 × 768)와 상대 offset의 Fourier feature(4 × 2 × 22)를 이어 붙여 `l_local`로 보낸다(`:110-114`).
- 보간 가중치가 아니라 **kNN gather**다.

## 4. ROI mean token: **사용한다**
- `roi_summary(F_r, b)`(`coordmlp.py:43`) = `inside & ~pad` token의 비가중 평균(ring 제외)이다.
- `sum_ln`을 거친 뒤 `box_vector`(7차원)와 이어 붙여 `l_glob`으로 보낸다(`:115`).

## 5. ROI mean의 출처: **Student full feature**
- `MLRDecoder.forward`: `F = F_fin_r + Adapter(F_fin_r)`(R00, `coordmlp.py:145`), 이어서 `body(F, b, P)`.
- ROI mean은 adapter를 거친 student full-image token에서 계산한다. GT crop teacher에서 계산하지 않는다.
- **Teacher feature는 decoder 입력 어디에도 들어가지 않는다.**
- Teacher가 관여하는 곳은 두 군데뿐이다.
  - (a) 출력 역정규화 buffer `mean_T`, `std_T`(`coordmlp.py:91-97`): teacher crop latent의 **채널별 dataset 통계 상수**다(RAE: stream cache 처음 3,000 ROI, `src/data/dataset.py:100-107`; DINOv2 R00: `C.load_norm`). Sample마다 바뀌는 teacher 값이 아니다.
  - (b) Loss target.

## 6. 위치·box 정보
- `l_pos`(`coordmlp.py:116-117`): Fourier(P)(crop canvas 기준 query 위치) + Fourier(u/G)(full image 기준 절대 위치).
- `l_local`: token과 query의 상대 offset `rel`(full-grid 단위).
- `l_glob`: `box_vector`(`:48-72`) = [j0/G, i0/G, (j1+1)/G, (i1+1)/G, log bw, log bh, log aspect]. bw/bh는 crop letterbox coverage(`valid_w`, box 크기에서 계산한 기하 값)와 `coords`에서 구한다.
- 이 정보들은 `h = l_local + l_glob + l_pos`로 합산된다.

## 7. MLP는 출력 위치마다 독립
- `h`(R, G², hid)에 `mlp = GELU → Linear(1536,1536) → GELU → Linear(1536,768)`을 위치별로 적용한다(`coordmlp.py:87-118`).
- 출력 위치 사이에 attention이나 convolution 같은 상호작용은 **없다**.
- 위치 사이에서 공유되는 정보는 ROI 단위 `l_glob`(ROI mean + box)뿐이다.
- 마지막 Linear는 zero-init이라 초기 출력은 `mean_T`(상수)다.

## 8. R00과 기존 CoordMLP의 관계
- **같은 body에 adapter가 하나 더 붙은 구조다.**
  - R00 = `MLRDecoder("final")` = zero-init bottleneck Adapter(LN → 768 → 192 → GELU → 768, 297,408 params, `coordmlp.py:121-133`)를 입력 ROI token에 residual로 더한 것 + CoordMLP body(gen_decoders `models_g.CoordMLP`를 그대로 복사, 9,863,424 params).
  - 초기화 시점에는 adapter 출력이 0이라 R00 출력 = CoordMLP body 출력이다.
- DINOv2 R00은 body를 기존 `gen_decoders mlp_s0` checkpoint로 초기화했다(`MLR/train_mlr.py:30-38`). RAE A/B/C는 seed 0 새 init(`cache/init/coordmlp_init_s0.pt`)으로 학습했다(`src/engine/train.py:40-48`).
- 11-05 MSD/S-series decoder(`src/msd/common.py: MSDDecoder`)는 같은 body에 block 4/8 adapter와 보조 head를 더한 **다른 구조**다.

## 9. Gradient 경로 (RAE A/B/C, `src/engine/train.py:63-80`)
- `Z = m(gather_roi(full), None, g, P)`(bf16 autocast). `full`은 cache의 frozen encoder latent라 encoder에는 gradient가 없다.
- **Feature loss**: `MSE_w + 1.0·(1 − cos)_w`(raw 공간, w = teacher content coverage, `src/losses/feature.py`). Gradient는 Z → CoordMLP + Adapter로 간다. Teacher target(cache)에는 gradient가 없다.
- **Pixel loss**: `Z → rae.decode(Z)`(frozen ViT-XL decoder, `requires_grad=False`, eval, bf16) → unclamped RGB → `L1 + β·LPIPS-VGG`(content mask, GT crop canvas, `src/losses/pixel.py:pix_terms`) → gradient가 decoder를 통과해 Z → CoordMLP로 간다. Decoder와 LPIPS 가중치는 바뀌지 않는다(fingerprint 확인).
- 조건별 loss: A = L_feat, B = 0.1·L_pix, C = L_feat + 0.1·L_pix, β = 0.2(`configs/raev2_crop_recon/train/*.yaml`).
- DINOv2 R00(`MLR/train_mlr.py:44-46, 152`): teacher는 학습 중 frozen encoder로 계산하는 GT crop token(fp32)이다. Loss = 같은 feature loss(R00은 relation λ = 0). Pixel 경로는 없다.

## 10. RAE checkpoint와 정규화
- Encoder: torch.hub `facebookresearch/dinov2` `dinov2_vitb14_reg`(register 4, **최종 LN affine 없음**, pos-embed를 16×16으로 resample). 전처리: 256 letterbox uint8 → `/255` → ImageNet mean/std → `F.interpolate(224, bicubic)`(`third_party/RAEv2/src/encoders/vision_encoder.py:195-201`). 출력 raw latent (B, 256, 768), token norm은 √768 ≈ 27.7이다.
- Decoder: `cache/raev2_models/stage1/imagenet/dinov2b-k1/decoder.pt`(sha256 aabfee19…, ViT-XL 415M, 입력 256 token × 768 → 256×256 RGB). 공식 `stats.pt`는 위치별 mean/var (768,16,16)이다.
- 공식 `RAE.decode(z)`는 z = (raw − mean)/√(var + 1e-5)를 먼저 역정규화한다. 그래서 wrapper `FrozenRAE.decode(raw)`(`src/rae/common.py:237`)는 정규화된 latent의 공식 decode와 수치적으로 같다(sanity S1: decode max |diff| 2.9e-5, encode 4.8e-5; `experiments/raev2_crop_recon/results/sanity.json`).
- Latent noise는 꺼져 있다(noise_tau 0).

## 이번 진단에 쓸 수 있는 기존 checkpoint
| 모델 | 이 서버 | 사용 |
|---|---|---|
| RAEv2 encoder / decoder / stats | 있음 | A/B/C/D 전부 |
| A Feature-only CoordMLP (12k) | **없음**(원래 서버 `perception_cache/runs/A`에만 있었음; `cache/runs/` 비어 있음) | 불가. 새로 학습하지 않는다 |
| C Feature+Pixel CoordMLP (12k) | **없음** | 불가 |
| B Pixel-only CoordMLP (12k) | 있음(`output/2026-11-04/B/ckpt_12000.pt`, 이 서버에서 같은 설정으로 학습. R00 구조, RAE latent와 호환) | 조건 MB로 cycle 진단에 포함 |
| OF100_pix (100-ROI pixel overfit) | 있음 | 과적합 sanity용이라 제외 |
| DINOv2 R00 / R10 (timm lvd142m 공간) | 없음 | RAE decoder와 latent 공간이 달라(register 없음, LN affine 있음, 24×24) **연결하지 않는다** |
