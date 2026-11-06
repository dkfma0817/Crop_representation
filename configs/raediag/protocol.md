# RAE Reconstruction 진단 — 프로토콜 (2026-11-05, 실행 전 고정)

목적: RGB Reconstruction Fidelity와 Feature(Latent) Representation Fidelity를 구분해, (1) full-image encoding이 작은 ROI의 정보를 잃는지, (2) ROI sampling feature가 기존 RAE decoder에 맞지 않는지에 대한 관찰을 얻는다. **학습은 없다.** 새 decoder, projection, loss 변경도 하지 않는다. PACO Test는 쓰지 않는다. 결과는 `output/2026-11-05/rae_diag/`, 코드는 새 파일 `src/raediag/*`, 고정 샘플은 `configs/raediag/sample_rois.json`이다.

## 모델과 공간 (`architecture_audit.md` 10번)
- 기존 RAEv2 K=1 쌍(`src/rae/common.FrozenRAE`)을 그대로 쓴다.
  - Encoder: `dinov2_vitb14_reg`, 입력 = 256 letterbox uint8 canvas → 공식 전처리(ImageNet 정규화, bicubic 224). 출력 = raw latent (256 token × 768).
  - Decoder: ViT-XL, raw latent 입력 → 256×256 RGB. 공식 per-position stats 정규화와 수치적으로 같다(sanity S1).
- Encoder는 fp32(TF32 off), decoder와 LPIPS는 bf16 autocast. 기존 `src/eval/recon.py`와 같다.
- **Latent 호환성**:
  - A, B: encoder가 직접 만든 latent라 decoder 입력 사양 그대로다.
  - C/D/E/F: 같은 raw 공간, 같은 16×16 grid, 같은 768 채널의 값을 재배치·보간한 것이다. 정의는 호환되지만 분포는 다를 수 있어, decoder 입력 분포를 따로 측정한다(token norm, 공식 stats로 정규화한 z², |z| > 3 비율, 인접 token cosine).
  - DINOv2 timm 공간의 R00/R10이나 IVC feature는 RAE decoder에 **연결하지 않는다**(공간이 다름).

## 데이터
- PACO **val** 전체: 11,109 ROI / 2,051 이미지. Manifest는 `src/readout/common.load_data`(IVC와 같음), flip 없음.
- 기존 RAEv2 실험은 hval(COCO)과 PACO test-primary를 썼다. 이번에는 사용자 지시대로 PACO val을 쓴다(학습이 없으므로 split 누수 문제는 없다).
- 크기 bin(주): manifest `n_in`(F336 24×24 grid) D1–4 / D5–16 / D17+ × object/part. 기존 RAE 보고서의 주 bin과 같은 정의다. 보조: RAE 16×16 grid의 `n_in`.
- Smoke test: 고정 58 ROI(무작위 40 + 시각화 18, seed 0).

## 조건 (모든 RGB는 GT crop canvas = 원본 픽셀 GT crop의 256 letterbox와, content 픽셀에서만 비교)
| | 경로 |
|---|---|
| **A** | GT crop canvas → encoder → T (= C0 latent) → decoder. Decoder 기준선 |
| **B** | full canvas → encoder → Zf → decoder → full RGB 256 → GT ROI 영역을 crop canvas pixel grid로 재샘플 |
| **B_in** (참고) | 같은 재샘플을 full **입력** canvas에 적용(encoder/decoder 없음). Full-image 입력 해상도 자체의 한계 |
| **C** | Zf → crop grid bilinear sampling(`roi_reader.bilinear`, G = 16) → decoder |
| **D** | 같은 grid, `mode='bicubic'` → decoder |
| **E** (진단, T 사용) | T의 content block을 그 ROI의 full-grid 크기(round(h/s) × round(w/s) cell, s = crop.sx/full.sx)로 area-pool한 뒤 bilinear로 원래 크기로 되돌림. Padding cell은 T 그대로 → decoder. "C0 latent의 평균·분포는 유지하고, 공간 해상도만 full grid 수준으로 낮추면?" |
| **F** (진단, T 사용) | C의 content cell에서 coverage 가중 평균만 T의 평균으로 바꿈(C − mean(C) + mean(T)) → decoder. "Bilinear latent의 평균 표현만 C0로 바꾸면?" |
| **MB** | 기존 Pixel-only CoordMLP(R00 구조, 12k) 출력 → decoder. Feature-only / Feature+Pixel checkpoint는 이 서버에 없어 제외 |

- E와 F는 GT crop latent를 쓰는 **분석용 대조군**이다. 사용 가능한 방법이 아니다.
- **B / B_in 재샘플**: crop canvas pixel 중심 (x + .5) → 원본 px(crop letterbox 역변환 + box 원점) → full canvas 좌표에서 bilinear `grid_sample`(align_corners=False). GT crop canvas는 원본 픽셀을 PIL BICUBIC으로 resize한 것이고, crop scale은 full scale 이상이라 B/B_in은 항상 확대(또는 같은 크기) 방향이다.
  - 차이 기록: RGB 재샘플은 bilinear(antialias 없음), GT는 PIL bicubic(축소 시 antialias). Feature sampling의 bilinear/bicubic은 latent 격자에서 한다.

## 지표
- **RGB**(clamp [0, 1], content 픽셀): L1, PSNR, LPIPS-VGG(content 밖은 GT로 합성, 기존과 같음), SSIM(채널별 11×11 gaussian, σ 1.5, 창 전체가 content 안인 픽셀 평균), hp_corr(기존 정의: high-pass 휘도의 Pearson, GT 정렬 디테일), dE_mean(평균 색 CIE76), hf_ratio.
- **재인코딩**: 복원 crop canvas(content = round(clamp·255), padding = GT 파이프라인과 같은 ImageNet 평균색) → 같은 encoder → R_x. T와 비교한다(content cell, coverage 가중): token cosine, token MSE, pooled cosine, pooled L2, spatial variance 비율, self-similarity |ΔS|.
  - 입력 latent가 있는 조건(A, C–F, MB)은 입력 latent vs T, 그리고 cycle(R_x vs 입력 latent)도 기록한다.
- **차이**: 같은 ROI의 조건 간 차이, val 이미지 단위 cluster bootstrap 1,000회, seed 0.

## 정렬 검증 (`alignment_checks/`)
1. Feature grid: 독립 수식 vs `bil_grid`(전 ROI).
2. 합성 좌표 latent(bilinear / bicubic).
3. 합성 좌표 canvas로 RGB 재샘플 검증.
4. B_in pixel 정합: ±2 px shift L1 scan.
5. Feature 정합: bicubic grid를 ±8 / ±16 px shift해 T와의 cosine scan(GPU).
6. 고정 ROI 12개 그림(IVC 정렬 ROI 재사용).

## 시각화
level × D-bin마다 3개(seed 0, 실행 전 고정). 행 = full 입력 | GT | A | B_in | B | C | D | E | F | MB, 각 칸에 LPIPS와 재인코딩 token cosine을 표시한다.

## 실행
GPU 2,3(사용자 QAT 강제 중단, 끝나면 `--resume`으로 재개). `python -m src.raediag.run --shard k --nshard 2`, `src.raediag.align`, `src.raediag.report`.
