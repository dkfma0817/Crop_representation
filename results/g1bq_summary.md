# G1-cross with Bilinear-initialized Decoder Queries (G1-bq) — Encoder Only 평가

- 학습: CLIP ViT-L/14@336, original CLIP 초기화, G1-cross와 동일한 COCO 12k stream / crop teacher / loss(L_roi + 1.11·L_detail + 0.5·L_global) / optimizer / 12,000 step / EMA, seed 0. 바꾼 것은 decoder query뿐이다: q0 = LN(B + pos) + q_grid, B = 학생 encoder token의 bilinear ROI 샘플(24×24 crop grid, letterbox 정렬).
- 평가: decoder를 제거하고 Encoder Only + R2/head(기존과 같은 yaml, 10 epoch)를 학습했다. Checkpoint는 사전 고정한 Dev AP 규칙((Obj+Part)/2 최대)으로 골랐고, 그 checkpoint 하나로 공식 Test 전체(9,443 images / 131,899 annotations, GT box + category oracle)를 평가했다. Test 결과를 보고 바꾼 설정은 없다.
- \* = image-level paired bootstrap(B=1000) 95% CI가 0을 포함하지 않음.

## 1. Dev AP 곡선과 checkpoint 선택 (선택: **epoch 6**)

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| g1bq | 52.4/53.5 | 55.2/57.4 | 56.6/57.8 | 56.4/59.2 | 57.5/59.5 | 57.7/59.5 | 57.4/57.6 | 55.8/56.7 | 54.9/57.2 | 55.0/57.8 |
| g1cross_enc | 53.1/54.1 | 55.8/57.1 | 57.3/57.2 | 58.9/59.4 | 58.2/59.4 | 58.7/59.2 | 58.2/58.4 | 56.3/58.2 | 56.0/58.8 | 56.2/58.2 |

## 2. PACO-LVIS Official Test — GT Box + Category Oracle

| Model | Obj AP | Obj AP col | Obj AP pat | Obj AP mat | Obj AP ref | Part AP | Part AP col | Part AP pat | Part AP mat | Part AP ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original CLIP (ep6) | 36.51 | 28.80 | 40.55 | 31.62 | 45.08 | 35.63 | 31.94 | 38.31 | 30.79 | 41.46 |
| MaskEmbed (ep6) | 39.32 | 32.03 | 44.24 | 31.06 | 49.93 | 38.17 | 36.62 | 39.48 | 32.90 | 43.70 |
| G1-pool (ep6) | 36.37 | 30.10 | 38.17 | 30.31 | 46.92 | 36.16 | 34.21 | 36.28 | 32.04 | 42.11 |
| G1-cross Encoder Only (ep6, M0 pick) | 39.21 | 32.08 | 44.11 | 31.32 | 49.32 | 37.47 | 35.91 | 38.60 | 31.89 | 43.50 |
| G1-cross Encoder Only (ep4, Dev AP pick) | 37.29 | 30.83 | 40.62 | 30.10 | 47.61 | 36.87 | 34.20 | 39.36 | 30.54 | 43.38 |
| **G1-bq Encoder Only (ep6, Dev AP pick)** | 39.00 | 32.35 | 43.35 | 31.17 | 49.14 | 37.23 | 35.69 | 38.17 | 31.97 | 43.09 |

Test Δ (paired bootstrap 95% CI):

| Contrast | Obj AP | Part AP | Obj mat | Obj pat | Part pat | Part col |
|---|---|---|---|---|---|---|
| G1-bq − G1-cross Enc (same Dev-AP rule, ep4) | +1.71 [+0.72, +2.88]\* | +0.36 [-0.17, +1.43] | +1.07 [-0.19, +3.26] | +2.73 [-0.06, +5.46] | -1.18 [-1.83, +2.09] | +1.49 [+0.56, +2.15]\* |
| G1-bq − G1-cross Enc (ep6, reported) | -0.20 [-1.13, +0.83] | -0.24 [-0.89, +0.39] | -0.15 [-1.34, +2.01] | -0.76 [-3.43, +2.09] | -0.42 [-1.98, +1.46] | -0.21 [-0.76, +0.45] |
| G1-bq − MaskEmbed | -0.31 [-1.54, +0.87] | -0.94 [-1.69, +0.13] | +0.11 [-1.41, +2.32] | -0.89 [-4.66, +1.85] | -1.30 [-2.87, +0.94] | -0.92 [-1.82, +0.03] |
| G1-bq − Original CLIP | +2.49 [+1.36, +3.82]\* | +1.61 [+1.11, +2.89]\* | -0.45 [-1.85, +2.54] | +2.80 [-1.19, +5.36] | -0.14 [-1.37, +2.73] | +3.76 [+2.93, +4.75]\* |

## 3. 공식 Val (2,410 images)

| Model | Obj AP | Obj AP col | Obj AP pat | Obj AP mat | Obj AP ref | Part AP | Part AP col | Part AP pat | Part AP mat | Part AP ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original CLIP (ep6) | 50.12 | 40.72 | 52.37 | 51.73 | 55.67 | 52.07 | 48.21 | 46.14 | 55.19 | 58.75 |
| MaskEmbed (ep6) | 48.77 | 45.05 | 48.67 | 48.71 | 52.65 | 53.84 | 50.95 | 51.40 | 54.57 | 58.44 |
| G1-cross Encoder Only (ep6) | 52.08 | 46.23 | 52.48 | 52.03 | 57.59 | 54.03 | 51.35 | 52.15 | 55.60 | 57.04 |
| G1-cross Encoder Only (ep4, Dev AP pick) | 50.55 | 45.12 | 49.49 | 50.22 | 57.34 | 53.45 | 49.80 | 54.75 | 52.92 | 56.32 |
| G1-bq Encoder Only (ep6) | 51.27 | 45.69 | 52.46 | 51.57 | 55.38 | 54.30 | 53.02 | 53.00 | 54.04 | 57.12 |
| G1-bq Encoder Only (ep10, fixed) | 50.04 | 42.59 | 53.45 | 50.67 | 53.46 | 53.42 | 51.03 | 54.53 | 51.61 | 56.52 |

| Contrast | Val Obj | Val Part | Dev Obj | Dev Part |
|---|---|---|---|---|
| G1-bq − G1-cross Enc (Dev-AP picks) | +0.73 [-1.41, +3.13] | +0.85 [-1.11, +2.30] | -1.16 [-3.69, +2.00] | +0.08 [-1.78, +2.58] |
| G1-bq − G1-cross Enc ep6 | -0.81 [-2.40, +0.83] | +0.26 [-1.26, +1.19] | -1.00 [-3.44, +2.07] | +0.29 [-1.37, +2.15] |
| G1-bq − MaskEmbed | +2.50 [+0.14, +4.44]\* | +0.46 [-1.33, +1.94] | -2.38 [-4.80, +1.12] | -1.77 [-3.52, +1.45] |

## 4. Encoder 학습 중 region feature 변화 (COCO dev, EMA; G1-cross vs G1-bq)

| step | model | ROI pooled cos (vs crop teacher) | in-ROI token cos (vs original) | out-of-ROI token cos | CLS cos | decoder token cos | decoder pooled cos |
|---|---|---|---|---|---|---|---|
| 0 | G1-cross init | 0.7428 | 1.0000 | 1.0000 | 1.0000 | 0.0073 | 0.0117 |
| 3000 | G1-cross ema | 0.9422 | 0.7210 | 0.7711 | 0.9885 | 0.6794 | 0.9666 |
| 6000 | G1-cross ema | 0.9459 | 0.6840 | 0.7268 | 0.9850 | 0.6939 | 0.9716 |
| 9000 | G1-cross ema | 0.9452 | 0.6761 | 0.7142 | 0.9846 | 0.6974 | 0.9726 |
| 12000 | G1-cross ema | 0.9450 | 0.6748 | 0.7123 | 0.9846 | 0.6979 | 0.9728 |
| 12000 | G1-cross student | 0.9426 | 0.6643 | 0.6962 | 0.9839 | 0.7004 | 0.9734 |
| 0 | G1-bq init | 0.7428 | 1.0000 | 1.0000 | 1.0000 | 0.0015 | 0.0025 |
| 3000 | G1-bq ema | 0.9427 | 0.7204 | 0.7690 | 0.9880 | 0.6812 | 0.9659 |
| 6000 | G1-bq ema | 0.9468 | 0.6838 | 0.7266 | 0.9844 | 0.6941 | 0.9715 |
| 9000 | G1-bq ema | 0.9463 | 0.6759 | 0.7141 | 0.9840 | 0.6971 | 0.9726 |
| 12000 | G1-bq ema | 0.9462 | 0.6746 | 0.7120 | 0.9840 | 0.6975 | 0.9727 |
| 12000 | G1-bq student | 0.9434 | 0.6641 | 0.6949 | 0.9833 | 0.6996 | 0.9733 |

## 5. 검증

- Bilinear query가 학생 encoder token에서 생성됨: dB/dF ≠ 0 (True); L_detail → encoder token gradient: query 경로 5.25e-03, key 경로 9.23e-04; encoder parameter gradient norm.weight 0.038, blocks.23.mlp.fc2.weight 0.523, blocks.0.attn.qkv.weight 0.570
- Crop grid ↔ ROI 좌표 정렬: src.decstudy.checks (96 ROI 최대 오차 3.7e-5 px, 실제 픽셀 상관 0.978) 와 같은 bilinear_roi 사용
- 학습 parameter: decoder 29,155,328 vs G1-cross 29,153,280 (+ln_qin 2,048); 공유 parameter 초기값 동일 (True); 추가 연산 ≈ bilinear sampling 8x576x1024x4 MACs = 0.019 GMAC / step
- 추론: decoder 미사용(Encoder Only), 이미지당 encoder 1회, ROI crop 재인코딩 없음
- 학습: world 2 (G1-cross는 world 1; 64 ROI/step 동일, gradient 전역 합산), 12000 step, 2.31 h

주요 비교 중 '학습 조건을 맞춘 MaskEmbed'와 'Mask + Crop View 결합 모델'은 이 세션에 결과가 없어 포함하지 못했다 (기존 MaskEmbed 공식 checkpoint 행만 참고로 표기).
