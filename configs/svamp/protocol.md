# Spatial Variance Amplification & C0 Spatial Oracle Pilot: 실행 전 프로토콜 (2026-11-05, 결과 전에 고정)

새 decoder는 학습하지 않는다. xdec의 Bicubic tensor B를 학습 없이 변환하고, 조건마다 PACO reader만 학습한다.
- **고정 규칙:** 사전에 고정한다.
- **평가 범위:** Val만 평가한다. Test는 실행하지 않는다.
- **Reader seed:** 0 하나만 쓴다.

## Feature
- **Backbone과 B:** frozen DINOv2-B/14(timm lvd142m), F336 final L12 + final LN, 24×24×768. B는 `src.xdec.geometry.bicubic_roi`로 만든다. 이는 xdec Bicubic-only, IVC `bicubic`과 같은 호출이다(`grid_sample` fp32, `bil_grid`, border, align_corners=False).
- **T (C0 teacher):** 같은 frozen encoder로 GT crop의 336 letterbox를 인코딩한다. xdec/IVC C0와 같다.
- **가중치:** w = C0 valid coverage(`valid_w`, R0/R2 reader의 `cov`와 같은 값, 차이 0으로 확인). 유효 token은 M = w > 0이다. C0 letterbox padding은 ROI로 보지 않는다.
- **변환 수식:** μ_B = Σ w B / Σ w (token 축 평균, 768 channel 유지), μ_T도 같은 방식.
  - V0 = B (기존 Bicubic)
  - V_k = μ_B + k(B − μ_B), k = 1.25 / 1.5 / 2.0
  - **O2 = μ_B + (T − μ_T) — Oracle.** 각 split(train, dev, val) 이미지 자신의 C0 crop을 학습과 평가 모두에 쓴다. 실행 가능한 method가 아니다.
- **Invalid cell 정책:** 모든 조건에서 B 값을 그대로 둔다.
- **Spatial variance:** xdec recon 정의와 같다. ROI마다 sv(x) = Σ w mean_c (x − μ_x)² / Σ w, 비율은 sv(x) / sv(T)를 ROI별로 구해 평균한다. 참고로 Σ sv / Σ sv(T)(ratio of means)도 함께 보고한다.

## 검증 (CPU smoke: 12장 100 ROI → 전체 val 11,109 ROI에서 다시 확인)
- **평균 보존:** |μ(V_k) − μ_B| ≤ 4e-6, O2도 같다.
- **Invalid cell:** 변화 0.
- **R0 입력:** coverage 평균이 V0와 ≤ 4e-6 차이다.
- **coverage 일치:** cov_c == valid_w.
- **분산 scaling:** sv(V_k)/sv(V0) = k²로 정확하다. sv(O2)/sv(T) = 1이다.
- **R0 reader:** coverage 가중 평균 → level별 z-score → Linear이고, pooling 전에 token별 비선형 변환이 없다. 그래서 V0–V3와 O2의 R0 입력은 수학적으로 V0와 같다. **R0는 다시 학습하지 않고 V0(IVC probe_bic R0) 결과를 쓴다.**

## Reader / 평가
- **Probe:** `src.ivc.train`(xdec/IVC와 같은 프로토콜).
  - 데이터: PACO train(중복 6장 제외)에서 PACO dev 1,000장(rng 0)을 epoch 선택용으로 떼어낸다.
  - 학습: R2(layers [12]), 10 epoch, ≈128 ROI/step, hflip, masked BCE, AdamW 3e-4 / 1e-3, init seed 7.
  - Epoch 선택: dev M0 Overall 최대.
  - 평가: val.
- **재사용(같은 tensor, 같은 프로토콜):**
  - V0: IVC `probe_bic`(R2, R0). IVC는 R0와 R2를 같은 process에서 학습했고, R2만 학습한 run과 같은 값이 나왔다(MSD 대비 재현 확인 42.30 = 42.30).
  - C0 Full Oracle: IVC `probe_c0`.
- **참고:** xdec R00 / E3 / E6.
- **Primary:** val M0 Overall, Object D5–16 (R2).
- **Secondary:** val M2 Overall, Object D1–4 / D17+, Part D1–4 / D5–16, Color / Material / Pattern, R0.
- **크기 bin:** manifest `n_in`(F336 grid). IVC/xdec와 같다.
- **Bootstrap:** paired image bootstrap 1,000회, seed 0, 같은 draw.
  - 주요 대비: V1−V0, V2−V0, V3−V0, O2−V0, O2−V2.
  - 추세 보조: V2−V1, V3−V2, C0 Full − O2.
- **Feature 통계(PACO val 전체):** token norm, pooled norm, token cos, pooled cos, sv ratio, centred diff, mean diff, centred cos. 전체와 object/part × D-bin별로 낸다.

## 해석 규칙
- CI가 0을 포함하면 차이가 확인되지 않은 것으로 본다.
- 분산 증폭은 norm과 입력 분포도 바꾼다(feature 통계 참조). 그래서 PACO 차이를 곧바로 "공간 정보 증가"의 인과 효과로 해석하지 않는다.
- O2는 oracle이다. O2 − V2는 "같은 분산 크기에서 실제 C0 공간 패턴을 주는 것"의 진단이다.
- xdec decoder의 실패 원인을 이 실험으로 확정하지 않는다.
- 후속 decoder 학습이나 실험은 자동으로 추가하지 않는다.

## GPU / QAT
- 물리 GPU 2, 3만 쓴다. GPU 0, 1의 작업은 건드리지 않는다.
- QAT 중단은 **이번 실험에 대한 사용자의 별도 승인**이 있을 때만 한다. 승인 없이는 QAT와 GPU를 같이 쓰는 방식(share) 또는 대기 중 하나로 진행한다.
