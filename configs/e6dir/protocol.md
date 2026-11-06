# E6 + Centered Direction Loss Pilot: 프로토콜 (2026-11-06, 학습 결과를 보기 전에 기록)

**목적:** 기존 feature matching loss가 C0 teacher의 위치별 feature 패턴(centred 방향)을 충분히 학습시키지 못했는지 확인한다. 아키텍처, 입력, teacher, 학습량은 바꾸지 않는다. 새로 학습하는 decoder는 **E6-direction 하나**다.

## 고정 조건 (= xdec E6)
- **Encoder와 입력:** frozen DINOv2-B/14, F336 final L12 + LN, 24×24×768을 이미지당 한 번 추출한다.
- **Teacher:** C0 GT crop을 같은 frozen encoder로 다시 인코딩한 것이다. 둘 다 frozen이다.
- **Decoder:** E6(Bicubic ROI query + full feature cross-attention 2 block + Bicubic skip, zero-init head, d 256)이다. **E6-original과 같은 seed-0 canonical init에서 새로 시작한다**(E6 checkpoint에서 이어서 학습하지 않음).
- **학습:** COCO 12k stream(64 ROI/step), AdamW 1e-4 (0.9, 0.95) wd 0.05, warmup 600 + cosine, clip 1.0, BF16, 12,000 step, GPU 2,3 DDP. Coverage 처리(`valid_w`)도 같다.
- **코드:** `src/e6dir/pretrain.py`는 `src/xdec/pretrain.py`를 그대로 복사하고 loss 항과 로깅만 추가했다(`pretrain_vs_xdec.diff`).

## Loss
- **L_old (변경 없음):** mean_ROI [MSE_w + 1.0 (1 − cos)_w], raw token, w = valid_w. 가중치와 reduction은 기존과 같다.
- **L_dir:** ROI마다 S와 T가 각자 coverage 가중 공간 평균을 뺀 뒤, ROI 전체(위치 × 768 channel)를 하나의 벡터로 보고 가중 cosine을 계산한다.
  - 식: c_dir = Σ_i w_i⟨S_c,i, T_c,i⟩ / (√Σ w‖S_c‖² · √Σ w‖T_c‖² + 1e-8), L_dir = mean_{valid ROI}(1 − c_dir).
  - w는 모든 합에 한 번씩만 들어간다(제곱하지 않음). √w-flatten 기준 구현과의 차이는 5e-10이다.
  - **Teacher 쪽:** detach되어 있어 gradient가 흐르지 않는다.
  - **ROI 제외 기준:** teacher centred energy e_T = Σ w‖T_c‖²/(768 Σ w)가 thr 이하인 ROI는 **L_dir에서만** 뺀다. L_old에서는 빼지 않는다. Student norm으로는 제외하지 않는다.
- **최종 loss:** L_new = L_old + λ_dir · L_dir, λ는 학습 내내 고정한다.
- **수치 검증:** `smoke/loss_checks.json`에 있고 ALL_PASS다.
  - 같은 패턴이면 c = 1이다. 공통 offset이나 양수 배율에 대해 c가 변하지 않는다. 음수 배율이면 −c가 된다.
  - Padding cell은 결과에 영향을 주지 않는다.
  - 상수 teacher와 coverage 0인 ROI는 제외되고, 그때 forward/backward가 유한하다. Student가 상수여도 유한하다.
  - Teacher grad는 None이다.

## λ_dir calibration (학습 전 한 번만, `calib/calib.json`)
- **데이터와 모델:** stream의 처음 16 row(= 학습의 처음 16 step, 64 ROI씩)를 seed-0 init E6에 넣는다. 출력 S를 fp32로 다시 leaf로 만든 뒤 ‖∂L_old/∂S‖_F와 ‖∂L_dir/∂S‖_F를 계산한다(training reduction 그대로).
- **결과:** λ = 0.5 × median ratio = **1.5877**.
  - Ratio median 3.175, 범위 2.47–3.68, CV 0.09.
  - g_old median 9.8e-4, g_dir median 3.2e-4, gradient cos median 0.64.
  - thr = 1e-3 × median e_T = 1.85e-3이고, 1,024 ROI 중 제외는 0이다.
- **부작용 없음:** calibration은 별도 process에서 돌렸고 optimizer update는 없다. 모델 checksum과 RNG 모두 변하지 않았다.
- **사용 범위:** PACO 결과는 λ를 정하는 데 쓰지 않았다.

## Checkpoint / 평가
- **Decoder checkpoint:** 3k/6k/9k/12k 중 **COCO-dev L_old 최소**를 쓴다(xdec와 같은 규칙). L_new, c_dir, PACO는 선택에 쓰지 않는다.
- **PACO probe:** xdec E6와 같은 config(경로만 다름). `src.ivc.train`, PACO train(중복 6장 제외)에서 dev 1,000장을 떼어 epoch 선택, 10 epoch, R2 + R0 seed 0, dev M0 최대 epoch을 쓴다. **Val만** 평가하고 test는 평가하지 않는다.
- **재사용:** Bicubic-only = IVC `probe_bic`, E6-original = xdec `probe_E6` (checkpoint `xdec/checkpoints/E6/selected.pt`, 12k). 둘 다 같은 프로토콜이다.
- **Primary:** R2 val M0 Overall.
- **Secondary:** Obj D5–16, Part D1–4, M2 Overall, R0.
- **Bootstrap 대비:** E6-direction − E6-original, E6-direction − Bicubic, E6-original − Bicubic(참고). Paired image bootstrap 1,000회, seed 0. Seed 0 하나뿐이라 학습 seed 분산은 포함하지 않는다.
- **표현 지표(PACO val 전체, COCO dev):** token MSE, token cos, **centred direction cos(학습과 같은 정의)**, pooled cos, mean-feature MSE, spatial variance ratio, centred diff.

## 판정
- **A. Representation:** E6-direction의 c_dir이 E6-original보다 높은가(같은 ROI, 같은 coverage, image bootstrap CI)?
- **B. Downstream:** R2 M0가 E6-original보다 높은가, 그리고 Bicubic 42.71을 넘는가(paired CI)?
- c_dir만 좋아지고 PACO가 그대로면 성공이 아니다.
- E6 결과를 모든 cross-attention decoder로 일반화하지 않는다.
- λ sweep, multi-seed, 구조 변경은 하지 않는다.
