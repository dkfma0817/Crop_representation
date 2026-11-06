# E6 + Centered Direction Loss Pilot — 결과 보고서 (2026-11-06)

- **평가 범위:** PACO **Val**만 평가했다. Test는 쓰지 않았다.
- **새로 학습한 decoder:** E6-direction 하나뿐이다. Seed 0 하나이므로 CI에는 평가 이미지 불확실성만 반영되고 학습 seed 분산은 들어 있지 않다.
- **관련 파일:** 사전 설정은 [`protocol.md`](../../configs/e6dir/protocol.md), 자동 생성 표 전체는 [`tables.md`](tables.md)에 있다. 원자료는 `paco_results.csv`, `bootstrap_results.csv`, `results/recon/*.parquet`, `calib/calib.json`이다.

## 결론 먼저

1. **Centered Direction Loss는 위치별 feature 패턴을 조금 더 정확하게 학습시켰다. 개선 폭은 작다.**
   - PACO val의 centered direction cosine이 E6-original 0.3865에서 0.4035로 올랐다(**+0.017 [+0.0168, +0.0173]**). COCO dev에서도 +0.017이다.
   - 모든 object/part × 크기 구간에서 같은 방향으로 올랐다.
   - Spatial variance ratio는 0.38에서 0.44로 돌아와 Bicubic(0.437) 수준이 됐다. 기존 loss(L_old)와 pooled cosine은 거의 변하지 않았다.
   - 다만 0.40은 정답 C0 패턴(c_dir = 1)과 여전히 거리가 멀다.
2. **PACO 판별력은 개선되지 않았다.**
   - R2 val M0 Overall은 42.48이다. E6-original 대비 **−0.23 [−0.80, +0.34]**, Bicubic-only(42.71) 대비 **−0.23 [−0.91, +0.31]**로 둘 다 차이가 확인되지 않았다.
   - Bicubic 42.71을 넘지 못했다.
   - Obj D5–16(−0.29), Part D1–4(−0.08), M2(−1.16), R0(+0.07)도 E6-original과 차이가 확인되지 않았다.
   - **실용적 성공이 아니다.** (A) 표현은 개선됐지만 (B) downstream은 개선되지 않았다.

## 1. 실험 설정

| 항목 | 내용 |
|---|---|
| 조건 | Bicubic-only(IVC `probe_bic` 재사용), E6-original(xdec E6 checkpoint와 probe 재사용), **E6-direction(신규)** |
| 모델 | xdec E6를 그대로 썼다(Bicubic ROI query + full F336 cross-attention 2 block + Bicubic skip, zero-init head, d 256, 2.18M params). **E6-original과 같은 seed-0 canonical init에서 새로 시작했다**(이어서 학습하지 않음) |
| 입력 / teacher | frozen DINOv2-B/14 F336 final L12 + LN, 24×24×768 / C0 GT crop 재인코딩(frozen) |
| 학습 | COCO 12k stream, 64 ROI/step, AdamW 1e-4 (0.9, 0.95) wd 0.05, warmup 600 + cosine, clip 1.0, BF16, 12,000 step, GPU 2,3 DDP. E6와 같다 |
| 코드 | `src/e6dir/pretrain.py` = `src/xdec/pretrain.py`에 loss 항과 로그만 추가한 것(diff 64줄, `pretrain_vs_xdec.diff`) |
| Checkpoint | 3k/6k/9k/12k 중 **COCO-dev L_old 최소** → 12k (dev L_old 2.3898). E6-original도 12k (2.3908) |
| PACO probe | xdec와 같은 config(decoder 경로만 다름). `src.ivc.train`, PACO train(중복 6장 제외)에서 dev 1,000장을 떼어 epoch 선택(dev M0 최대), R2 + R0 seed 0, val 평가. 세 조건의 train/val 행 수, 이미지 수, step 수가 일치함을 확인했다 |

## 2. Loss 구현

- **L_old (변경 없음):** mean_ROI [MSE_w + 1.0 (1 − cos)_w], raw token, w = C0 valid coverage.
- **L_dir:** ROI마다 S와 T에서 각자 coverage 가중 공간 평균을 뺀 뒤, ROI 전체(위치 × 768)를 한 벡터로 보고 cosine을 구한다. `src/e6dir/loss.py`에 있다.
  - 식: c_dir = Σ w⟨S_c, T_c⟩ / (√Σ w‖S_c‖² √Σ w‖T_c‖² + 1e-8), L_dir = mean_{유효 ROI}(1 − c_dir).
  - 계산은 fp32이고, teacher는 detach한다.
- **최종 loss:** L_new = L_old + λ_dir · L_dir.
- **수치 검증(`smoke/loss_checks.json`, ALL_PASS):**
  - 같은 패턴이면 c = 1이다. 공통 offset을 더하거나(변화 8e-10) 양수 배율을 곱해도(5e-10) 변하지 않고, 음수 배율이면 −c가 된다.
  - √w-flatten 기준 구현과의 차이는 5e-10이다. Weight를 제곱했다면 1e-3만큼 달랐을 것이므로 weight가 제곱되지 않았음을 확인했다.
  - Padding cell은 영향이 없다(0).
  - 상수 teacher와 coverage 0인 ROI는 제외되고 gradient가 0이며 유한하다. Student가 상수일 때도 유한하다.
  - Teacher gradient는 None이다.
- **ROI 제외 규칙:** teacher centred energy e_T ≤ thr인 ROI는 L_dir에서만 뺀다. thr = 1e-3 × calibration median = 1.85e-3이다. 실제로 제외된 ROI는 calibration 1,024개, 학습 전체, COCO dev 모두 **0개**다(e_T 최소 0.73).

## 3. λ_dir 계산 결과 (학습 전 한 번, `calib/calib.json`)

- **방법:** stream의 처음 16 batch(64 ROI씩, 학습의 처음 16 step과 같음)를 seed-0 init E6에 넣었다. 출력 S를 fp32 leaf로 만들고 ‖∂L_old/∂S‖_F와 ‖∂L_dir/∂S‖_F를 구했다(학습과 같은 reduction).
- **Gradient norm:** median ‖∂L_old/∂S‖ = 9.81e-4, ‖∂L_dir/∂S‖ = 3.21e-4.
- **비율:** median 3.175, 범위 2.47–3.68, CV 0.09로 안정적이었다.
- **두 gradient 사이 cosine:** median 0.64. 두 loss의 gradient 방향이 상당 부분 겹친다.
- **λ_dir = 0.5 × 3.175 = 1.5877**로 정했고, 학습 내내 고정했다.
- **부작용 없음:** calibration은 별도 process에서 했고 optimizer update는 없다. 모델 checksum과 RNG가 변하지 않았음을 확인했다(True / True).
- PACO 결과로 λ를 정하거나 탐색하지 않았다.

## 4. 학습 로그 (COCO dev, 3k/6k/9k/12k)

| step | E6-original L_old | E6-direction L_old | E6-dir token cos | E6-dir L_dir | E6-dir L_new | E6-dir c_dir |
|---|---|---|---|---|---|---|
| 0 (init = Bicubic) | – | – | – | 0.641* | – | 0.359* |
| 3000 | 2.5071 | 2.5062 | 0.5720 | 0.5419 | 3.3666 | 0.4581 |
| 6000 | 2.4201 | 2.4197 | 0.5836 | 0.5312 | 3.2631 | 0.4688 |
| 9000 | 2.3953 | 2.3944 | 0.5868 | 0.5278 | 3.2325 | 0.4722 |
| 12000 | 2.3908 | **2.3898** | 0.5878 | 0.5271 | 3.2267 | **0.4729** |

\* step 0은 GPU smoke(dev 16장)의 값이다.

- **L_dir 추이:** 3k까지 대부분 줄었고(0.641 → 0.542), 그 뒤로는 0.527에서 거의 멈췄다.
- **L_old:** 방향 항을 추가해도 E6-original과 같은 궤적이다(차이 ≤ 0.001).
- **학습 시간과 메모리:** 526초, peak 1.94 GB.
- **곡선:** `figures/train_curves.png`.

## 5. 표현 지표 (PACO val 11,109 ROI, 같은 ROI 목록과 coverage)

| Method | Token MSE | Token Cos | **Centered Dir Cos** | Pooled Cos | Mean-feature MSE | Spatial Var / C0 |
|---|---|---|---|---|---|---|
| Bicubic | 3.567 | 0.350 | 0.239 | 0.474 | 1.947 | 0.437 |
| E6-original | **2.172** | 0.515 | 0.387 | **0.663** | **0.824** | 0.382 |
| E6-direction | 2.184 | **0.521** | **0.404** | 0.662 | 0.831 | **0.444** |

**c_dir(E6-direction) − c_dir(E6-original):**
- PACO val **+0.0170 [+0.0168, +0.0173]**, COCO dev **+0.0174 [+0.0167, +0.0180]** (image bootstrap 1,000, seed 0).
- CI가 매우 좁은 것은 거의 모든 ROI에서 같은 방향으로 조금씩 올랐기 때문이다.

크기별 c_dir(PACO val)은 다음과 같다. 모든 구간에서 +0.01 ~ +0.02이다.

| | obj D1–4 | obj D5–16 | obj D17+ | part D1–4 | part D5–16 | part D17+ |
|---|---|---|---|---|---|---|
| Bicubic | 0.108 | 0.179 | 0.479 | 0.046 | 0.156 | 0.460 |
| E6-original | 0.298 | 0.325 | 0.543 | 0.307 | 0.305 | 0.521 |
| E6-direction | 0.321 | 0.345 | 0.552 | 0.329 | 0.326 | 0.531 |

## 6. PACO 결과 (val, dev로 고른 epoch)

| Method | R2 epoch | **R2 M0 Overall** | Obj D5–16 | Part D1–4 | R2 M2 | R0 M0 |
|---|---|---|---|---|---|---|
| Bicubic-only | 6 | **42.71** | 46.47 | 40.10 | **44.05** | 37.04 |
| E6-original | 6 | 42.70 | **47.53** | **41.19** | 43.43 | 38.07 |
| E6-direction | 6 | 42.48 | 47.23 | 41.11 | 42.28 | **38.14** |

Paired image bootstrap 결과(B = 1,000, seed 0, 같은 draw):

| 대비 | R2 M0 Overall | R2 Obj D5–16 | R2 Part D1–4 | R2 M2 Overall | R0 M0 Overall |
|---|---|---|---|---|---|
| **E6-direction − E6-original** | −0.23 [−0.80, +0.34] | −0.29 [−0.83, +0.45] | −0.08 [−0.65, +0.48] | −1.16 [−2.02, +0.57] | +0.07 [−0.02, +0.15] |
| **E6-direction − Bicubic** | −0.23 [−0.91, +0.31] | +0.76 [−0.30, +1.73] | +1.01 [−0.03, +2.08] | −1.78 [−3.03, +0.57] | **+1.10 [+0.71, +1.46]** |
| E6-original − Bicubic (참고) | −0.01 [−0.73, +0.58] | +1.05 [−0.07, +1.88] | **+1.09 [+0.15, +2.10]** | −0.62 [−2.11, +1.16] | **+1.04 [+0.64, +1.38]** |

- **E6-direction vs E6-original:** R2에서 유의한 차이는 하나도 없다. 세부 attribute와 크기 bin은 tables.md에 있다.
- **E6-direction vs Bicubic:** Obj D17+만 −1.10 [−2.30, −0.09]으로 유의하게 낮다.
- **R0:** 두 E6 모두 Bicubic보다 약 +1.1 높다(평균 pooling reader). 하지만 R2에서는 그 이득이 나타나지 않는다. 이는 xdec 결과와 같은 패턴이다.

## 7. 두 질문에 대한 답

**Q1. Centered Direction Loss가 C0 teacher의 위치별 feature 변화 방향을 더 정확하게 학습시켰는가?**
- **그렇다. 다만 개선 폭이 작다.**
- c_dir이 +0.017 올랐고(유의, 모든 크기·level에서 같은 방향), 공간 분산비는 0.38에서 0.44로 회복됐다. 기존 feature loss는 나빠지지 않았다.
- 그러나 개선 폭은 E6-original이 Bicubic보다 얻은 폭(+0.148)의 약 11%다. 학습 중 L_dir은 3k step 이후 거의 정체했다(dev c_dir 0.458 → 0.473).
- 두 loss의 gradient cosine이 0.64로 높아서, 새 항은 기존 loss와 상당 부분 같은 방향을 밀었다.

**Q2. 그 변화가 PACO R2 판별 성능 개선으로 이어졌는가?**
- **아니다.**
- E6-original 대비 −0.23 [−0.80, +0.34], Bicubic 대비 −0.23 [−0.91, +0.31]이다. 42.71을 넘지 못했고 Obj D5–16에서도 개선이 없다.
- 이번 설정(E6, λ = 1.59 고정, seed 0)에서 **c_dir 0.39 → 0.40 정도의 개선은 R2 판별력을 바꾸지 못한다.**

## 8. 한계

- **Seed / λ:** seed 하나, λ 하나(규칙으로 정한 값)만 썼다. λ나 학습량을 바꿨을 때의 결과는 알 수 없다(sweep은 지시대로 하지 않음).
- **일반화:** E6 하나의 결과다. 다른 cross-attention decoder(E3/E4 direct 출력 등)에서도 효과가 없다고 일반화하지 않는다.
- **같은 seed 비교:** c_dir 차이의 CI가 매우 좁은 것은 같은 seed의 두 모델을 같은 ROI에서 비교했기 때문이다. 학습 seed를 바꿨을 때 이 차이가 얼마나 흔들리는지는 이 CI에 포함되지 않는다.
- **O2와의 비교:** svamp의 O2(C0 공간 패턴 oracle, +3.4)와 비교할 때, O2의 c_dir은 정의상 1이고 E6-direction은 0.40이다. 두 값 사이 어디서부터 R2 이득이 생기는지는 이번 데이터로 알 수 없다.

## 9. 다음 단계에 대한 판단 (실행하지 않음)

- **이 decoder의 loss 가중치를 더 조정하는 것은 우선순위가 낮다.**
  - 방향 loss를 직접 최적화해도 c_dir이 0.47(dev)에서 정체했다.
  - 같은 frozen F336 final-layer 입력과 이 크기의 E6에서 C0 공간 패턴을 예측할 수 있는 정도에 한계가 있어 보인다. 다만 확정한 것은 아니다.
- **입력 정보 쪽을 바꾸는 것이 더 근거 있는 후보다.**
  - 근거 1: svamp O2 결과는 "정확한 C0 공간 패턴이 있으면 C0 Full 수준(+3.4)"을 보여준다.
  - 근거 2: Full Multi-layer 입력(45.02)은 같은 프로토콜에서 decoder 없이도 Bicubic보다 높다.
  - 후보: decoder가 multi-layer feature나 encoder 내부 정보를 쓰도록 하는 것. 이는 이번 실험이 직접 검증한 것이 아니라 기존 결과를 종합한 판단이다.

## 10. 실행 상태와 QAT

| 항목 | 값 |
|---|---|
| 큐 | tmux `e6dir_q`, loss_checks → qat_pause → calib → smoke_gpu → train → probe → recon → qat_resume → analysis **모두 DONE** |
| QAT 중단 | 05:15 사용자 승인("지금 QAT 중단") → 05:20:19 SIGTERM, **step 31,088**, 재개 기준 **checkpoint-30000** → **1,088 step 유실** |
| 우리 GPU 사용 | 05:20:25 – 05:43:01 (≈ 23분 × H200 2장 ≈ 0.75 GPU-h). GPU 0,1은 사용하지 않았다 |
| QAT 재개 | 05:43:01 tmux **`qat_800m`**에서 checkpoint-30000으로 재개(saved cmdline `--resume`, env, cwd). 05:45:32에 step 30,056, loss 3.208로 **검증 OK** ([`qat_resume_status.md`](qat_resume_status.md)). 05:48 기준 step 30,172로 증가 중, GPU 2/3 100% |

- **수동 복구 명령:** `bash output/2026-11-06/e6_direction/scripts/recover_qat.sh`
- **덮어쓰지 않은 것:** 기존 xdec, IVC, svamp의 코드, checkpoint, 결과는 덮어쓰지 않았다. 새 코드는 `src/e6dir/`이고, 산출물은 `output/2026-11-06/e6_direction/`이다.
