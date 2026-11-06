# E6 Multi-layer Memory Pilot — 결과 보고서 (2026-11-06)

**핵심 질문:** 같은 E6 decoder에서 K/V memory에 실제 중간 layer(L4/L8)를 넣는 것이 최종층(L12)만 넣는 것보다 유용한가?

- **평가 범위:** PACO **Val**만 평가했다. Test는 실행하지 않았다.
- **Seed:** 0 하나다. CI는 평가 이미지의 불확실성만 반영하고 학습 seed 분산은 포함하지 않는다.
- **관련 파일:** 사전 설정은 [`protocol.md`](../../configs/e6ml/protocol.md), 자동 생성 표 전체는 [`tables.md`](tables.md)에 있다. 원자료는 `paco_results.csv`, `bootstrap_results.csv`, `results/recon/*.parquet`, `smoke/smoke.json`이다.

## 결론 먼저

1. **Q1. 중간 layer 제공 자체의 효과는 약하고 일관되지 않다.**
   - R2 M0 Overall에서 B − A(parameter를 맞춘 L12 대조군) = **+0.72 [+0.09, +1.17]**로 유의하다.
   - 그러나 대조군 A가 E6-original보다 −0.59 [−1.04, −0.04] 낮았다. B − E6-original은 **+0.13 [−0.46, +0.66]**으로 차이가 확인되지 않았다.
   - A와 E6-original은 같은 init에서 시작하고 L12 projection만 2개 더 있는 사실상 같은 모델이다. 그런데도 두 모델 사이에 0.6 정도 차이가 났으므로, B − A의 +0.72는 seed 수준의 변동과 구분하기 어렵다.
   - R0에서는 B가 A와 E6-original 모두보다 +0.32 / +0.34로 작지만 유의하게 높았다. A − E6-original은 +0.02다.
   - Obj D5–16(R2)에서 B − A는 +0.68 [−0.03, +1.31]로 경계에 있다.
2. **Q2. C0 feature 복원은 아주 조금 개선됐다.**
   - B − A: token cos +0.004, token MSE −0.010, centered direction cos +0.0012(모두 CI > 0). 크기로 보면 미미하다.
3. **Q3. 직접 readout보다 나은가? 아니다.**
   - B − Full Multi-layer + R2 = **−2.18 [−3.30, −1.33]**, Obj D5–16 −2.70 [−4.17, −1.44].
   - 같은 L4/L8/L12 정보를 R2로 직접 읽는 것(45.02)이 그 정보를 decoder로 C0 L12 공간에 복원한 뒤 읽는 것(42.84)보다 크게 높다.
   - **해석:** 중간 layer 접근의 효과는 매칭 대조군 대비로는 유의하지만 E6-original 대비로는 확인되지 않아 약한 근거에 그친다. Decoder의 추가 실용 가치는 입증되지 않았다.

## 1. 구조 변경

| | memory token m(p) (그 뒤 kv = m + pe, block별 ln_kv → K/V projection은 변경 없음) | Params |
|---|---|---|
| E6-original | P12(F12) | 2,183,168 |
| **A Matched L12** | P12(F12) + P4(F12) + P8(F12) | 2,579,968 |
| **B Multi-layer** | P12(F12) + P4(F4) + P8(F8) | 2,579,968 |

- **P12:** 기존 `mem_in` = LN(768) → Linear(768, 256)이고, canonical init을 그대로 쓴다.
- **P4/P8:** P12와 같은 구조다(각자 LN + Linear). 정규화 규칙은 두 조건이 같다(projection마다 자체 LN). Linear는 zero-init이다.
- **Feature 정의:** F4/F8 = block 4/8 raw residual, F12 = final-LN이다. 이미지당 F336 forward를 한 번 한다. **Full Multi-layer + R2 baseline과 같은 layer 정의**다(값 차이 0.0).
- **변하지 않는 부분:** query(L12 Bicubic ROI), skip(L12 Bicubic), 2-block cross-attention, head, PE, 출력, teacher, loss(L_old만)는 E6와 같다. Memory token은 576개다.
- **코드:** `src/e6ml/models.py`(XDec을 상속한 E6Mem), `src/e6ml/pretrain.py`(xdec pretrain과 diff 26줄: decoder builder와 skip check만), `src/e6ml/probe.py`.

## 2. 초기화 검증 (CPU smoke `smoke/smoke.json` ALL_PASS + GPU smoke)

- **초기 출력 일치:**
  - A와 B의 init state가 완전히 같다. E6-original과 공통인 모든 파라미터도 같다.
  - 초기 memory 차이: A − E6orig 0.0, B − E6orig 0.0.
  - 초기 출력 차이: A − B 0.0, A − E6orig 0.0, B − Bicubic 0.0. GPU smoke에서도 step 0에서 Z == B를 확인했다.
- **Gradient 흐름:**
  - Update 1: head만 gradient를 받는다(E6의 zero-init head 때문, E6-original과 같음).
  - Update 2: trunk, P12, **P4/P8 Linear**(0.012)가 gradient를 받는다.
  - Update 3: **P4/P8 LN**도 gradient를 받는다. Gradient가 영구적으로 막히지 않는다.
  - GPU smoke의 update 2에서 P4/P8 gradient는 P12와 같은 크기(1e-5)였다. 5 update 뒤 A와 B의 출력이 달라졌다.
- **입력 정렬:**
  - Layer 4/8/12 feature가 같은 canvas와 24×24 grid에서 나오는 것을 확인했다. Full-ML 인코더 호출과의 차이는 0.0이다.
  - Coverage(`valid_w`)와 위치(`full_uv`)는 E6 코드를 그대로 쓴다.
  - Token norm은 L4 1.35, L8 5.8, L12 49.3으로 크게 다르다. 각 LN이 이를 흡수한다.

## 3. 학습 결과 (COCO dev, 선택 = L_old 최소 → A, B 모두 12k)

| step | E6-original L_old | A L_old | B L_old |
|---|---|---|---|
| 3000 | 2.5071 | 2.5072 | 2.5009 |
| 6000 | 2.4201 | 2.4169 | 2.4072 |
| 9000 | 2.3953 | 2.3906 | 2.3780 |
| 12000 (선택) | 2.3908 | 2.3857 | **2.3727** |

- **학습 시간:** A 525초, B 518초.
- **추이:** B의 dev L_old가 6k부터 일관되게 가장 낮다.

## 4. PACO 결과 (val, dev로 고른 epoch)

| Method | R2 epoch | **R2 M0 Overall** | **Obj D5–16** | Part D1–4 | R2 M2 | R0 M0 |
|---|---|---|---|---|---|---|
| Bicubic-only | 6 | 42.71 | 46.47 | 40.10 | 44.05 | 37.04 |
| E6-original | 6 | 42.70 | 47.53 | 41.19 | 43.43 | 38.07 |
| A Matched L12 | 6 | 42.12 | 47.14 | 41.04 | 42.80 | 38.09 |
| **B Multi-layer Memory** | 6 | **42.84** | **47.82** | **41.23** | 43.31 | **38.41** |
| Full Multi-layer + R2 | 7 | **45.02** | **50.52** | **43.72** | **45.51** | – |

**검증:** 다섯 조건이 같은 val 행·이미지와 같은 train 행·이미지를 쓰는 것을 확인했다(probe config 일치). Reader 프로토콜(`src.ivc.train`, dev 1,000장, 10 epoch)도 같다.

## 5. Paired Image Bootstrap 95% CI (B = 1,000, seed 0, 같은 draw)

| 대비 | R2 M0 Overall | R2 Obj D5–16 | R2 Part D1–4 | R2 M2 Overall | R0 M0 Overall |
|---|---|---|---|---|---|
| **1. B − A (Matched L12)** | **+0.72 [+0.09, +1.17]** | +0.68 [−0.03, +1.31] | +0.19 [−0.38, +0.96] | +0.51 [−0.52, +1.89] | **+0.32 [+0.16, +0.49]** |
| **2. B − E6-original** | +0.13 [−0.46, +0.66] | +0.30 [−0.43, +1.01] | +0.04 [−0.64, +1.05] | −0.12 [−1.14, +1.42] | **+0.34 [+0.16, +0.54]** |
| **3. B − Bicubic-only** | +0.12 [−0.65, +0.68] | **+1.35 [+0.21, +2.27]** | **+1.13 [+0.06, +2.52]** | −0.75 [−1.96, +1.21] | **+1.37 [+0.97, +1.72]** |
| **4. B − Full Multi-layer + R2** | **−2.18 [−3.30, −1.33]** | **−2.70 [−4.17, −1.44]** | **−2.49 [−3.91, −0.82]** | **−2.20 [−4.07, −0.68]** | – (Full ML는 R2만 있음) |
| (참고) A − E6-original | **−0.59 [−1.04, −0.04]** | −0.38 [−1.03, +0.36] | −0.15 | −0.64 | +0.02 [−0.10, +0.14] |
| (참고) Full ML + R2 − Bicubic | **+2.30 [+1.41, +3.28]** | **+4.05** | **+3.62** | +1.45 | – |

- **B − Bicubic 세부:** Obj D17+에서 B − Bicubic = −1.35 [−2.61, −0.37]이다. 큰 object에서는 decoder가 Bicubic보다 낮은 패턴이 이번에도 반복됐다.
- **B − Full ML 세부:** Color −5.13에서 차이가 가장 크다.

## 6. Feature 진단 (C0 teacher 대비, PACO val 11,109 ROI)

| Method | Token MSE | Token Cos | Centered Dir Cos | Pooled Cos | Mean-feature MSE | Spatial Var / C0 |
|---|---|---|---|---|---|---|
| Bicubic | 3.567 | 0.350 | 0.239 | 0.474 | 1.947 | 0.437 |
| E6-original | 2.172 | 0.515 | 0.387 | 0.663 | 0.824 | 0.382 |
| A Matched L12 | 2.169 | 0.515 | 0.387 | 0.663 | 0.822 | 0.381 |
| B Multi-layer | **2.159** | **0.519** | **0.388** | **0.669** | **0.814** | 0.381 |

- **B − A:** token cos +0.0040 [+0.0038, +0.0043], MSE −0.0101 [−0.0107, −0.0096], centered dir cos +0.0012 [+0.0012, +0.0013].
- **B − E6-original:** +0.0043 / −0.0131 / +0.0014.
- **COCO dev:** 같은 방향이다.
- **크기·level별:** centered dir cos는 모든 크기와 level에서 +0.001 ~ +0.002다. 공간 분산비는 그대로다(0.381).
- **해석:** 중간 layer가 복원을 개선하는 방향은 일관되지만 폭은 매우 작다. 같은 seed의 두 모델을 같은 ROI에서 비교했기 때문에 CI가 좁다.

## 7. 질문별 답

**Q1. 중간 layer 제공 자체가 도움이 되었는가?**
- **효과는 약하고, 확정하기 어렵다.**
- 사전에 정한 대비(B − A)에서는 R2 M0가 +0.72로 유의하다.
- 하지만 같은 init의 사실상 같은 모델인 A와 E6-original이 −0.59로 갈렸다. B는 E6-original과 차이가 없다(+0.13). 그래서 R2에서 seed 하나로 0.6 수준의 차이는 학습 변동과 구분하기 어렵다.
- R0에서는 B가 두 L12 모델보다 +0.3씩 일관되게 높았다(A ≈ E6-original). 중간 layer가 작은 이득을 준다는 방향의 근거다.

**Q2. C0 feature 복원도 개선되었는가?**
- **아주 조금 개선됐다.** Token cos +0.004, MSE −0.01, centered dir cos +0.001.
- 이것만으로 성공이라고 판단하지 않는다.

**Q3. Decoder가 직접 readout보다 추가 가치를 제공했는가?**
- **아니다.** B − Full Multi-layer + R2 = −2.18 [−3.30, −1.33]이고, primary와 secondary 지표 대부분에서 유의하게 낮다.
- 같은 L4/L8/L12 정보를 R2로 직접 읽으면 45.02인데, decoder가 그 정보를 C0 L12 teacher 공간으로 복원하는 과정을 거치면 42.84로 대부분 사라진다.
- **해석:** 중간 layer 정보가 decoder 입력에 들어가도, 학습 목표가 최종층 C0 feature 복원(L_old)이고 R2가 그 복원 결과만 읽는 구조에서는 multi-layer의 판별 정보가 출력에 거의 남지 않는다. 기존 teacher matching의 한계가 다시 나타난 것으로 보인다. 다만 이 원인은 이번 실험에서 직접 검증하지 않았다.

**종합:** 중간 layer 접근의 효과는 matched control 대비로는 보였지만(R2 +0.72, R0 +0.32), E6-original 대비로는 R2에서 확인되지 않아 근거가 약하다. Decoder의 추가적인 실용 가치는 입증되지 않았다(Full ML + R2보다 −2.2).

## 8. 한계

- **Seed:** 0 하나다. A와 E6-original 사이의 −0.59가 보여주듯 R2에서 seed 수준의 변동이 대조 효과와 비슷한 크기일 수 있다. 이를 확인하려면 multi-seed가 필요하지만 지시에 따라 하지 않았다.
- **설정 고정:** E6 하나, L_old 하나, d 256만 썼다. 다른 decoder나 loss(예: multi-layer teacher 감독)에서의 결과로 일반화하지 않는다.
- **Full ML + R2 비교의 성격:** Full ML + R2는 decoder 없이 multi-layer feature를 R2(학습 파라미터 포함)로 바로 읽는다. 그래서 이 비교는 같은 입력 정보에서 "decoder 경로 vs 직접 readout" 비교이고, 같은 parameter 조건의 구조 비교는 아니다.

## 9. 실행 상태와 QAT

| 항목 | 값 |
|---|---|
| 큐 | tmux `e6ml_q`, smoke_cpu → qat_pause → smoke_gpu → train_A → train_B → probe_A → probe_B → recon → qat_resume → analysis **모두 DONE** |
| QAT 중단 | 06:14 사용자 승인("지금 QAT 중단") → 06:16:30 SIGTERM, **step 31,261**, 재개 기준 **checkpoint-30000** → **1,261 step 유실** |
| 우리 GPU 사용 | 06:16:37 – 06:59:47 (≈ 43분 × H200 2장 ≈ 1.4 GPU-h). GPU 0,1은 사용하지 않았다 |
| QAT 재개 | 06:59:48 tmux **`qat_800m`**에서 checkpoint-30000으로 재개(saved cmdline `--resume`, env, cwd). 07:02:34에 step 30,057, loss 3.208로 **검증 OK** ([`qat_resume_status.md`](qat_resume_status.md)). 07:07 기준 step 30,231로 증가 중 |

- **수동 복구 명령:** `bash output/2026-11-06/e6_multilayer_memory/scripts/recover_qat.sh`
- **덮어쓰지 않은 것:** 기존 xdec, IVC, MSD, e6_direction의 코드, checkpoint, 결과는 덮어쓰지 않았다. 새 코드는 `src/e6ml/`이고, 산출물은 `output/2026-11-06/e6_multilayer_memory/`이다.
