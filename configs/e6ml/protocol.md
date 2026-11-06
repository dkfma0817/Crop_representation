# E6 Multi-layer Memory Pilot: 프로토콜 (2026-11-06, 학습 결과를 보기 전에 기록)

**핵심 질문:** 같은 E6 decoder에서 K/V memory에 실제 중간 layer(L4/L8)를 넣는 것이 최종층(L12)만 넣는 것보다 유용한가?

## 구조 변경 (최소 수정)
- **기존 E6 memory 경로:** `kv(p) = mem_in(F12_p)[img] + pe(p)`, `mem_in = LN(768) → Linear(768, 256)`. 이후 block마다 `ln_kv` → K/V projection이 이어진다.
- **변경:** `mem_in`(= P12) 앞의 memory token만 다음처럼 바꾼다.
  - **A Matched L12:** m(p) = P12(F12) + P4(F12) + P8(F12)
  - **B Multi-layer:** m(p) = P12(F12) + P4(F4) + P8(F8)
  - 그 뒤 kv = m + pe이고, ln_kv와 K/V projection은 바꾸지 않는다.
- **P4, P8:** P12와 같은 구조다(각자 LN(768) + Linear(768, 256)).
  - 정규화 규칙: 각 projection의 자체 LN이다. 두 조건 모두 같다.
  - 초기화: Linear를 zero-init한다.
- **Parameter:** A = B = 2,579,968 (E6 2,183,168 + 396,800).
- **변하지 않는 부분:** query(L12 Bicubic ROI), skip(L12 Bicubic), 2-block full cross-attention, head, 출력 24×24×768, PE, memory token 576개는 E6와 같다.
- **Feature 정의:** F4, F8 = block 4/8 출력(raw residual), F12 = final-LN 출력이다. `MS.Student`에서 이미지당 한 번 forward해서 얻는다. **Full Multi-layer + R2 baseline과 같은 layer 정의**다(`RC.encode_layers` ML_LAYERS와 차이 0.0, smoke에서 확인).
- **코드:** `src/e6ml/models.py`(E6Mem ⊂ XDec), `src/e6ml/pretrain.py`(xdec pretrain에서 decoder builder만 바꿈, diff 26줄), `src/e6ml/probe.py`.

## 초기화 검증 (CPU smoke, `smoke/smoke.json`, ALL_PASS)
- **초기 출력 일치:** A와 B의 init state가 같다. 공통 key는 E6-original init과도 같다. 초기 memory와 출력 차이는 A−B 0.0, A−E6orig 0.0, B−Bicubic 0.0이다.
- **Gradient 흐름:**
  - Update 1에서는 head만 gradient를 받는다(E6의 zero-init head 때문).
  - Update 2부터 trunk, P12, **P4/P8 Linear**(0.012)가 gradient를 받는다.
  - Update 3부터 **P4/P8 LN**도 gradient를 받는다. Gradient가 영구적으로 막히지 않는다.
  - 5 update 뒤 A와 B 출력이 달라진다(0.016).
- **Feature 정렬과 scale:**
  - 정렬: 같은 canvas와 24×24 grid다. coverage(`valid_w`)와 위치(`full_uv`) 코드는 E6와 같다.
  - Scale: token norm이 L4 1.35, L8 5.8, L12 49.3으로 크게 다르다. 각 projection의 LN이 이를 흡수한다.

## 학습 / 선택 / 평가 (= xdec E6)
- **학습:** COCO 12k stream(64 ROI/step), 12,000 step, seed 0, AdamW 1e-4 (0.9, 0.95) wd 0.05, warmup 600 + cosine, clip 1.0, BF16, GPU 2,3 DDP.
- **Loss:** **L_old만** 쓴다(MSE_w + 1.0(1 − cos)_w). Direction loss는 쓰지 않는다. A와 B 모두 처음부터 학습한다.
- **Checkpoint:** 3k/6k/9k/12k 중 COCO-dev L_old 최소.
- **Probe:** xdec probe_E6와 같은 config(decoder 경로만 다름). `src.ivc.train`, dev 1,000장, 10 epoch, R2 + R0 seed 0, dev M0 최대 epoch을 쓴다. **Val만** 평가하고 test는 평가하지 않는다.
- **재사용:** Bicubic(IVC probe_bic), E6-original(xdec probe_E6), Full Multi-layer + R2(MSD probe_mlr2, R2만).
- **Primary:** R2 M0 Overall, Obj D5–16.
- **Secondary:** Part D1–4, R2 M2, R0 M0.
- **Bootstrap:** paired image bootstrap 1,000회, seed 0.
  - 대비: B−A, B−E6orig, B−Bicubic, B−Full ML(+ 참고로 A−E6orig).
- **표현 진단:** token MSE/cos, centered direction cos, spatial variance ratio, pooled cos, mean-feature MSE를 PACO val 전체와 COCO dev에서 낸다.

## 해석 규칙
- **Q1:** B−A(중간 layer 효과, parameter와 init을 맞춘 조건).
- **Q2:** 표현 지표. 복원만 좋아지면 성공이 아니다.
- **Q3:** B − Full ML + R2.
- **결과 조합별 해석:** Q1이 개선되고 Q3가 개선되지 않으면 "중간 layer 접근 효과는 확인, decoder의 추가 실용 가치는 미입증"이다. 복원만 개선되고 PACO가 그대로면 "teacher matching 한계의 반복"이다.
- 중간 layer가 반드시 도움이 된다고 가정하지 않는다. Seed는 하나다.
