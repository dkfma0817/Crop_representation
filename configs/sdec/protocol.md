# S-series Spatial ROI Decoder 파일럿: 실행 전 문서 (2026-11-05)

배경: H1(multi-depth 감독)은 H0보다 PACO M0가 +1.36 높았다. 하지만 공간 분산은 teacher 대비 0.13~0.15였고, Full Multi-layer+R2(45.02)보다는 낮았다. 그래서 이번에는 **decoder 내부의 공간 처리 구조**를 바꿔서 이 한계가 개선되는지 본다. Reader만 바꾸는 실험이 아니다. 실제로 spatial ROI feature(24×24×768)를 만드는 decoder를 COCO에서 학습한다.

## 공통 (A0/A1/B0/B1 동일, H1 프로토콜 그대로)
- **Encoder:** Frozen DINOv2-B/14, full image F336, Block 4/8/12. 입력 융합은 H1과 같은 `F = F12 + A4(F4) + A8(F8)`(zero-init adapter)이다.
- **COCO 사전학습:** 기존 12k stream(64 ROI/step), seed 0. PACO label은 쓰지 않는다. Checkpoint는 COCO dev L12로 고른다(H1과 같은 dev 468장).
- **Teacher:** 원본 COCO GT crop(C0) → frozen DINOv2 → Block 4/8/12. 학습 감독에만 쓴다.
- **Loss (H1과 동일):** `L12 + 0.5·L8 + 0.5·L4`. L_d는 depth별 teacher 정규화 공간에서 coverage로 가중한 `[MSE + (1 − cos)]`이다. 정규화 통계는 H1과 같은 `teacher_stats.npz`를 쓴다.
- **보조 head:** L4/L8용으로 H1과 같은 Linear 768→768을 두고, R2 평가에서는 제거한다.
- **최적화:** AdamW lr 1e-4 (0.9, 0.95), wd 0.05, warmup 600 + cosine, clip 1.0, BF16, 12,000 step, GPU 2+3 DDP.
- **새 decoder 4개의 공통 출력 규칙:** `Z = Bilinear(F12) + σ12 · Linear_out(LN(h))`이고, Linear_out은 0으로 초기화한다. 그래서 시작 시 출력이 Bilinear 기준선과 같다.
  - 이 규칙은 네 조건 모두 같으므로 쌍 안의 비교에는 영향이 없다.
  - H1(CoordMLP, 평균에서 시작)과는 다른 점이라, H1과의 비교에서는 이 규칙 차이도 함께 들어간다.
- **내부 차원:** d = 384, heads 6이고, 초기화 seed는 0으로 네 조건 모두 같다.
- **PACO probe:** encoder와 decoder를 frozen하고 R2만 학습한다. 이전 실행과 같은 프로토콜이다(PACO train에서 dev 1,000장을 떼어 epoch 선택, val에서 최종 평가, Primary는 M0 Overall). Test는 쓰지 않는다.

## 실험 A: S1 Hierarchical vs Flat
| | A0 Flat | A1 Hierarchical |
|---|---|---|
| Block | 4개 [Self-attn, Cross-attn(ROI+ring cache token), FFN×4] | 같은 4개 block |
| 해상도 | 24×24 (모든 block) | 4×4 → 8×8 → 16×16 → 24×24 (block마다 하나씩) |
| 단계 사이 | 시작에 grid feature를 한 번 입력 | 상태를 bilinear upsample하고, 그 해상도의 grid feature와 위치 임베딩을 더함 |
| 파라미터 (aux 제외) | 10,708,992 | 10,708,992 |
| 출력 | 24×24×768 | 24×24×768 |
| 연산량 | 많음 (24² self-attn × 4) | 적음 |

## 실험 B: S3 One-way vs Two-way
| | B0 One-way | B1 Two-way |
|---|---|---|
| Query | box 조건 query 16개 (COCO ROI box) | 같음 |
| Working map | 24×24, 원본 cache에서 bilinear로 만든 W0. **고정** | 같은 W0에서 시작해 **매 stage 갱신** |
| Stage (3회) | q += CA(q, cache); q += CA(q, W); q += FFN; 대조 연산 q += SA(q); q += FFN2(q) | 앞의 세 연산은 같고, 대조 연산 대신 W += CA(W ← q); W += FFN_w(W) |
| 최종 readout | W0 + CA(W0, q) + FFN → 출력 head | W_L + CA(W_L, q) + FFN → 출력 head |
| 파라미터 (aux 제외) | 11,467,392 | 11,467,392 |

- 원본 full-image feature cache는 두 조건 모두 읽기 전용이다. Smoke test에서 체크섬이 같고 `requires_grad=False`인 것을 확인했고, 직접 검사의 모든 forward에서도 다시 확인한다.
- B0의 대조 연산은 query 쪽에 있다. 그래서 파라미터는 같지만 연산량은 B1(576개 map token 처리)보다 적다. 연산량을 "일부" 통제한 것이다.

## Smoke test 확인 (60 step)
- **출력 shape:** 4조건 모두 [R, 576, 768].
- **Encoder:** gradient 없음.
- **Gradient 흐름:** step 1에는 0으로 초기화한 출력 head만 gradient를 받고, step 2부터 decoder 본체도 gradient(1e-3 수준)를 받아 가중치가 바뀐다. L4/L8 loss만 따로 역전파해도 decoder 본체까지 도달한다(0.12).
- **Cache:** 변경 없음.

## 직접 검사 (PACO val 700장, 보고 전용)
- L12, 공간 분산 비율, effective rank(teacher 대비 비율)
- 이웃 cell 차이 비율, centred cosine(위치별 패턴이 teacher와 맞는지), patch affinity 차이
- 단계별 변화: A는 각 block 상태를 출력 head에 통과시켜 측정, B는 매 stage의 working map 분산과 effective rank
- 작은 object(n_in 1–16), part, 무늬 있는 ROI(non-plain pattern)

## 판정
- 쌍별로 A1−A0, B1−B0의 PACO val M0 Overall paired CI를 본다.
- 직접 검사의 개선 기준은 분산 비율↑, effective rank 비율↑, affinity 차이↓이다. 이것과 PACO 개선은 따로 기록한다.
- PACO가 개선되지 않으면 성공이 아니다.
- 기준선(H1 43.67, Full ML+R2 45.02, Bilinear 42.30, C0 45.74)은 같은 probe 프로토콜로 저장된 예측을 같은 bootstrap으로 다시 비교한다.

## 기존 실험과의 중복
- **H1:** 같은 loss·데이터·입력·probe를 쓰고, decoder 구조만 다르다(CoordMLP → attention 기반 spatial decoder, 출력이 Bilinear residual).
- **R00/R10, Pretrained Region Decoder:** CoordMLP 계열이고, checkpoint가 이 서버에 없다.
- **계층형/two-way 구조:** 이 저장소의 기존 실험에는 없다.
