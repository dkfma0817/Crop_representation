# Crop Representation Decoder Architecture Pilot (E1–E7) — 결과 보고서 (2026-11-05)

Frozen F336 final-layer feature에서 C0 crop feature를 만드는 decoder 구조 7개를 같은 loss, 데이터, recipe, reader로 비교했다.
- **평가 범위:** PACO **Val만** 평가했다. Test는 평가하지 않았다.
- **Seed:** 학습 seed는 0 하나다. CI는 평가 표본(이미지) 불확실성이고 학습 seed 분산은 아니다.
- **관련 문서:** 사전 프로토콜은 [`protocol.md`](protocol.md), 기존 구조 감사는 [`architecture_audit.md`](architecture_audit.md)에 있다. 자동 생성 표 전체는 [`tables.md`](tables.md), 수치 원본은 [`results/analysis.json`](results/analysis.json)에 있다.

## 결론 먼저

1. **실용적 성공은 없다.**
   - Primary(R2 val M0 Overall)에서 7개 decoder 중 어느 것도 Bicubic(42.71)이나 Direct Full(42.64)을 넘지 못했다.
   - 가장 좋은 E6이 42.70(−0.01 [−0.73, +0.58])이다.
   - E1, E3, E4, E7은 Bicubic보다 유의하게 낮다.
   - 모든 decoder가 Full Multi-layer(45.02)보다 2.3–3.7, C0(45.45)보다 2.7–4.1 낮다(전부 CI < 0).
   - Obj D5–16(R2)에서는 어떤 decoder도 Bicubic과 차이가 확인되지 않았다(−0.15 ~ +1.05, 모든 CI가 0을 포함).
2. **R2 기준으로는 구성 요소 효과가 하나도 확인되지 않았다.**
   - 주요 대비 7개 중 E1−Bicubic(−0.58, CI < 0)만 유의하다.
   - E3−E2, E4−E3, E5−E3, E6−E4, E6−E5, E7−E3과 2×2 interaction은 모두 Overall CI가 0을 포함한다.
3. **R0(secondary)에서는 효과가 분명하다. 다만 R2로 이어지지 않는다.**
   - Bicubic skip이 크게 좋고(E5−E3 +1.79, E6−E4 +1.45), Bicubic query도 좋다(E4−E3 +0.75).
   - E1, E5, E6은 Bicubic보다 높다(E6 +1.04 [+0.64, +1.38]).
   - 그러나 R0 절대 수준(≈36–38)은 R2(≈41–43)보다 훨씬 낮고, 같은 이득이 R2에서는 사라진다. 그래서 이것은 **reader 차이**(평균 pooling reader에 유리한 변화)로 분리해서 본다.
4. **Teacher feature 복원은 크게 좋아졌지만, PACO 개선으로 이어지지 않았다. 방법 간 순서는 오히려 반대다.**
   - PACO-val token cosine이 Bicubic 0.35에서 0.50–0.57로 올랐다.
   - 복원이 가장 좋은 R00(0.57)과 direct decoder(E2/E3/E4/E7)가 R2에서 가장 낮다.
   - 방법 간 Spearman(PACO-val feature loss vs R2 M0)은 +0.80이다. 즉 feature loss가 높을수록 M0가 높았다(n = 9, CI 없음).
   - Direct decoder는 ROI 내부 공간 분산을 teacher의 18%로 줄인다(Bicubic 44%). 출력이 평탄해지는 방향이다.

## 1. 실행 상태

| 항목 | 상태 |
|---|---|
| E1–E7 decoder 학습 + PACO probe (R0 + R2) | **모두 DONE** (16:24–18:44 UTC, 조건당 학습 8분 + probe 11분) |
| B0 Direct Full probe (신규) | DONE |
| R00 재학습 + probe (B2, 신규) | DONE |
| Feature reconstruction | DONE (GPU 실행이 코드 버그로 실패해 **QAT 재개 후 CPU fp32로 다시 실행**. GPU bf16 dev 평가와 차이 ≤ 5e-4, 예: E1 2.4136 vs 2.4141) |
| Bootstrap / 2×2 / 표 | DONE |
| QAT 재개 | **DONE, 검증 OK** (7절) |

**선택된 checkpoint:** 8개 decoder 모두 사전 규칙(3k/6k/9k/12k 중 COCO-dev feature loss 최소)에 따라 **12k**가 선택됐다. Dev loss는 9k에서 12k 사이에도 조금씩 계속 줄었다(감소폭 0.004–0.009).

**Reader epoch:** dev M0 최대 epoch을 썼다. R2는 6–8, R0는 9–10이다.

## 2. 주요 표 (PACO val, R2 primary)

| Method | COCO-dev Feature Loss | COCO-dev Token Cos | PACO-val Token Cos | **Val M0 R2** | **Obj D5–16** | Val M2 R2 | Val M0 R0 |
|---|---|---|---|---|---|---|---|
| Direct Full (B0) | – | – | – | 42.64 | 46.88 | 43.34 | 36.70 |
| Bicubic-only (B1) | 3.618 | 0.448 | 0.350 | **42.71** | 46.47 | 44.05 | 37.04 |
| R00 / CoordMLP (B2, 재학습) | **2.045** | **0.628** | **0.572** | 41.39 | 46.44 | 41.33 | 37.79 |
| E1 Bic + SA refine (skip) | 2.414 | 0.577 | 0.505 | 42.13 | 46.77 | 42.75 | 38.01 |
| E2 Mask + ROI CA | 2.223 | 0.592 | 0.549 | 41.87 | 47.00 | 41.30 | 36.16 |
| E3 Mask + Full CA | 2.227 | 0.591 | 0.548 | 41.55 | 46.88 | 41.03 | 35.87 |
| E4 Bic query + Full CA | 2.178 | 0.601 | 0.556 | 41.97 | 47.07 | 42.71 | 36.62 |
| E5 Mask + Full CA + skip | 2.541 | 0.571 | 0.503 | 42.25 | 46.32 | 42.66 | 37.66 |
| E6 Bic query + Full CA + skip | 2.391 | 0.581 | 0.515 | **42.70** | **47.53** | 43.43 | **38.07** |
| E7 Mask + Full CA + SA | 2.215 | 0.594 | 0.552 | 41.34 | 47.24 | 40.92 | 35.74 |
| Full Multi-layer + R2 (B3, 재사용) | – | – | – | 45.02 | 50.52 | 45.51 | – |
| C0 Oracle (B4, 재사용) | 0 | 1.0 | 1.0 | 45.45 | 54.92 | 47.72 | 39.64 |

- Bicubic과 C0는 IVC probe, Full ML은 MSD probe를 재사용했다. 같은 probe 코드, 같은 PACO dev/val 분할, 같은 val 행·label을 쓴 것을 확인했다(probe config 일치).
- Direct Full과 Full ML은 full-grid를 읽으므로 crop-grid 복원 지표가 없다.
- 세부 attribute, 크기 bin, M2와 R0 표는 [`tables.md`](tables.md) 2절에 있다.

## 3. 구성 요소 대비 (paired image bootstrap, B = 1000, seed 0, 같은 draw)

| 대비 (질문) | M0 Overall **R2** | Obj D5–16 R2 | M2 Overall R2 | M0 Overall R0 |
|---|---|---|---|---|
| E1 − Bicubic (spatial refine) | **−0.58 [−1.33, −0.04]** | +0.30 [−0.66, +1.22] | −1.31 [−3.02, +0.11] | **+0.97 [+0.53, +1.33]** |
| E3 − E2 (Full vs ROI memory) | −0.32 [−0.89, +0.17] | −0.12 [−0.95, +0.63] | −0.27 [−1.66, +1.24] | **−0.29 [−0.59, −0.02]** |
| E4 − E3 (Bic vs Mask query, no skip) | +0.42 [−0.31, +1.25] | +0.19 [−0.80, +1.22] | +1.68 [−0.85, +2.96] | **+0.75 [+0.48, +1.13]** |
| E5 − E3 (skip, Mask query) | +0.70 [−0.04, +1.75] | −0.56 [−1.76, +0.80] | +1.63 [−1.65, +2.77] | **+1.79 [+1.25, +2.53]** |
| E6 − E4 (skip, Bic query) | +0.73 [−0.06, +1.61] | +0.46 [−0.83, +1.55] | +0.72 [−1.41, +1.85] | **+1.45 [+0.86, +2.03]** |
| E6 − E5 (Bic vs Mask query, with skip) | +0.46 [−0.21, +1.02] | **+1.20 [+0.31, +1.96]** | +0.77 [−0.72, +1.89] | **+0.41 [+0.06, +0.68]** |
| E7 − E3 (query self-attention) | −0.20 [−0.75, +0.48] | +0.36 [−0.59, +1.20] | −0.11 [−1.85, +0.93] | −0.13 [−0.38, +0.13] |

### 2×2 (E3 / E4 / E5 / E6), M0 Overall

| | No Skip | Bicubic Skip |
|---|---|---|
| Mask Query | E3 41.55 (R0 35.87) | E5 42.25 (R0 37.66) |
| Bicubic Query | E4 41.97 (R0 36.62) | E6 42.70 (R0 38.07) |

| 효과 | R2 | R0 |
|---|---|---|
| Query, no skip (E4 − E3) | +0.42 [−0.31, +1.25] | **+0.75 [+0.48, +1.13]** |
| Query, with skip (E6 − E5) | +0.46 [−0.21, +1.02] | **+0.41 [+0.06, +0.68]** |
| Skip, Mask query (E5 − E3) | +0.70 [−0.04, +1.75] | **+1.79 [+1.25, +2.53]** |
| Skip, Bicubic query (E6 − E4) | +0.73 [−0.06, +1.61] | **+1.45 [+0.86, +2.03]** |
| **Interaction** (E6 − E5) − (E4 − E3) | +0.03 [−1.00, +0.89] | −0.34 [−0.88, +0.00] |

R2에서는 두 주효과 모두 +0.4 ~ +0.7 방향이지만 CI가 0을 포함하고, interaction은 0이다. R0에서는 두 효과가 모두 유의하고 거의 가산적이다(interaction은 약하게 음수이고 경계선).

### 각 decoder − 기준선 (M0 Overall)

| Decoder | − Direct Full (R2) | − Bicubic (R2) | − R00 (R2) | − Full ML (R2) | − C0 (R2) | − Bicubic (R0) |
|---|---|---|---|---|---|---|
| E1 | −0.51 [−1.24, +0.11] | **−0.58 [−1.33, −0.04]** | +0.74 [−0.26, +1.63] | −2.89 [−4.04, −2.08] | −3.32 | **+0.97** |
| E2 | −0.77 [−1.72, +0.10] | −0.84 [−1.83, +0.02] | +0.48 [−0.33, +1.17] | −3.15 | −3.58 | **−0.88** |
| E3 | **−1.09 [−2.10, −0.32]** | **−1.17 [−2.21, −0.39]** | +0.16 [−0.73, +0.88] | −3.47 | −3.90 | **−1.17** |
| E4 | −0.67 [−1.52, +0.07] | **−0.74 [−1.63, −0.06]** | +0.58 [−0.37, +1.47] | −3.05 | −3.48 | −0.41 |
| E5 | −0.39 [−1.04, +0.29] | −0.47 [−1.10, +0.19] | +0.85 [−0.06, +1.81] | −2.77 | −3.20 | **+0.63** |
| E6 | +0.07 [−0.64, +0.74] | −0.01 [−0.73, +0.58] | **+1.31 [+0.35, +2.25]** | −2.31 [−3.33, −1.44] | −2.74 | **+1.04** |
| E7 | **−1.29 [−2.34, −0.33]** | **−1.37 [−2.39, −0.47]** | −0.05 [−0.90, +0.75] | −3.67 | −4.10 | **−1.30** |
| R00 | **−1.25 [−2.21, −0.23]** | **−1.32 [−2.25, −0.41]** | – | – | – | **+0.75** |

- Full ML과 C0 열의 차이는 모두 CI < 0이다. 전체 CI는 tables.md 3.2절에 있다.
- **크기별(R2, − Bicubic):** decoder의 손실은 큰 ROI에 몰려 있다.
  - Obj D17+: E3 −2.93, E7 −2.93, R00 −2.28, E1 −1.40 (모두 CI < 0). E5 −0.84와 E6 −0.86은 n.s.
  - Part D1–4: E6만 +1.09 [+0.15, +2.10]이다.
  - Color와 Pattern: direct decoder(E3/E7)와 R00가 유의하게 낮다.

## 4. Feature Reconstruction (teacher = C0, raw final-LN 공간)

| Method | PACO-val loss | token cos | pooled cos | centred diff e_cent | spatial var ratio | self-sim diff | COCO-dev loss |
|---|---|---|---|---|---|---|---|
| Bicubic | 4.217 | 0.350 | 0.474 | 1.621 | 0.437 | 0.337 | 3.618 |
| R00 | **2.222** | **0.572** | **0.736** | **1.256** | 0.190 | 0.280 | **2.045** |
| E1 | 2.692 | 0.505 | 0.650 | 1.354 | 0.386 | 0.253 | 2.414 |
| E2 | 2.330 | 0.549 | 0.704 | 1.294 | 0.175 | 0.264 | 2.223 |
| E3 | 2.333 | 0.548 | 0.704 | 1.298 | 0.178 | 0.259 | 2.227 |
| E4 | 2.297 | 0.556 | 0.711 | 1.280 | 0.183 | 0.260 | 2.178 |
| E5 | 2.850 | 0.503 | 0.649 | 1.409 | **0.457** | **0.252** | 2.541 |
| E6 | 2.657 | 0.515 | 0.663 | 1.349 | 0.382 | 0.256 | 2.391 |
| E7 | 2.320 | 0.552 | 0.706 | 1.292 | 0.179 | 0.258 | 2.215 |

PACO-val token cos는 크기·level별로 다음과 같다(tables.md 5.2절).

| | Object D1–4 | Object D5–16 | Object D17+ | Part D1–4 | Part D5–16 | Part D17+ |
|---|---|---|---|---|---|---|
| Bicubic | 0.240 | 0.322 | 0.579 | 0.113 | 0.301 | 0.573 |
| R00 | 0.505 | 0.532 | 0.660 | 0.522 | 0.528 | 0.655 |
| E3 (direct) | 0.501 | 0.518 | 0.598 | 0.531 | 0.517 | 0.594 |
| E6 (skip) | 0.433 | 0.472 | 0.650 | 0.414 | 0.463 | 0.642 |

**관찰**
- **복원 개선:** 모든 decoder가 Bicubic보다 teacher에 훨씬 가깝다. 작은 ROI일수록 개선이 크다(Part D1–4 0.11 → 0.39–0.54).
- **구조별 차이:** Direct 출력 decoder(E2/E3/E4/E7)와 R00는 token/pooled cosine이 가장 높지만, 공간 분산비는 0.18–0.19로 Bicubic(0.44)보다 훨씬 작다. ROI 안에서 평탄한 평균형 map을 만든다는 뜻이다. Skip decoder(E1/E5/E6)는 Bicubic의 분산(0.38–0.46)을 유지하고, self-similarity 차이도 가장 작다.
- **큰 ROI:** D17+에서 direct decoder의 token cos(0.60)는 skip decoder(0.65)와 R00(0.66)보다 낮다. 이는 direct decoder의 R2 Obj D17+ 손실과 같은 방향이다.

**PACO와의 연결(Q7)**
- 방법 간 rank 상관(n = 9)은 R2 M0 vs PACO-val loss ρ = +0.80, vs token cos ρ = −0.80이다. Self-sim diff는 −0.13이다. 즉 **teacher에 더 가깝게 복원한 방법일수록 R2 성능이 낮았다.**
- R0에서는 상관이 약하다(±0.30).
- 같은 feature loss로 학습했을 때 MSE + cos를 낮추는 가장 쉬운 방법은 공간 구조를 평균 쪽으로 줄이는 것이고, R2는 이것을 손해로 읽는다. 이 해석은 지표들과 일치하지만 따로 검증하지는 않았다.

## 5. 질문별 답

- **Q1. Bicubic spatial refinement(E1)가 도움이 되었는가?**
  - R2에서는 **아니다.** E1 − Bicubic = −0.58 [−1.33, −0.04]이고, 큰 ROI에서 손실이 있다(Obj D17+ −1.40).
  - R0에서는 +0.97로 유의하다. Teacher 복원은 크게 좋아졌다(token cos 0.35 → 0.51).
  - 즉 "Bicubic보다 teacher에 가까운 feature"는 만들었지만, 강한 reader의 속성 판별은 개선하지 못했다.
- **Q2. 원본 token을 cross-attention으로 읽는 것이 R00보다 유리했는가?**
  - R2에서 E2/E3/E4/E7 − R00는 +0.16 ~ +0.58로 모두 n.s.다. E6만 +1.31 [+0.35, +2.25]로 유의하다.
  - 그러나 R00 자체가 Bicubic보다 −1.32 낮으므로, 이는 "덜 나빴다"는 뜻이다.
  - R0에서는 오히려 R00가 E2/E3/E4/E7보다 1.2–2.1 높다.
  - 같은 프로토콜의 R00 대비 cross-attention 구조의 일관된 이득은 확인되지 않았다.
- **Q3. ROI memory보다 full memory가 유리했는가?**
  - **아니다.** E3 − E2 = −0.32 [−0.89, +0.17](R2), −0.29 [−0.59, −0.02](R0). 복원 지표도 거의 같다.
  - Full memory 접근(576 token)이 이득을 주지 않았다. 외부 문맥의 의미적 활용 여부는 이 실험으로 판단할 수 없다.
- **Q4. Mask query vs Bicubic query.**
  - Bicubic query 쪽이 방향은 일관되게 좋다. R2 +0.42(no skip), +0.46(skip)이지만 Overall은 둘 다 n.s.다.
  - Skip이 있을 때 Obj D5–16 +1.20 [+0.31, +1.96]이다.
  - R0에서는 +0.75와 +0.41로 둘 다 유의하다.
- **Q5. Bicubic을 query에 넣는 것과 output skip으로 쓰는 것 중 무엇이 유리했는가?**
  - Skip 효과(+0.70 / +0.73, R2 n.s.; R0 +1.79 / +1.45 유의)가 query 효과(+0.42 / +0.46; R0 +0.75 / +0.41)보다 크다.
  - E5 − E4는 사전 대비 목록에 없어 CI를 계산하지 않았고, 점추정은 R2 +0.28(42.25 − 41.97)이다.
  - 두 가지를 함께 쓰는 E6이 모든 reader에서 최상위이고, interaction은 0이다(R2 +0.03).
  - Skip 대비에는 skip 연산 자체와 **초기 출력이 Bicubic에서 시작하는 효과**(zero-init delta)가 함께 들어 있다. Direct 조건은 zero-init을 쓰지 않았다.
- **Q6. Query self-attention이 추가 이득을 주었는가?**
  - **아니다.** E7 − E3 = −0.20 [−0.75, +0.48](R2), −0.13(R0). 복원도 거의 같다(0.552 vs 0.548).
  - +0.53M params, +1.3 GFLOPs/ROI를 추가한 이번 구성의 결과다.
- **Q7. Teacher 복원 개선이 PACO 개선으로 이어졌는가?**
  - **아니다.** 4절 참조. 방법 간 순서는 오히려 반대였다(ρ = +0.80, feature loss vs R2 M0).
- **Q8. Direct Full, Bicubic, Full Multi-layer 대비 경쟁력이 있는가?**
  - **없다.** 최선인 E6이 Direct Full이나 Bicubic과 같은 수준이다(+0.07 / −0.01).
  - Full ML보다 −2.31 [−3.33, −1.44], C0보다 −2.74 낮다.
  - Full ML은 입력 정보가 더 많아 순수한 구조 비교는 아니지만, 실용 기준선으로서 격차가 크다.
- **Q9. 비용 대비 이득이 있는가?**
  - **없다.** E 계열은 1.98–2.51M params, 2.6–4.1 GFLOPs/ROI, 32 ROI당 2.2–3.3 ms(H200, BF16, decoder만), 추론 추가 메모리 0.2 GB 수준이다. R00는 10.2M params, 10.2 GFLOPs/ROI다.
  - R2 이득이 없으므로 이 비용은 정당화되지 않는다.
  - 참고로 Bicubic과 Direct Full은 decoder 비용이 0이다.
- **Q10. 기존 S-series와 비교해 무엇이 달라졌는가?**
  - 연산과 정보 접근 측면에서 이번 실험은 다음이 다르다.
    - 입력: L12 하나만 썼다(S-series는 L4/L8/L12 multi-depth 융합).
    - Loss: raw-token RAEv2 loss를 썼다(S는 teacher 정규화 multi-depth loss + aux head).
    - Memory: ROI token만(E2)과 전체 576 token(E3–E7)을 나눴다(S는 ROI + ring).
    - Query: mask와 Bicubic query를 분리했다(S는 bilinear grid feature 또는 box query 16개).
    - 출력: direct와 Bicubic skip을 분리했다(S는 모두 bilinear residual).
    - 크기: d 256 / 2 block, 약 2M params(S는 d 384 / 4 block, 약 11M).
  - S-series(A0 43.21, A1 43.53 등)는 Bilinear(42.30)보다 높았지만, 프로토콜이 달라서 이번 수치와 직접 차이를 계산하지 않았다.
  - 이번 결과에서 확인할 수 있는 것은 "출력이 보간 feature에서 출발하는가"(skip)가 query나 memory 종류보다 중요한 요인이었다는 점뿐이다(R0에서 유의, R2에서는 같은 방향이지만 n.s.).

## 6. 해석 범위와 주의

- **해석 범위:** 이번 설정(final-layer 입력, raw feature loss, d 256 / 2 block, 12k step, seed 0)에서 **decoder 구조 변경의 R2 이득은 확인되지 않았다.**
  - Shared feature에 정보가 없다거나 decoder 방식 전체가 불가능하다는 뜻은 아니다.
  - Capacity(d 256) 한계도 배제할 수 없다.
- **Dev loss:** 12k에서도 dev loss가 아주 조금 줄고 있었다. Step을 늘린 효과는 검증하지 않았다.
- **Reader 차이:** R0와 R2의 결론이 다르다. R0 이득(skip, Bicubic query)은 pooled 평균에 유리한 변화로 보이고, R2에서는 재현되지 않는다.
- **Metric과 기준:** M0와 M2는 정의가 달라 따로 보고했다. M2의 소규모 bin(Obj D5–16 M2 등)은 CI가 매우 넓다. 크기 bin은 IVC와 같은 F336 `n_in` 기준이다.
- **R00와 B0는 이번에 새로 만든 기준선이다.**
  - R00: 이 서버에 R00 checkpoint가 없어서 같은 프로토콜로 다시 학습했다(fresh init).
  - B0: 같은 프로토콜의 final-only Direct Full 결과가 없어서 새로 probe했다.
- **프로토콜에서 바뀐 점**
  - Feature reconstruction은 QAT 재개 후 CPU fp32로 실행했다(GPU 실행이 numpy 변환 버그로 실패).
  - Self-similarity 지표는 valid cell 전체(M)에서 계산했다(IVC는 완전 피복 직사각형 M1).
  - 분석과 보고서(CPU)는 QAT 재개 후에 실행했다.
- **좌표 정렬 smoke**(`smoke_cpu/alignment.json`)
  - Query/memory 좌표 구현 오차는 ≤ 1e-5 px다.
  - Bicubic의 선형 ramp 오차는 내부 0.67 px(IVC와 같음)이고, canvas 경계 1–2 cell에서는 최대 1.35 px다. Kernel과 border clamp의 성질이고 grid 오류가 아니다.
- **Mask 정책:** E2 ROI token은 n_in ≥ 1이 보장되어 fallback을 0건 적용했다. 크기 때문에 평가에서 뺀 ROI는 없다.

## 7. QAT 중단·재개 기록

| 항목 | 값 |
|---|---|
| QAT | `/home/irteam/low-bit-training-v2` 800m-ste-code7.5e-5-compile, torchrun port 30871, GPU 2,3 (GPU 0,1의 다른 QAT run은 건드리지 않음) |
| 1차 중단 | 16:11:58. 사용자 승인으로 checkpoint를 기다리지 않음. **step 22,468에서 중단, 재개 기준 checkpoint-20000 → 2,468 step 유실(재학습 필요)** |
| 1차 자동 재개 | 16:13. GPU smoke 실패 후 exit trap이 실행했으나 프로세스가 출력 없이 즉시 종료됨(원인 미확정). 16:16에 수동 재개해 step 20,032까지 증가를 확인 |
| 2차 중단 | 16:18:57, step 20,045 (checkpoint-20000 이후 45 step 유실) |
| 우리 GPU 사용 | 16:21:58 – 19:12:46 (≈ 2 h 51 m × H200 2장 ≈ 5.7 GPU-h, smoke 포함) |
| 재개 | 19:12:46. tmux **`qat_800m`**, 저장된 cmdline(`--resume` 포함)과 env, cwd 그대로. **checkpoint-20000**(model, optimizer, scheduler, rng, qat-rank-state)에서 시작 |
| 재개 검증 | 19:15:47에 step 20,051 > 20,000, loss 3.427 로그, torchrun 1433362, GPU 2/3 90/80 GB 94% → **OK** ([`qat_resume_status.md`](qat_resume_status.md)) |
| 현재 (20:08) | step **22,080** (≈ 1.59 s/step), loss 3.362, GPU 2/3 90.4 / 80.2 GB, util 100%, tmux `qat_800m` 유지. 원래 목표 max_steps 305,176 그대로. 다음 저장 checkpoint-25000 ≈ 21:25 |
| 로그 | `.../800m-ste-code7.5e-5-compile/train.log`에 이어 씀 (`===== resumed by xdec queue` 표시 뒤) |

- **그대로 둔 것:** 2차 중단 때 생긴 zombie(`[pt_elastic]` pid 1394293)는 GPU 메모리를 쓰지 않는다. 컨테이너 init이 회수하지 않아 프로세스 목록에만 남아 있고, 실행 중인 QAT에는 영향이 없다.
- **삭제하지 않은 것:** QAT의 checkpoint, 로그, config는 삭제하거나 덮어쓰지 않았다. checkpoint-15000은 QAT 자신의 save_total_limit = 2 규칙에 따라 25000이 저장될 때 지워진다.
- **재시도 명령:** `bash output/2026-11-05/xdec/scripts/recover_qat.sh`

## 8. 산출물

`output/2026-11-05/xdec/`:
- **문서:** `architecture_audit.md`, `protocol.md`, `REPORT.md`(이 문서), `REPORT_auto.md`(규칙 기반 초안), `tables.md`, `qat_resume_status.md`
- **설정:** `configs/` (probe yaml)
- **checkpoint:** `checkpoints/<cond>/`(3k/6k/9k/12k, `selected.pt`, `train_log.jsonl`, `grad_check.json`, `done.json`)
- **결과:** `results/probe_<cond>/`(reader 매 epoch ckpt, val/dev score), `results/recon/`(ROI별 복원 지표), `results/analysis.json`
- **그림:** `figures/`(train/dev 곡선, val M0 막대)
- **smoke:** `smoke_cpu/`, `smoke_gpu/`
- **QAT 기록:** `qat_pause/`(cmdline, env, 중단 기록 2회)
- **큐 관리:** `logs/`(events.tsv 등), `queue_status.tsv`, `scripts/`

코드는 `src/xdec/`(geometry, models, pretrain, probe, smoke, recon, analyze, report)에 있다. tmux 세션은 `xdec_q`(종료됨)와 `qat_800m`(QAT 실행 중)이다.
