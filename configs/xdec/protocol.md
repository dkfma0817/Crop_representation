# Decoder Architecture Pilot (E1–E7): 실행 전 프로토콜 (2026-11-05, 결과를 보기 전에 고정)

Frozen F336 full-image feature에서 C0 crop feature를 만드는 decoder 구조 7개를 비교한다. Decoder 구조 외의 조건은 모두 고정한다. 구조 상세와 기존 실험과의 차이는 [`architecture_audit.md`](architecture_audit.md)에 있다.

## 고정 조건
- **Encoder / Teacher:** frozen DINOv2-B/14 (timm lvd142m), 336 letterbox, final L12 + final LN, 24×24×768. Teacher는 원본 GT crop(C0 letterbox)을 같은 encoder로 인코딩한 것이다. Teacher는 loss에만 쓰고 decoder 입력에는 넣지 않는다.
- **Bicubic B:** `src.ivc.common.sample_roi(·, 'bicubic')`(IVC와 같은 호출). 하나의 함수를 E1/E4/E5/E6과 B1에서 공유한다.
- **Loss (RAEv2 K=1 Feature-only = R00, 변경 금지):** raw token에서 ROI마다 `L_mse = Σ_p w_p mean_c(Z−T)² / Σ w`, `L_cos = Σ_p w_p(1−cos) / Σ w`, `L = mean_ROI(L_mse + 1.0·L_cos)`. 계산은 fp32, w = C0 valid coverage, teacher는 no_grad. λ_mse = 1, λ_cos = 1. 다른 loss는 없다.
- **학습:** COCO 12k stream(seed 2027, 16장 × 4 ROI = 64 ROI/step, PACO val/test와 겹치는 이미지 2장 제외), flip/augmentation 없음, 12,000 step. AdamW lr 1e-4 (0.9, 0.95), wd 0.05, warmup 600 + cosine, clip 1.0, BF16 autocast. Decoder init은 canonical seed 0 state에서 복사한다(조건 간 공통 module 초기값이 같음). GPU 2,3 DDP(rank당 8장, gradient 합산 = 64-ROI step).
- **Decoder 공통:** d 256, 8 heads, FFN ×4, pre-LN, dropout 0, 2 block, 출력 24×24×768.
  - Direct 출력: `μ12 + σ12·head`. Skip 출력: `B + σ12·head`이고 head를 zero-init한다.
  - Position encoding: ROI-relative (u, v). sym-log로 압축하되 clamp하지 않는다. 그 위에 Fourier feature와 log cell size를 쓰고, 이 PE 하나를 query와 memory가 공유한다.
- **Checkpoint:** 3k/6k/9k/12k를 저장한다. 이 넷 중 **COCO dev feature loss**(468장, 1,872 ROI)가 최소인 것을 쓴다. PACO 결과로는 바꾸지 않는다.
- **PACO probe:** `src.ivc.train`(IVC와 같음). PACO train(중복 6장 제외)에서 PACO dev 1,000장(rng 0)을 epoch 선택용으로 뗀다. R0와 R2(layers [12])를 seed 0으로 같은 process에서 학습한다(10 epoch, masked BCE). Epoch은 dev M0 Overall 최대로 고르고, 최종 평가는 PACO **val**에서 한다. **Test는 평가하지 않는다.**

## 조건
| | Query | Memory | Self-attn | Output |
|---|---|---|---|---|
| E1 | Bicubic | 없음 | SA (CA 없음) | B + Δ |
| E2 | Mask | ROI token (가변 n_in) | – | Direct |
| E3 | Mask | Full 576 | – | Direct |
| E4 | Bicubic | Full 576 | – | Direct |
| E5 | Mask | Full 576 | – | B + Δ |
| E6 | Bicubic | Full 576 | – | B + Δ |
| E7 | Mask | Full 576 | CA → SA → FFN | Direct |
| R00 (B2) | CoordMLP(k=4 local) | ROI+ring | – | μ + σ·MLP |

Baselines: B0 Direct Full(신규 probe), B1 Bicubic(IVC 재사용), B2 R00(같은 프로토콜로 신규 학습), B3 Full ML+R2(MSD 재사용, R2만), B4 C0(IVC 재사용).

## 판정 규칙
- **Primary:** PACO val M0 Overall과 Object D5–16(R2). Secondary는 M2 Overall, Object/Part, Part D1–4, Object D17+, All D17+, Color/Material/Pattern(M0와 M2를 따로), R0 전체.
- **Paired image bootstrap:** 1,000회, seed 0, 모든 조건이 같은 draw, 95% CI.
  - 주요 대비: E1−Bic, E3−E2, E4−E3, E5−E3, E6−E4, E6−E5, E7−E3. 그리고 각 E − {Direct Full, Bicubic, R00}.
  - 2×2: query 효과(E4−E3, E6−E5), skip 효과(E5−E3, E6−E4), interaction (E6−E5)−(E4−E3), 같은 draw로 CI를 구한다.
- CI가 0을 포함하면 차이가 확인되지 않은 것으로 본다. Seed 1개이므로 학습 seed 분산은 반영되지 않는다.
- Feature reconstruction은 COCO dev와 PACO val에서 따로 측정한다(MSE, token cos, pooled cos, centred spatial diff, self-similarity diff; D-bin과 object/part별). PACO 개선과는 분리해서 판정한다.
- Skip 비교에는 "skip 연산"과 "초기 출력이 Bicubic에서 시작하는 효과"가 함께 들어 있다(Direct 조건은 zero-init을 쓰지 않음).
- E7−E3은 SA 추가분(+0.53M params)을 포함한 이번 구성의 효과다.
- 결과가 나쁘더라도 loss, depth, width는 바꾸지 않는다.

## 데이터 누수 확인
COCO stream/dev와 PACO val/test의 겹침(id, md5, pixel md5, dHash)은 `src.msd.pretrain.eval_exclusions`로 제외한다. PACO train과 val의 중복 6장은 `data_check`로 제외한다. PACO dev는 train의 부분집합이고 val과 겹치지 않는다(`src.ivc.train`).
