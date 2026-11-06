# EVA-CLIP / CLIPSelf / DeCLIP — PACO Fine-grained Region Representation (2026-11-06)

**결론 (PACO test-primary M2, R2 reader, seed 0, 입력 1024 / 64×64 token)**
1. CLIPSelf는 원본 EVA-CLIP보다 **+13.39 [+12.16, +15.33]** 높다. 모든 열(크기·attribute 종류)에서 95% CI가 0을 넘는다.
2. DeCLIP도 EVA-CLIP보다 **+10.87 [+9.89, +12.99]** 높지만, CLIPSelf보다는 **−2.53 [−3.60, −1.07]** 낮다. 단, 1024는 CLIPSelf의 학습 해상도이고 DeCLIP은 560에서 학습됐다(아래 §4 주의).
3. 개선은 작은 Object에서 가장 크다: Obj 5–16에서 CLIPSelf +18.78, DeCLIP +14.37. Part 1–4는 +11.38 / +9.83으로 Overall과 비슷하거나 약간 작다.
4. EVA-CLIP GT crop 재인코딩 오라클(@224)과 비교하면, CLIPSelf와의 Overall 격차는 사실상 0이다(−0.05 [−1.35, +1.29]). Obj 5–16은 오라클이 +2.84로 앞서지만 유의하지 않다. DeCLIP과의 격차는 2.48(유의)로 남는다. 원본 EVA-CLIP은 오라클보다 13.34 낮다.

## 1. 결과 (PACO test-primary, M2 = 기본 지표)

Test-primary: decoder COCO pool 밖의 test ROI 26,442개 / 이미지 4,848장(object 7,493, part 18,949). 크기 bin은 기존 manifest의 n_in(DINOv2 24×24 grid 기준)을 그대로 썼다.

| Model | Overall | Obj 5–16 | Part 1–4 | Color | Material | Pattern |
|---|---:|---:|---:|---:|---:|---:|
| EVA-CLIP | 30.83 | 28.06 | 30.93 | 33.04 | 26.14 | 29.78 |
| CLIPSelf | **44.22** | **46.84** | **42.31** | 40.85 | **36.82** | 47.76 |
| DeCLIP | 41.69 | 42.43 | 40.76 | 39.28 | 34.80 | 46.78 |
| *Oracle: EVA-CLIP GT crop @224 (별도 대조군)* | *44.17* | *49.67* | *36.24* | *41.77* | *35.85* | *50.26* |

Paired image bootstrap(B=1,000, seed 0, 모든 모델이 같은 draw). `*`는 95% CI가 0을 포함하지 않는다는 뜻이다.

| Contrast | Overall | Obj 5–16 | Part 1–4 | Color | Material | Pattern |
|---|---|---|---|---|---|---|
| CLIPSelf − EVA-CLIP | +13.39 [+12.16, +15.33]* | +18.78 [+15.08, +23.22]* | +11.38 [+6.38, +14.90]* | +7.81 [+6.56, +9.04]* | +10.67 [+9.22, +14.28]* | +17.98 [+14.05, +21.83]* |
| DeCLIP − EVA-CLIP | +10.87 [+9.89, +12.99]* | +14.37 [+9.70, +19.56]* | +9.83 [+4.93, +12.74]* | +6.25 [+5.23, +7.63]* | +8.65 [+7.17, +12.09]* | +17.00 [+12.52, +20.30]* |
| DeCLIP − CLIPSelf | −2.53 [−3.60, −1.07]* | −4.41 [−8.09, −0.86]* | −1.55 [−5.39, +1.75] | −1.57 [−2.41, −0.25]* | −2.02 [−4.33, −0.21]* | −0.98 [−4.03, +1.45] |
| Oracle − EVA-CLIP | +13.34 [+12.20, +15.36]* | +21.61 [+18.31, +26.13]* | +5.32 [+0.63, +9.01]* | +8.73 [+7.74, +10.23]* | +9.71 [+7.61, +13.09]* | +20.48 [+17.02, +24.40]* |
| Oracle − CLIPSelf | −0.05 [−1.35, +1.29] | +2.84 [−1.10, +6.94] | −6.07 [−9.70, −2.42]* | +0.92 [+0.02, +2.32]* | −0.97 [−4.03, +0.83] | +2.50 [−0.24, +5.91] |
| Oracle − DeCLIP | +2.48 [+0.86, +3.89]* | +7.24 [+2.94, +12.28]* | −4.51 [−7.72, −0.77]* | +2.49 [+1.39, +3.80]* | +1.05 [−1.85, +3.12] | +3.48 [+0.37, +8.02]* |

나머지 열(Object, Part, Obj 17+, Part 5–16, Part 17+), M0(참고 지표), PACO val, exposure 민감도 표는 `final_tables.md`에 있다. 기계가 읽을 수 있는 전체 값은 `results.csv`(660행)에 있다.

| Model | 선택 epoch (dev M0) | val M2 (선택 epoch) | 학습 시간 |
|---|---|---|---|
| EVA-CLIP | 8 (41.25) | 31.94 | 37분 |
| CLIPSelf | 8 (53.12) | 46.34 | 38분 |
| DeCLIP | 9 (51.69) | 43.73 | 50분 |
| Oracle crop @224 | 5 (51.23) | 46.72 | 14분 |

**참고만 (이번 결과가 아님):** 기존 DINOv2-B/14 실험(IVC, 336 입력, 24×24)의 test-primary M2 R2는 Bilinear 40.42, C0 crop 45.92였다(`experiments/ivc/TEST_REPORT.md`). Encoder, 해상도, reader 입력 방식(crop-grid sampling vs full grid)이 모두 달라서 직접 비교할 수 없다.

## 2. 질문별 답

**Q1. CLIPSelf가 원본 EVA-CLIP보다 fine-grained attribute 예측을 개선하는가?**
그렇다. Overall +13.4이고, Color +7.8, Material +10.7, Pattern +18.0이 모두 유의하다. COCO val2017 이미지만 쓴 부분집합(CLIPSelf가 학습에서 본 적 없는 이미지)에서도 +14.50 [+11.79, +18.91]으로 유지된다.

**Q2. DeCLIP이 CLIPSelf보다 더 좋은 region representation을 주는가?**
이번 조건(1024)에서는 아니다. Overall −2.53으로 유의하게 낮다. 다만 두 가지를 같이 봐야 한다.
- 1024는 CLIPSelf의 학습 해상도이고 DeCLIP은 560에서 학습됐다. Zero-shot box 분류에서도 1024에서는 CLIPSelf 71.3 > DeCLIP 68.0이었지만, 560에서는 DeCLIP 73.0 > CLIPSelf 69.4로 뒤집혔다.
- 560 비교는 사용자 결정으로 생략했으므로, "DeCLIP이 일반적으로 더 나쁘다"고 말할 수는 없다. 말할 수 있는 것은 "공통 해상도 1024에서는 CLIPSelf가 더 좋다"까지다.
- COCO val2017 부분집합에서는 −1.18 [−4.68, +2.61]로 유의하지 않다.

**Q3. 개선이 작은 Object / Part에서 특히 나타나는가?**
- 관찰: Obj 5–16(+18.8 / +14.4)이 Overall(+13.4 / +10.9)보다 크다. Obj 17+는 +15.1 / +12.9, Part 1–4는 +11.4 / +9.8(CI가 넓음), Part 5–16은 +15.5 / +11.5다.
- 해석: 작은 object에서 특히 크지만, 효과가 작은 영역에만 몰린 것은 아니고 전 구간에서 크다. 작은 part(1–4)에서는 Overall보다 오히려 작다.
- 주의: val2017 부분집합에서 Part 1–4 M2 unit은 12개뿐이라 그 열은 해석하지 않는다.

**Q4. 기존 dense representation 학습만으로 crop 재인코딩과의 격차가 줄어드는가?**
줄어든다. EVA-CLIP full image와 오라클의 격차는 13.34인데, CLIPSelf에서는 −0.05(n.s.)로 사라지고 DeCLIP에서는 2.48로 남는다.
- Obj 5–16에서는 오라클이 아직 앞선다(CLIPSelf 대비 +2.84 n.s., DeCLIP 대비 +7.24 유의).
- Part 1–4에서는 오히려 CLIPSelf와 DeCLIP이 오라클보다 높다(−6.07*, −4.51*).

이 결과에는 세 가지 단서가 붙는다.
1. 오라클은 EVA-CLIP 고유 해상도인 224에서 crop을 다시 인코딩한 것이고 token은 14×14다. 반면 full image는 1024(64×64)라서, 작은 Part ROI도 15–33개 token을 갖는다(verify/geometry_check.json). 즉 "full 1024 vs crop 224" 비교이며, 해상도를 맞춘 crop 상한선이 아니다. Part 1–4에서 오라클이 진 것은 아주 작은 crop을 224로 확대하면서 문맥이 사라진 영향일 수 있다(검증 안 함).
2. 원본 EVA-CLIP은 1024에서 positional embedding과 RoPE를 14→64로 보간해서 매우 약하다(zero-shot 17.5 vs 560에서 42.3). 그래서 "EVA → 오라클" 격차 13.3의 상당 부분은 고해상도 일반화 실패일 수 있다.
3. CLIPSelf의 학습 신호 자체가 "dense ROI feature ← crop CLS feature" 정렬이므로, crop과 비슷한 결과가 나오는 것은 설계 의도와 일치한다.

**Q5. 우리 Encode Once, Ask Later 연구에 주는 시사점**
- 관찰: 한 번 인코딩한 full image dense map에서 ROI token만 읽어도, encoder를 region 단위 self-distillation으로 미리 학습해 두면 crop 재인코딩 오라클 수준(Overall)에 도달했다. 우리 post-hoc decoder들(xdec / svamp)은 Bicubic 대비 ±1 안쪽에 머물렀던 것과 대조된다.
- 해석: 격차는 "token을 나중에 다시 만드는 것"보다 "encoder가 처음부터 crop과 정렬된 local feature를 만들도록 학습하는 것"으로 더 잘 닫혔다. 이는 같은 encoder에 crop 정렬 학습 신호를 주는 방향(학습 시점에서 encoder 쪽 해결)을 지지한다.
- 다만 이번 결과는 1024 입력(token 4,096개)에서 나왔다. 해상도를 올린 효과와 학습 효과를 분리하려면 같은 해상도 비교(예: 336/448)가 필요하다. 사용자가 경계하는 "해상도만 올리기"와 겹칠 수 있으므로, 결과를 인용할 때 이 점을 함께 밝혀야 한다.
- 원본 EVA-CLIP은 dense token 자체가 약해서(1024 test M2 30.8), DINOv2 대신 EVA 계열로 바꾸는 것 자체가 이득이라고 볼 근거는 없다.

## 3. 모델 출처와 checkpoint

| 모델 | 저장소 (commit) | Checkpoint (SHA256) | 출처 URL | 구조 / 학습 데이터 |
|---|---|---|---|---|
| EVA-CLIP | – | EVA02_CLIP_B_psz16_s8B.pt (`4aab21dd652dc19aab7ff2302e5a07582d7838e8a3b26c80d697115fb0c6ddf0`, HF와 일치) | huggingface.co/QuanSun/EVA-CLIP @11afd20 | EVA02-CLIP-B/16 (ViT-B, 86M, patch 16, RoPE, SwiGLU, sub-LN, embed 512). 이미지 tower는 EVA02-B MIM(IN-21K) 초기화, CLIP 학습은 Merged-2B(LAION-2B 1.6B + COYO-700M 0.4B) |
| CLIPSelf | wusize/CLIPSelf @`1c7fe9c` | eva_vitb16_coco_clipself_patches.pt (`9a8e9109efbbecd3c77c72d5f5769b95c25f98c8a75df837b0da6e6a005732a1`, 출처에 공개 hash 없음) | README의 Google Drive 링크 8개와 폴더 2개는 모두 404. 저자가 issue #32/#38에서 안내한 OneDrive(`entuedu-my.sharepoint.com/.../opensource/clipself/clipself_checkpoints/`, 2024-10-16 업로드)에서 받음 | 같은 구조. COCO train2017 **이미지만**(box annotation 없음) 사용, 1024 입력, 6 epoch. Teacher는 원본 EVA-CLIP의 224 crop CLS. Visual block 12개 학습 (`--lock-image-unlocked-groups 12`) |
| DeCLIP | xiaomoguhz/DeCLIP @`6cbda51` | evab_dinov2B_csa_560_0.25_seg.pt (`2196da6579e23df4f756617d98f46d46c605c630491966e106e0306e813397c4`, HF와 일치). 내부 이름은 `evab_dinov2B_qq_560_0.25`, epoch 6 | huggingface.co/xiaomoguhzz/DeCLIP_evab_dinov2B_csa_560_0.25_seg @3984d86 (repo ZSSS branch에서 링크) | 같은 구조. COCO train2017 이미지(annotation 없음), 560 입력, VFM teacher DINOv2-B, context = Q, λ = 0.25 (논문 기본 설정) |

- **학습 완료 확인:** CLIPSelf는 visual tensor 260개 중 249개, DeCLIP은 242개가 원본과 다르다. 12개 block 모두 바뀌었고, patch_embed / pos_embed / norm / head와 text tower는 그대로다(verify/param_diff.json). 따라서 초기화 가중치가 아니라 학습된 가중치다.
- **로드 확인:** 공식 `eva_clip.create_model`로 로드한 visual tensor 286개가 checkpoint와 bit 단위로 같다. 누락·불일치·예상 외 key는 0개다. RoPE buffer는 공식 loader가 다시 계산한다(차이 ≤ 2.4e-4).
- **환경 재현 확인:** COCO val2017 instance box zero-shot 분류(RoIAlign, 133 클래스, 1024)에서 EVA 17.5 / CLIPSelf 71.3이 나왔다. 논문 Table 2는 18.2 / 72.1(panoptic boxes)이므로, 이 환경(torch 2.7, fp16, SDPA)이 공식 feature를 재현한다(verify/zeroshot_*.json).
- **DeCLIP 상태:** 학습 완료 checkpoint를 확보했으므로 재학습은 필요 없다. 다른 DeCLIP 가중치 두 개는 쓰지 않았다. CAT-Seg branch 것은 재학습본에 csa 모드이고 loss weight가 기본값이 아니다. DeCLIP2는 TPAMI 확장판이다.

## 4. Feature 추출과 검증

```
Full image → letterbox 1024 (가운데 정렬, mean-color padding, OpenAI-CLIP mean/std, PIL bicubic)
   → frozen encoder (공식 encode_dense, fp16 autocast) → 64×64×512 L2-normalised token map (CLS 제외)
   → R2: GT box와 겹치는 cell(cov > 0)만 attention mask로 읽음. Encoder 안에서는 이미지 전체 attention을 유지
```

- **EVA-CLIP / CLIPSelf:** CLIPSelf `encode_dense`(extract v2) 경로. Block 1–11은 그대로 쓰고, 마지막 block은 attention mixing 없이(v-proj + MLP) 통과시킨 뒤 LN → head(768→512) → L2.
- **DeCLIP:** `encode_dense(mode="qq")` 경로. 마지막 block은 softmax(qqᵀ)v이고 residual과 FFN이 없으며, 이어서 LN → head → L2. 이것이 **content feature**이며, DeCLIP의 zero_shot.py가 qq_vfm_distill 모델에 쓰는 inference 모드다.
  - Context feature(q)는 attention map을 만드는 데만 쓰이고 따로 출력되지 않는다. 그래서 둘을 임의로 합친 secondary 평가는 하지 않았다.
- **오라클:** 원본 이미지에서 manifest 정수 box를 crop하고 224 letterbox로 넣은 뒤 EVA-CLIP v2 경로를 통과시켜 14×14 token을 얻는다. R2는 crop-grid geometry로 읽는다(IVC C0와 같은 규칙).
- **Reader:** 기존 R2(`src.readout.models.Reader('r2')`)에 입력 어댑터 LN(512)→Linear(512, 768)만 추가했다. 나머지 학습 설정은 기존 프로토콜 그대로다.
  - Optimizer / LR / 10 epoch / batch / seed 7 / dev M0로 epoch 선택.
  - Test는 선택된 checkpoint로 한 번만 채점했다. 그 전에 val을 다시 채점해 저장된 값과 정확히 같은지(최대 차이 0.0) 확인했다.

검증 항목:

| 항목 | 결과 |
|---|---|
| 기존 DINOv2 geometry 재현 | 3,000 ROI와 40 canvas에서 불일치 0 (verify/geometry_check.json) |
| 1024 좌표 정렬 | 원본 px에서 box를 rasterize해 cell 비율을 구하면, R2가 쓰는 cov와 최대 0.013 차이 |
| ROI token 수 (1024 grid) | 0개인 ROI 없음. D1–4는 15–33개, D5–16은 18–104개 |
| 샘플 출력 | shape / token 수 / 좌표 / 경로: verify/samples_*.json, 정렬 그림: verify/align_*_1024.png |
| Encoder frozen | 학습 전후 checksum이 같음 (probes/*/frozen_check.json) |
| Train/val/test 이미지 겹침 | 없음 (assert) |
| Label / metric | 기존 `Boot`(eval_tables M2 / M0). Unit 수가 IVC test 보고서와 같음 (4173 / 113, D1–4 508 / 72, D5–16 1521 / 100, D17+ 1620 / 110) |

## 5. 데이터 노출

- PACO의 train / val 이미지는 모두 COCO train2017이고, test-primary는 2,959장이 train2017, 1,889장이 val2017이다.
- CLIPSelf와 DeCLIP은 COCO train2017 이미지(annotation 없음)로 self-distillation을 했으므로, test-primary 이미지의 61%를 label 없이 본 셈이다.
- EVA-CLIP의 web 데이터(LAION / COYO)와 COCO가 겹치는지는 통제할 수 없다. DeCLIP teacher인 DINOv2의 LVD-142M도 마찬가지다.
- COCO val2017 부분집합(10,215 ROI / 1,889장)에서도 결론 방향은 같다.
  - Overall: CLIPSelf − EVA +14.50*, DeCLIP − EVA +13.32*, DeCLIP − CLIPSelf −1.18 (n.s.)
  - 오라클과의 차이: CLIPSelf 대비 +1.15 (n.s.), DeCLIP 대비 +2.33 (n.s.)
- PACO의 attribute label 자체는 어떤 모델의 학습에도 쓰이지 않았다.

## 6. 기존 계획과 달라진 점

- **Encoder 정밀도:** bf16 대신 fp16 autocast를 썼다(repo 기본값 `amp`와 같음). bf16에서는 EVA dense token의 3%가 fp32 대비 cos < 0.99였고 최소 0.32까지 떨어졌다(verify/precision_check.txt).
- **Letterbox:** 공식 평가의 ResizeLongest(왼쪽 위 정렬, 검은색 padding) 대신 우리 PACO letterbox(가운데 정렬, mean-color padding)를 세 모델 모두에 똑같이 적용했다. 이 설정에서도 zero-shot 재현 차이는 1 point 안쪽이었다.
- **560 보조 해상도:** 사용자 요청으로 생략했다. 그래서 DeCLIP의 학습 해상도에서는 비교하지 않았다.
- **Seed:** 0 하나만 썼으므로 CI는 평가 표본의 불확실성만 반영하고, 학습 seed 분산은 포함하지 않는다.

## 7. 파일과 실행 명령

- 코드: `src/densecmp/` — geom, encoders, data, train, test_eval, report, verify, zeroshot
- 환경: `/home/irteam/miniforge3/envs/evaclip_eval` (dinov2_ord_py310을 clone한 뒤 einops / ftfy / regex / gdown 추가)
- 공식 repo와 checkpoint: `/home/irteam/ext_repos/{CLIPSelf,DeCLIP,checkpoints}` (checkpoints/SHA256SUMS)
- 이 폴더:
  - `protocol.md`: 실행 전 고정한 설정과 변경 기록
  - `final_tables.md`, `results.csv`, `results.json`: 결과
  - `probes/<model>/`: config, dev/val curve, epoch별 reader, scores_test.npy, val_recheck, frozen_check
  - `verify/`: 검증 결과
  - `logs/`: 실행 로그
  - `scripts/`: queue와 QAT pause/resume
- QAT: 02:12:55 UTC에 step 33,796에서 강제 중지했고, 13:50 KST(04:50 UTC)에 checkpoint-30000부터 재개했다(qat_resume_status.md: OK, step 30054).

```bash
PY=/home/irteam/miniforge3/envs/evaclip_eval/bin/python
CUDA_VISIBLE_DEVICES=2,3 $PY -m torch.distributed.run --nproc_per_node 2 -m src.densecmp.train --enc {eva|clipself|declip} --mode full --side 1024 --out <run>/probes/<enc>_full1024
CUDA_VISIBLE_DEVICES=2,3 $PY -m torch.distributed.run --nproc_per_node 2 -m src.densecmp.train --enc eva --mode c0 --side 224 --out <run>/probes/eva_c0224
CUDA_VISIBLE_DEVICES=2 $PY -m src.densecmp.test_eval --probe <probe dir>          # one-time test
python -m src.densecmp.report --run <run>                                          # tables (CPU)
$PY -m src.densecmp.verify {geometry|params|samples} --out <run>/verify ;  $PY -m src.densecmp.zeroshot --enc <enc> --out <run>/verify
```
