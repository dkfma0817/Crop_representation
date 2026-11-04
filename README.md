# Crop_representation

작은 ROI의 full-image feature를 GT crop을 다시 인코딩한 것 같은(crop-like) representation으로 바꿀 수 있는지 보는 실험 기록입니다.
평가는 PACO attribute(test-primary, M2)로 합니다.

## 실험 목록

| 폴더 | 날짜 | 내용 | 결론 |
|---|---|---|---|
| [`experiments/raev2_crop_recon/`](experiments/raev2_crop_recon/) | 2026-11-04 | frozen RAEv2 pixel decoder를 통한 pixel supervision으로 CoordMLP 학습. A = feature loss만, B = pixel loss만, C = feature + pixel | 복원 지표는 좋아졌지만 PACO 이득은 없음. C−A는 R2 +0.62로 seed 편차(0.49) 수준이고 R0는 +0.11. GT crop 상한과의 격차(Obj 5–16 약 −20)는 그대로 |

보고서는 각 실험 폴더의 `README.md`, 사전 등록 프로토콜은 [`configs/raev2_crop_recon/protocol.md`](configs/raev2_crop_recon/protocol.md)에 있습니다.

## 폴더 구조

```
Crop_representation/
├── src/                        # 코드 (python -m src.<패키지>.<모듈> 로 실행)
│   ├── config.py               # 모든 경로 (데이터셋, cache/, third_party/, 이전 프로젝트 경로). 환경변수로 덮어쓰기 가능
│   ├── rae/common.py           # 공식 RAEv2 encoder/decoder 래퍼(FrozenRAE), 256 letterbox, ROI geometry
│   ├── data/                   # make_rois → make_stream → extract_cache (데이터 준비), dataset.py (학습용 Dataset), relation.py (stream sampler)
│   ├── models/coordmlp.py      # CoordMLP (MLRDecoder)
│   ├── losses/                 # feature.py (MSE/cos), pixel.py (L1 + LPIPS, RGB 지표)
│   ├── engine/                 # train.py (A/B/C 학습, make_init), calib.py (β, λ 보정)
│   ├── eval/                   # sanity, verify_pair, refs, recon, probe(PACO reader), paco(M2 AP), s0_table, hval_teacher_fullpad
│   ├── viz/                    # analysis, large, plot_loss
│   └── utils/rcp.py            # rae_crop_pilot에서 가져온 RGB 지표 (hp_corr 등)
├── scripts/raev2_crop_recon/   # 실행 스크립트 (run_extract_local.sh = 이 서버용, 나머지는 원래 서버에서 돌린 기록)
├── configs/raev2_crop_recon/   # protocol.md
├── experiments/raev2_crop_recon/  # 보고서 README.md, results/, figures/
├── env/                        # requirements.txt, setup_env.sh
├── cache/                      # (git 제외) 모델 가중치, ROI 목록, feature 캐시, runs, probe
├── third_party/RAEv2/          # (git 제외) 공식 RAEv2 코드
└── logs/                       # (git 제외) 실행 로그와 flag
```

## 실행

모든 명령은 repo 루트에서 실행합니다.

```bash
source ~/miniforge3/bin/activate dinov2_ord_py310        # 설치: env/README.md

# 1) 데이터 준비 (한 번)
python -m src.data.make_rois                             # cache/data/  ROI 목록, PACO val/test 제외 목록
python -m src.data.make_stream                           # cache/stream/ 12k step 학습 순서
python -m src.data.extract_cache plan                    # cache/feat/  캐시 index
GPUS="2 3 2 3" EXTRACT_WORKERS=28 EXTRACT_BS=64 \
  nohup bash scripts/raev2_crop_recon/run_extract_local.sh > logs/run_extract_local.log 2>&1 &   # 캐시(73 GB) + 검증 + init

# 2) 학습 (A / B / C)
CUDA_VISIBLE_DEVICES=2 python -m src.engine.train --run C --w_feat 1 --lam 0.1 --beta 0.2 --micro 16 --out cache/runs/C
```

## 이 서버에서 새로 만든 것 (2026-11-04)

원래 서버의 데이터 일부는 다른 프로젝트에서 만든 것이라 이 서버에 없습니다. 그래서 아래 항목은 **같은 형식과 규칙으로 새로 만들었고, 원래 것과 같지 않습니다.**
이 서버에서 낸 숫자는 `experiments/raev2_crop_recon/README.md`의 숫자와 직접 비교하면 안 됩니다.

| 항목 | 원래 | 이 서버 |
|---|---|---|
| COCO ROI 목록 (`coco_rois.parquet`) | dinov2_oracle_region_decoder | `src/data/make_rois.py`: train 60,000장 × 4 ROI(GT 3 + random 1), dev 500장 × 4 |
| hval ROI (`rae_rois.parquet`) | rae_crop_pilot head_val 3,194 ROI | 1,000장, GT 최대 4개 → 3,819 ROI |
| PACO val/test 제외 | paco_lp manifest | PACO-LVIS val/test json의 COCO 이미지 id 11,853개 |
| 학습 stream, feature 캐시, init | 같은 코드 | 같은 코드로 다시 생성 (stream 48,829장, 캐시 검증 오차 fp16 반올림 수준) |

생성 기록은 [`experiments/raev2_crop_recon/results/rebuild/`](experiments/raev2_crop_recon/results/rebuild/)에 있습니다.

## 외부 의존

- **RAEv2:** 코드는 [nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2) @ `8a0d238` → `third_party/RAEv2`, 가중치는 HF `nyu-visionx/RAEv2-models` @ `9770b7b` (`stage1/imagenet/dinov2b-k1`) → `cache/raev2_models/`. 라이선스가 CC BY-NC 4.0이라 repo에 넣지 않았습니다.
- **데이터셋:** COCO train2017, PACO-LVIS 어노테이션 (`src/config.py`의 `COCO_TRAIN`, `COCO_ANN`, `PACO_DIR`).
- **아직 없는 것 (이전 프로젝트 코드):** PACO 평가가 쓰는 `paco_region_reader/code`(common_rr), `paco_feat_upsample/code`(eval_fu), `paco_lp` 데이터. 그래서 `src.eval.probe`, `src.eval.paco`, `src.eval.recon`의 PACO 부분, `src.viz.analysis`는 아직 실행되지 않습니다. 경로는 `src/config.py`에서 바꿀 수 있습니다.

## 정정 메모

- `experiments/raev2_crop_recon/README.md` §4 복원 표의 **Teacher (hval) 행**은 padding cell이 0인 캐시 latent를 디코딩한 값입니다.
  GT crop을 padding까지 포함해 다시 인코딩한 올바른 값은 L1 .076 / hp_corr .343이고,
  [`results/recon/hval_teacher_fullpad.csv`](experiments/raev2_crop_recon/results/recon/hval_teacher_fullpad.csv)에 있습니다.
  A/B/C, Bilinear 행과 학습 자체는 이 문제의 영향을 받지 않습니다.
- `figures/loss_curves.png`의 Full(Bilinear)과 GT crop oracle 기준선은 학습 곡선과 같은 COCO dev 2,000 ROI에서 계산했습니다
  (`src/eval/refs.py` → `results/dev_refs.json`, teacher는 padding까지 포함해 다시 인코딩).
