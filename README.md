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
│   ├── config.py               # configs/paths.yaml 을 읽어 경로 상수로 제공 (환경변수로 덮어쓰기 가능)
│   ├── rae/common.py           # 공식 RAEv2 encoder/decoder 래퍼(FrozenRAE), 256 letterbox, ROI geometry
│   ├── data/                   # make_rois → make_stream → extract_cache (데이터 준비), dataset.py (학습용 Dataset), relation.py (stream sampler)
│   ├── models/coordmlp.py      # CoordMLP (MLRDecoder)
│   ├── losses/                 # feature.py (MSE/cos), pixel.py (L1 + LPIPS, RGB 지표)
│   ├── engine/                 # train.py (A/B/C 학습, make_init), calib.py (β, λ 보정)
│   ├── eval/                   # sanity, verify_pair, refs, recon, probe(PACO reader), paco(M2 AP), s0_table, hval_teacher_fullpad
│   ├── viz/                    # analysis, large, plot_loss
│   └── utils/                  # rcp.py (rae_crop_pilot RGB 지표), yaml_config.py (YAML 로더, _base_ 상속)
├── configs/
│   ├── paths.yaml              # 모든 경로 (데이터셋, cache/, third_party/, 이전 프로젝트 데이터·코드)
│   └── raev2_crop_recon/
│       ├── protocol.md         # 사전 등록 프로토콜
│       └── train/              # base.yaml (공통 레시피) + A / B / C / C_L1 / overfit_{feat,pix}.yaml
├── scripts/raev2_crop_recon/   # 실행 스크립트 (run_extract_local.sh = 이 서버용 캐시 추출, run_main.sh RUN GPU = 학습 + probe)
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

# 1) 데이터 준비 (한 번). ROI 목록 원본은 cache/data/, PACO manifest는 cache/paco_lp/ 에 둡니다 (아래 표)
python -m src.data.make_stream                           # cache/stream/ 12k step 학습 순서
python -m src.data.extract_cache plan                    # cache/feat/  캐시 index
GPUS="2 3 2 3" EXTRACT_WORKERS=28 EXTRACT_BS=64 \
  nohup bash scripts/raev2_crop_recon/run_extract_local.sh > logs/run_extract_local.log 2>&1 &   # 캐시(73 GB) + 검증 + init

# 2) 학습 (A / B / C). 설정은 configs/raev2_crop_recon/train/*.yaml, CLI 인자가 YAML보다 우선
CUDA_VISIBLE_DEVICES=2 python -m src.engine.train --config configs/raev2_crop_recon/train/C.yaml
CUDA_VISIBLE_DEVICES=3 python -m src.engine.train --config configs/raev2_crop_recon/train/A.yaml --steps 30 --out cache/runs/A_smoke   # 일부만 바꾸기
```

- 경로를 바꾸려면 `configs/paths.yaml`을 고치거나, 같은 이름의 대문자 환경변수를 주거나(`COCO_TRAIN=/data/coco/train2017`), `CROP_PATHS=<다른 yaml>`을 지정하세요.
- 새 run 설정은 `_base_: base.yaml`로 공통 레시피를 상속하고 바뀌는 값만 적으면 됩니다 (`C_L1.yaml` 참고).

## 이 서버로 옮긴 데이터 (2026-11-04)

원래 서버에서 다른 프로젝트가 만든 데이터와 코드를 복사해 와서, 아래 위치에 두었습니다 (모두 git 제외).

| 항목 | 원래 서버 | 이 서버 |
|---|---|---|
| COCO ROI 목록 | `dinov2_oracle_region_decoder/perception_cache/data/coco_rois.parquet` | `cache/data/coco_rois.parquet` |
| hval ROI | `rae_crop_pilot/configs/rois.parquet` | `cache/data/rae_rois.parquet` |
| PACO manifest, label | `dinov2_oracle_region_decoder/perception_cache/paco_lp` | `cache/paco_lp/` |
| lpips 0.1.4 | `rae_crop_pilot/_pylib` | `third_party/rae_crop_pilot/_pylib` |
| PACO reader / 평가 코드 | `paco_region_reader/code`, `paco_feat_upsample/code` | `third_party/paco_region_reader/code`, `third_party/paco_feat_upsample/code` |

- 학습 stream, feature 캐시, init은 원본 ROI 목록으로 같은 코드를 돌려 다시 만들었습니다. stream은 protocol과 같은 pool 55,861장 → 50,491장 / 201,964 ROI이고, A 설정 step-0 COCO dev feature loss가 원래 보고서와 같은 1.239로 재현됩니다.
  생성 기록: [`experiments/raev2_crop_recon/results/rebuild/`](experiments/raev2_crop_recon/results/rebuild/).
- 원본이 없을 때를 위한 `src/data/make_rois.py`는 비슷한 ROI 목록을 `cache/data_rebuilt/`에 만듭니다. 원본과 같은 세트가 아니므로 숫자를 직접 비교하면 안 됩니다.

## 외부 의존

- **RAEv2:** 코드는 [nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2) @ `8a0d238` → `third_party/RAEv2`, 가중치는 HF `nyu-visionx/RAEv2-models` @ `9770b7b` (`stage1/imagenet/dinov2b-k1`) → `cache/raev2_models/`. 라이선스가 CC BY-NC 4.0이라 repo에 넣지 않았습니다.
- **데이터셋:** COCO train2017, PACO-LVIS 어노테이션 (`src/config.py`의 `COCO_TRAIN`, `COCO_ANN`, `PACO_DIR`).
- **아직 없는 것 (이전 프로젝트 코드):** `common_rr`이 불러오는 `2026-10-30/paco_full_vs_crop_ft/code`(`common.py`)와 `eval_fu`가 불러오는 `2026-10-30/paco_token_upsample/code`(`eval_tu.py`). 그래서 PACO probe·평가(`src.eval.probe`, `src.eval.paco`, `src.eval.s0_table`)는 아직 실행되지 않습니다. 경로는 `src/config.py`에서 바꿀 수 있습니다.

## 정정 메모

- `experiments/raev2_crop_recon/README.md` §4 복원 표의 **Teacher (hval) 행**은 padding cell이 0인 캐시 latent를 디코딩한 값입니다.
  GT crop을 padding까지 포함해 다시 인코딩한 올바른 값은 L1 .076 / hp_corr .343이고,
  [`results/recon/hval_teacher_fullpad.csv`](experiments/raev2_crop_recon/results/recon/hval_teacher_fullpad.csv)에 있습니다.
  A/B/C, Bilinear 행과 학습 자체는 이 문제의 영향을 받지 않습니다.
- `figures/loss_curves.png`의 Full(Bilinear)과 GT crop oracle 기준선은 학습 곡선과 같은 COCO dev 2,000 ROI에서 계산했습니다
  (`src/eval/refs.py` → `results/dev_refs.json`, teacher는 padding까지 포함해 다시 인코딩).
