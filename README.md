# Crop_representation

작은 ROI의 full-image feature를 GT crop을 다시 인코딩한 것 같은(crop-like) representation으로 바꿀 수 있는지 보는 실험 기록입니다.
평가는 PACO attribute로 합니다. 실험마다 split(val / test-primary)과 지표(M0 / M2)가 다르므로 결론 열에 함께 적었습니다.

## 실험 목록

| 폴더 | 날짜 | 내용 | 결론 |
|---|---|---|---|
| [`experiments/raev2_crop_recon/`](experiments/raev2_crop_recon/) | 2026-11-04 | frozen RAEv2 pixel decoder를 통한 pixel supervision으로 CoordMLP 학습. A = feature loss만, B = pixel loss만, C = feature + pixel | 복원 지표는 좋아졌지만 PACO 이득은 없음. C−A는 R2 +0.62로 seed 편차(0.49) 수준이고 R0는 +0.11. GT crop 상한과의 격차(Obj 5–16 약 −20)는 그대로 |
| [`experiments/readout_queue/`](experiments/readout_queue/) | 2026-11-04 | frozen DINOv2 공유 feature 위 reader 변형: E08 반복 읽기, E09 query 적응 샘플링, E11 local/context gate, E12 scale 일관성 (+ D0 binding 진단) | 네 reader 모두 Multi-layer + R2(val M2 46.23)보다 유의한 개선 없음 (PACO val) |
| [`experiments/msd/`](experiments/msd/) | 2026-11-05 | COCO에서 multi-depth(L4/L8/L12) crop distillation decoder 사전학습(H0–H3) → frozen PACO R2 probe | multi-depth 감독은 유의한 개선(H1−H0 +1.36, H3−H2 +1.01). 그래도 모두 Full Multi-layer + R2보다 1.3–2.7 낮음 (PACO val M0) |
| [`experiments/sdec/`](experiments/sdec/) | 2026-11-05 | 공간 ROI decoder 구조 비교: A = 계층적(coarse-to-fine) vs flat, B = two-way vs one-way | 둘 다 PACO 개선 없음(A1−A0 +0.32, B1−B0 +0.34, CI가 0 포함). Full Multi-layer + R2보다 1.5–1.7 낮음 (PACO val M0) |
| [`experiments/ivc/`](experiments/ivc/) | 2026-11-05 | 학습 없이 Bilinear / Bicubic ROI feature와 C0 crop feature의 차이, 그리고 성능 차이와의 연결 분석 (+ test-primary 1회) | C0−Bicubic R2 +2.73(유의), Bicubic−Bilinear는 차이 없음. 평균 표현과 ROI 내부 공간 구조 모두 크게 다르지만 ROI 단위 연결은 약함 (PACO val M0) |
| [`experiments/xdec/`](experiments/xdec/) | 2026-11-05 | frozen F336에서 C0 crop feature를 만드는 decoder 구조 7종(E1–E7) + R00 비교 | 어느 decoder도 Bicubic(42.71)을 넘지 못함(최고 E6 42.70). feature 복원은 좋아졌지만(token cos 0.35 → 0.57) PACO 순서는 오히려 반대 (PACO val M0) |
| [`experiments/svamp/`](experiments/svamp/) | 2026-11-05 | 학습 없이 Bicubic feature의 공간 분산 증폭(V1–V3) vs C0 공간 패턴 oracle(O2) | 분산만 키우면 효과 없음(−0.08 ~ −0.35). 같은 분산에 C0 패턴을 넣으면 +3.41로 C0 Full 수준 → 필요한 것은 분산 크기가 아니라 패턴 방향 (PACO val M0) |
| [`experiments/raediag/`](experiments/raediag/) | 2026-11-05 | 학습 없이 RAEv2 복원 진단(GT crop / full 복원 후 crop / 보간 ROI latent decode)과 R00 구조 확인 | GT crop latent는 잘 복원되지만 보간 ROI latent는 full 복원보다도 나쁨. ROI 안이 평평한 latent가 원인이라는 설명과 가장 일관됨 |
| [`experiments/densecmp/`](experiments/densecmp/) | 2026-11-06 | dense CLIP 계열 region representation 비교: EVA-CLIP / CLIPSelf / DeCLIP (입력 1024) + EVA-CLIP GT crop oracle | CLIPSelf가 EVA-CLIP보다 +13.39, GT crop oracle과 차이 없음(−0.05). DeCLIP은 +10.87 (PACO test-primary M2) |
| [`experiments/e6dir/`](experiments/e6dir/) | 2026-11-06 | xdec E6에 centered direction loss 추가 | 위치별 패턴 cos는 0.387 → 0.404로 조금 개선. PACO는 개선 없음(−0.23, Bicubic 미만) (PACO val M0) |
| [`experiments/e6ml/`](experiments/e6ml/) | 2026-11-06 | E6 decoder의 K/V memory에 중간층 L4/L8 추가 | 매칭 대조군 대비 +0.72(유의)지만 E6-original 대비 +0.13(차이 없음). Full Multi-layer + R2보다 −2.18 (PACO val M0) |

보고서는 각 실험 폴더의 `REPORT.md`(raev2_crop_recon은 `README.md`), 사전 등록 프로토콜은 `configs/<실험>/protocol.md`에 있습니다.
ROI별 원자료(csv / parquet), checkpoint, probe 결과 같은 대용량 파일은 git에서 제외된 서버의 `output/<날짜>/<실행>/`에 남아 있습니다.

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
│   ├── utils/                  # rcp.py (rae_crop_pilot RGB 지표), yaml_config.py (YAML 로더, _base_ 상속)
│   ├── readout/                # readout_queue (E08–E12 reader, D0 binding)
│   ├── msd/                    # msd (H0–H3) + sdec (sdec.py, sdirect.py, sreport.py)
│   └── ivc/ xdec/ svamp/ raediag/ densecmp/ e6dir/ e6ml/   # 실험별 패키지 (폴더 이름 = 실험 이름)
├── configs/
│   ├── paths.yaml              # 모든 경로 (데이터셋, cache/, third_party/, 이전 프로젝트 데이터·코드)
│   ├── raev2_crop_recon/
│   │   ├── protocol.md         # 사전 등록 프로토콜
│   │   └── train/              # base.yaml (공통 레시피) + A / B / C / C_L1 / overfit_{feat,pix}.yaml
│   └── <실험>/                 # protocol.md + probe_*.yaml (일부는 architecture_audit.md, sample_rois.json)
├── scripts/raev2_crop_recon/   # 실행 스크립트 (run_extract_local.sh = 이 서버용 캐시 추출, run_main.sh RUN GPU = 학습 + probe)
├── scripts/<실험>/             # queue / run_all, QAT pause·resume 스크립트 (실행 당시 output/ 폴더에 있던 것을 그대로 복사)
├── experiments/raev2_crop_recon/  # 보고서 README.md, results/, figures/
├── experiments/<실험>/         # 보고서 REPORT.md, tables.md, 요약 csv/json, figures/
├── env/                        # requirements.txt, setup_env.sh
├── cache/                      # (git 제외) 모델 가중치, ROI 목록, feature 캐시, runs, probe
├── third_party/RAEv2/          # (git 제외) 공식 RAEv2 코드
├── output/                     # (git 제외) 실행 결과 output/<날짜>/<run>/ (run.py)
├── run.py                      # 런처: output 폴더 생성, 설정 저장, 로그, --gpus N 이면 torchrun
└── logs/                       # (git 제외) 기타 실행 로그와 flag
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
#    run.py 가 output/<날짜>/<run>/ 을 만들고 config.yaml, cmd.txt, train.log 를 남긴다
CUDA_VISIBLE_DEVICES=2 python run.py --config configs/raev2_crop_recon/train/C.yaml --bg                 # GPU 1개
CUDA_VISIBLE_DEVICES=2,3 python run.py --config configs/raev2_crop_recon/train/B.yaml --gpus 2 --bg      # GPU 2개 (약 2배 빠름)
CUDA_VISIBLE_DEVICES=3 python run.py --config configs/raev2_crop_recon/train/A.yaml --steps 30 --output_dir output/test/A   # 일부만 바꾸기

# 3) 복원 평가 (hval + PACO test-primary, teacher / bilinear 포함)
CUDA_VISIBLE_DEVICES=2 python -m src.eval.recon --root output/2026-11-04 --runs B --steps 4000,12000 --out_dir output/2026-11-04/B/recon
```

- **결과 폴더:** `output/<YYYY-MM-DD>/<run>/`에 `config.yaml`, `cmd.txt`, `train.log`, `train_log.jsonl`(50 step마다 + dev 평가), `ckpt_<step>.pt`, `resume.pt`, `done.json`이 생깁니다. git에는 넣지 않습니다.
- **GPU 2개 이상:** 한 step의 ROI 64개(micro-batch 16 × 4)를 GPU끼리 나눠 계산하고 gradient를 합칩니다. 업데이트는 GPU 1개와 같습니다 (50 step 비교에서 train loss 소수점 6자리까지 일치). 처음 통신 초기화에 2~3분 걸립니다.

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
