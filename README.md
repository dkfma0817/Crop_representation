# Crop_representation

작은 ROI의 full-image feature를 GT crop을 다시 인코딩한 것 같은(crop-like) representation으로 바꿀 수 있는지 보는 실험 기록입니다.
평가는 PACO attribute(test-primary, M2)로 합니다.

## 실험 목록

| 폴더 | 날짜 | 내용 | 결론 |
|---|---|---|---|
| [`raev2_crop_recon/`](raev2_crop_recon/) | 2026-11-04 | frozen RAEv2 pixel decoder를 통한 pixel supervision으로 CoordMLP 학습. A = feature loss만, B = pixel loss만, C = feature + pixel | 복원 지표는 좋아졌지만 PACO 이득은 없음. C−A는 R2 +0.62로 seed 편차(0.49) 수준이고 R0는 +0.11. GT crop 상한과의 격차(Obj 5–16 약 −20)는 그대로 |

자세한 내용은 각 폴더의 `README.md`(보고서)와 `configs/protocol.md`(사전 등록 프로토콜)를 보세요.

## 실험 환경

[`env/`](env/)에 패키지 목록(`requirements.txt`)과 설치 스크립트(`setup_env.sh`)가 있습니다. 새 서버에서는 `bash env/setup_env.sh <env 경로>`로 설치하세요.

## `raev2_crop_recon/` 구성

| 경로 | 내용 |
|---|---|
| `README.md` | 실험 보고서 (결론, sanity check, 복원·PACO 표, 해석) |
| `configs/protocol.md` | 학습 전에 고정한 프로토콜 |
| `code/` | 학습·평가·시각화 코드 |
| `results/` | 표(csv/md/json), ROI별 복원 지표(`recon/`), 시각화(`viz/`) |
| `figures/` | loss 곡선(`loss_curves.png`), 복원 비교, 오차 맵 |

**포함하지 않은 것**
- `perception_cache/` (약 78 GB): feature 캐시, probe 결과, checkpoint, RAEv2 가중치. 모두 다시 만들거나 공식 배포처에서 받을 수 있습니다.
- RAEv2 코드와 가중치: 라이선스가 CC BY-NC 4.0이라 함께 넣지 않았습니다. 아래 출처에서 받으세요.

## 외부 의존과 재현 시 주의

- **RAEv2:** 코드는 [nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2) @ `8a0d238`, 가중치는 HF `nyu-visionx/RAEv2-models` @ `9770b7b` (`stage1/imagenet/dinov2b-k1`)입니다.
  코드는 이 repo를 `raev2_crop_recon/third_party/RAEv2`에 clone해 둔 것을 전제로 합니다.
- **같은 연구 폴더의 다른 실험 코드와 데이터:** 코드가 아래 경로를 절대경로로 참조하고, 이 repo에는 들어 있지 않습니다.
  - `2026-10-26/rae_crop_pilot`: `rcp.py`, lpips 0.1.4
  - `2026-10-24/dinov2_oracle_region_decoder`: `coco_rois.parquet`, PACO 전처리 데이터
  - `2026-10-30/paco_region_reader`, `2026-10-31/paco_feat_upsample`: PACO probe(R0/R2 reader) 코드
- **데이터셋:** COCO train2017, PACO(LVIS 기반), ImageNet val(pair 검증용).
- **환경:** [`env/`](env/) 참고 (Python 3.10.19, torch 2.7.1+cu118).

따라서 이 repo만으로는 바로 실행되지 않습니다. 실험 기록과 코드 열람용입니다.

## 정정 메모

- `raev2_crop_recon/README.md` §4 복원 표의 **Teacher (hval) 행**은 padding cell이 0인 캐시 latent를 디코딩한 값입니다.
  GT crop을 padding까지 포함해 다시 인코딩한 올바른 값은 L1 .076 / hp_corr .343이고,
  [`results/recon/hval_teacher_fullpad.csv`](raev2_crop_recon/results/recon/hval_teacher_fullpad.csv)에 있습니다.
  A/B/C, Bilinear 행과 학습 자체는 이 문제의 영향을 받지 않습니다.
- `figures/loss_curves.png`의 Full(Bilinear)과 GT crop oracle 기준선은 학습 곡선과 같은 COCO dev 2,000 ROI에서 계산했습니다
  (`code/eval_refs.py` → `results/dev_refs.json`, teacher는 padding까지 포함해 다시 인코딩).
