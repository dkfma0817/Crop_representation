# 재현

기존 입력은 읽기 전용으로 사용합니다. 결과 폴더가 이미 있으므로 재현 실행 시 이 폴더의 산출물은 덮어씁니다. 원본 encoder·reader checkpoint, score, 기존 HTML과 코드는 수정하지 않습니다.

Python: `/home/irteam/miniforge3/envs/dinov2_ord_py310/bin/python`.
기존 구현: `/home/irteam/sg/crop-rep`.
공유 cache·output: `/home/irteam/Crop_representation/{cache,output}`.
실행 출발 디렉터리: `/home/irteam/Crop_representation`.

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
PAIR_RUN=/home/irteam/Crop_representation/results/roi_pair_feature_difference_2026-11-09
PAIR_PY=/home/irteam/miniforge3/envs/dinov2_ord_py310/bin/python

"$PAIR_PY" "$PAIR_RUN/source/prepare.py"

# Frozen 추론만 수행합니다. 각 명령은 별도 GPU에서 동시에 실행할 수 있습니다.
CUDA_VISIBLE_DEVICES=0 "$PAIR_PY" "$PAIR_RUN/source/extract.py" --condition C0
CUDA_VISIBLE_DEVICES=1 "$PAIR_PY" "$PAIR_RUN/source/extract.py" --condition P384
CUDA_VISIBLE_DEVICES=2 "$PAIR_PY" "$PAIR_RUN/source/extract.py" --condition P768
CUDA_VISIBLE_DEVICES=3 "$PAIR_PY" "$PAIR_RUN/source/extract.py" --condition P768-centered

CUDA_VISIBLE_DEVICES=0 "$PAIR_PY" "$PAIR_RUN/source/analyze.py"
CUDA_VISIBLE_DEVICES=0 "$PAIR_PY" "$PAIR_RUN/source/difficulty_match.py"
"$PAIR_PY" "$PAIR_RUN/source/report.py"
"$PAIR_PY" "$PAIR_RUN/source/validate.py"
```

`*_first_pass.npy`는 최초 frozen 추론 결과입니다. 완료된 산출물에 함께 보존했으며, `validate.py`가 두 번째 frozen 추론의 feature와 원소별 동일함을 검사합니다. `extract.py`는 이 파일이 없을 때만 자동으로 보존하고, 재추론 때 덮어쓰지 않습니다. 새 환경에서 반복 추론의 동일성까지 검증하려면 네 extract 명령을 한 번 더 실행하십시오. 한 번만 실행해도 Mean/R2 score와 기존 저장 score의 exact replay는 검증됩니다.

Mean/R2 epoch은 기존 train-internal dev M0 최대, 동점이면 이른 epoch으로 선택합니다. Val에서 새 checkpoint나 hyperparameter를 선택하지 않았습니다. Pair는 공통 seed 0으로 고정하고, 각 M2 유효 category–attribute 단위에서 최대 4,096개의 중복 없는 조합을 사용합니다. 어려운 pair의 close matching 및 여러 threshold 비교는 진단 목적의 탐색적 sensitivity입니다.

`prepare.py`가 과거 Test 기반 ranking 코드와 HTML을 복사해 provenance에 보존하지만, 해당 코드를 실행하거나 Test prediction을 로드하지 않습니다. 이번 통계 계산은 Val 배열과 기존 Val score만 사용합니다.
