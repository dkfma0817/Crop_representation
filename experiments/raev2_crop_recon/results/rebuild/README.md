# 이 서버에서 다시 만든 stream·캐시 기록 (2026-11-04)

- `*.json` (이 폴더): **원본 ROI 목록**(이전 서버에서 복사한 `coco_rois.parquet`, `rae_rois.parquet`) 기준.
  stream pool 55,861장 → 50,491장 / 201,964 ROI (protocol과 같음), 캐시 검증 오차는 fp16 반올림 수준.
  재현 확인: A 설정 step-0 COCO dev feature loss 1.239 (원래 보고서 값과 같음).
- `rebuilt_rois/`: 원본을 받기 전에 `src/data/make_rois.py`로 만든 비슷한 ROI 목록 기준 기록. 지금 캐시는 이것이 아님.
