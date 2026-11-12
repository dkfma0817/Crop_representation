# PACO GT ROI Attribute 추론·평가 가이드

`index.html`을 브라우저로 여십시오. 이미지·예측 데이터·SVG·JavaScript·한글 폰트를 내장한 단일 HTML로, 인터넷 없이 사용할 수 있습니다.

- 실제 PACO Val 4개 이미지 / 6개 ROI의 GT box, Object/Part, 59개 label 및 저장된 C0·P768 예측 탐색
- Full-image feature의 ROI 읽기와 C0 RGB crop 재인코딩의 차이
- Mean + Linear(R0), attention reader(R2), 학습 단계와 inference 구성
- 설명용 AP simulator: score 순서, unknown 제외, threshold와 AP의 차이
- Project M0/M2와 공식 bbox attribute evaluator의 차이, 계층 집계
- 저장된 C0 ep6 / P768 ep5의 공식 Val·Test 결과와 선택 규칙

새 모델 학습·probe 학습·GPU 추론·공식 AP 재평가를 실행하지 않았습니다. 기존 코드·checkpoint·결과 파일도 수정하지 않았습니다. `collect.py`는 기존 예측과 annotation을 CPU로 읽고 검증합니다.

파일:

- `index.html`: 단독으로 열 수 있는 완성본
- `report.html`, `report.js`, `build.py`: 재생성용 템플릿·동작·빌더
- `data.json`: 실제 예시와 평가 결과 스냅샷, 내장 원본 이미지
- `example_predictions.csv`: 6 ROI × 59 attribute = 354행의 label/logit/probability
- `provenance.json`: 읽은 코드·데이터·결과·이미지의 SHA256
- `validation.json`: ID·label·sigmoid·링크·원자료·기존 AP kernel 검증
- `qa/browser_qa.json`: 브라우저에서 실제로 조작한 검증 결과
- `qa/*.png`: desktop/mobile 및 주요 시각화 화면
- `assets/GuideKorean.woff`, `assets/FONT_LICENSE.txt`: Nanum Gothic subset과 라이선스

재생성:

```bash
/home/irteam/envs/dinov2_ord_py310/bin/python experiments/paco_gt_roi_guide_2026-11-11/collect.py
/home/irteam/envs/dinov2_ord_py310/bin/python experiments/paco_gt_roi_guide_2026-11-11/build.py
node --check experiments/paco_gt_roi_guide_2026-11-11/report.js
/home/irteam/envs/dinov2_ord_py310/bin/python experiments/paco_gt_roi_guide_2026-11-11/validate.py
```

설명용 AP 예시는 bbox 매칭·IoU·ignore·max_dets를 포함하는 전체 공식 evaluator가 아닙니다. 실제 예측 6개도 대표 성능을 추정하기 위한 통계 표본이 아닙니다. C0 ep6와 기존 다른 보고서의 C0 ep5는 reader가 다르므로 값이 다를 수 있습니다. 비교할 때는 split과 reader epoch·선택 규칙을 확인하십시오.
