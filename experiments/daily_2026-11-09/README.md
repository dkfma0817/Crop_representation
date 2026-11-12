# 2026-11-09 실험 HTML 보고서

`index.html`을 브라우저에서 여십시오. 본문·스타일·JavaScript·결과 데이터·두 핵심 그래프가 한 파일에 내장되어 인터넷 연결 없이 읽을 수 있습니다. 원자료 링크와 별도 CSV 링크는 작업 폴더 기준입니다.

- `index.html`: 사용자용 HTML. Backbone·질문 축·완료 상태·검색 필터, 프로토콜별 대표 성능표, 실험별 구조·학습 변경, 결과, 근거와 제한.
- `data.json`: 집계 시점의 수치와 원자료 경로·수정 시각·SHA256.
- `experiments.csv`: 17개 묶음의 질문·대조군·변경·초기화·예산·상태.
- `dinov3_audit.json`, `siglip_audit.json`: 기존 config·결과·코드에서 확인한 감사 기록. 전날 학습과 오늘 평가, 독립 실행, 미완료 후처리를 구분.
- `assets/`: 재생성용 두 PNG와 보고서 글자만 포함한 한글 WOFF. 배포 HTML에는 base64로 내장.
- `validation.json`: 구조·링크·데이터·프로토콜 sanity check.
- `build.py`: 캡처된 `data.json`과 `assets/`로 HTML을 재생성.
- `collect.py`: 원래 프로젝트 자료를 읽어 스냅샷 구성. 감사 JSON은 고정 기록이며 새 날짜의 전체 실험 감사를 자동 수행하는 스크립트는 아님.

기존 checkpoint·코드·결과·diagnosis HTML은 수정하지 않았고, 보고서 작성을 위해 새 모델 학습·평가를 실행하지 않았습니다. 진행 중인 원자료는 보고서 집계 이후 바뀔 수 있습니다.

같은 스냅샷의 HTML 재생성:

```bash
/home/irteam/miniforge3/envs/dinov2_ord_py310/bin/python experiments/daily_2026-11-09/build.py
node --check experiments/daily_2026-11-09/report.js
```

`공식 Test AP`, `공식 Val AP`, `project M2 Val`은 선택 프로토콜을 분리했습니다. 공식 Test 대표 표는 CI가 없는 점추정이며, 다른 Val M2 분석의 CI를 가져와 붙이지 않았습니다. 각 카드에 CI를 쓴 경우 해당 원자료의 비교와 split을 그대로 표시했습니다.

폰트는 Noto Sans CJK KR Regular에서 보고서에 필요한 문자만 subset한 `Crop Report Sans`입니다. 원본 출처는 `https://github.com/notofonts/noto-cjk`이며 SIL Open Font License 1.1을 따릅니다. 전체 라이선스는 `assets/FONT_LICENSE.txt`와 HTML 내부의 `font-license` 블록에 포함했습니다. 시스템 폰트를 설치하거나 변경하지 않았습니다.
