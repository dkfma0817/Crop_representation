# P1 — B0 + fixed-reference DINOv3 Gram Anchoring

GPU1로 변경한 사용자 지시에 따라 준비·CPU 검증을 끝낸 후에만 기존 GPU1 작업을 checkpoint pause한다. GPU0의 N1은 중단하거나 변경하지 않는다.

Student는 Original DINOv3-B/16에서 독립적으로12k 학습한다. Native Crop768 및 Full384 Global target은 B0의 cache를 그대로 재사용하고 `src.fgtrain.train.terms`를 그대로 호출한다.
Full384의 최종 LN patch576개만 L2 normalize하여 이미지별 Gram을 계산한다. CLS/register는 제외한다. Gram Teacher는 동일 Full384 tensor를 받는 frozen Original이며 학습 내내 고정한다. Decoder/Adapter/추가 추론 모듈은 없다.

`L_P1 = L_crop + 0.5 L_global + lambda_gram L_gram`
`L_gram(b) = sum_ij wi*wj*(Gs_ij−Gt_ij)^2 / sum_ij wi*wj`; 유효 이미지별 loss를 평균한다. zero-valid 이미지는 평균에서 제외하고 전부 zero-valid이면 graph-connected0을 반환한다. 음수 similarity 및 대각 pair를 유지한다.

정규화·이미지별 Gram MSE의 참고 구현: `third_party/BRIXEL/src/brixel/dinov3_main/dinov3/loss/gram_loss.py`, apply_norm=True/remove_neg=False. Frozen Original을 끝까지 고정하는 reference 선택, fractional letterbox pair weighting 및 zero-valid 처리는 본 실험의 확장이다.

B0의 중간 recovery가 덮어써져 step12000만 남아 있어, 별도 `calibration_B0_replay/`에 동일 B0의 첫6000step을 재현한다. LR와 EMA는 원래12000step schedule 그대로이며,6000 EMA와 사전 고정 stream0 batch로 lambda를 한 번 보정한다. 실제 P1 초기화는 이6k 모델이 아니라 Original이다. λ 탐색은 없다.

lambda_gram=0.131657957079; calibration Gram loss=0.05854324; gradient ratio=0.50000000. Checkpoint=/home/irteam/Crop_representation/output/2026-11-11/fgtrain_p1_s0/calibration_B0_replay/recovery.pt, EMA step6000, image IDs=[261603, 406676, 532689, 394608, 297091, 60602, 327338, 371072, 518966, 224522, 315404, 414851, 226545, 513861, 444533].
Crop/Gram gradient 및 patch embedding·12block norm은 `lambda_gram.json`; 초기/중간/후반 weighted gradient ratio와 cosine은 `P1/gradient_diagnostics.jsonl`에 저장한다. 원본 초기 Student==Teacher의 Gram0, teacher detach/frozen, fractional padding 제외 및 zero-valid를 검증한다.

| Method | Val Overall | Object | Part | Object Pattern | Part Pattern | Obj Color | Part Color | Obj Material | Part Material | ADE mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 54.213 | 52.266 | 56.159 | 51.944 | 52.787 | 42.736 | 50.396 | 52.403 | 57.031 | 49.367 |
| B0 | 55.450 | 54.511 | 56.390 | 59.644 | 55.243 | 45.183 | 52.671 | 51.225 | 56.587 | 44.829 |
| P1 | 55.898 | 54.204 | 57.593 | 59.043 | 59.744 | 45.769 | 51.310 | 52.289 | 55.923 | 45.237 |

P1−B0 (pp): `{"overall": 0.44808709313955575, "object": -0.30767737296380204, "part": 1.2038515592429064, "obj_color": 0.5867701472673446, "obj_pattern": -0.6016704004879543, "obj_material": 1.0638771476921107, "obj_ref": -2.2796863863267234, "part_color": -1.3605169707905063, "part_pattern": 4.501116435819284, "part_material": -0.6643290698149187, "part_ref": 2.3391358417577734, "miou": 0.40790094969555213}`

동일 COCO dev 이미지의 Original 대비 Gram drift: `{"Original": 0.0, "B0": 0.07032157576805542, "P1": 0.025186923428601217}`. 이 진단만으로 ADE 회복을 판단하지 않는다.

Strict standard export=True, params=85641216, patch shape=[2, 576, 768], forward equality=True, extra modules=[].

PACO Test는 자동 실행하지 않는다. N1의 기존 Test Gate와 별개다. ADE는 기존 `src.g1eval.ade`의 frozen encoder + Linear768→150 probe, seed0,40k,batch16,world2,동일 전처리/최종 iteration 평가를 재사용한다. GPU0 사용은 N1이 모두 완료된 뒤 기존 A/B만 안전하게 일시 중지할 수 있을 때로 한정한다.
결과는 Pattern 유지+ADE회복 / Pattern하락+ADE회복 / Pattern유지+ADE회복없음으로 구분한다. 작은 차이의 유지·하락 여부는 단일 seed 점추정만으로 확정하지 않는다. ADE 결과가 없으면 dense preservation은 미검증이다.

## 실제 초기·중간·후반 gradient

| Step | Crop norm | Weighted Gram norm | Gram/Crop | Crop–Gram cosine | Gram drift |
|---|---:|---:|---:|---:|---:|
| 1 | 0.853306 | 0.000000 | 0.000000 | 정의 없음(Gram gradient0) | 0.000000 |
| 6000 | 0.680695 | 0.079664 | 0.117033 | -0.646475 | 0.022566 |
| 12000 | 0.621216 | 0.099540 | 0.160234 | -0.279370 | 0.030301 |

## 최종 해석

ADE20K: P1=45.236717, B0=44.828816, Frozen=49.367149. P1−B0=+0.407901 pp, P1−Frozen=-4.130432 pp.
PACO Official Val: Overall +0.448087 pp, Object Pattern -0.601670 pp, Part Pattern +4.501116 pp. Object/Part Pattern 변화의 방향이 다르므로 두 Pattern의 동시 개선으로 부르지 않는다.
Gram drift 감소는 확인됐지만 ADE 점추정 회복은 작고 Original 수준과의 격차가 남았다. 원본 Dense Representation의 보존 또는 강한 Segmentation 개선을 입증했다고 결론 내리지 않는다. 단일 seed의 작은 mIoU 차이는 학습 변동과 구분하지 못한다.
PACO Part Pattern의 점추정 이득과 제한적인 ADE 회복을 분리해 기록한다. 이 결과로 추가 λ/Teacher/Loss 탐색이나 PACO Test를 자동 실행하지 않는다.

## 실행 중 수정

GPU 인계 중 NVML에 종료 PID가 잠시 남는 경우를 처리했다. 살아 있는 외부 프로세스는 계속 보호한다.12k 학습 및 strict export 후 완료 metadata의 변수 오타로 중단된 실행은 저장된12k EMA와 optimizer checkpoint를 검증해 추가 optimizer step 없이 재개했다. `export_metadata_recovery.json`의 EMA SHA와 `implementation_failures/`에 근거를 남겼다. N1/B0/B1의 가중치·설정·소스는 `protected_inputs.json`으로 전후 일치 검증했다.
