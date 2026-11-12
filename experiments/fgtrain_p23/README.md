# P2 / P3 — Uniform vs Region-Selective Patch Preservation

P2와 P3는 Original pretrained `vit_base_patch16_dinov3.lvd1689m`에서 각각 독립적으로 시작한다. Student는 RGB384 letterbox → 기존12block → 최종 LN576×768이다. Native RGB Crop768 target 및 Full384 global target은 기존 B0 cache/ROI stream 그대로다. Patch preservation teacher는 동일 Full384 tensor를 받는 frozen/eval Original이며 학습 내내 업데이트하지 않는다.

`L = L_crop + 0.5 L_global + lambda_preserve L_patch`

Crop/Global은 기존 `src.fgtrain.train.terms`를 수정 없이 호출한다. `d_p=1−cos(normalize(Fs_p),normalize(F_original_p))`는 FP32로 계산하고 Teacher를 detach한다. CLS/register는 제외한다. Fractional letterbox coverage `v_p`에 대해 이미지별 `sum(v*w*d)/sum(v)`를 계산하고 유효 이미지별로 평균한다. Zero-valid 이미지는 제외하며 전체가 zero-valid면 Student graph에 연결된0을 반환한다.

P2는 `w=1`. P3는 실제 Crop loss의 GT+Random+fallback ROI **기하학적 Union**에 대한 유효 patch 면적 비율 `m`을 구하고, `raw=1−0.75*m`, `mean_raw=sum(v*raw)/sum(v)`, `w=raw/mean_raw`를 적용한다. 정규화 후에도 분모는 `sum(v)`다. Float64 canvas rectangle inclusion/exclusion은 중복·겹침을 한 번만 세며 독립 sweep-line 참조와 대조했다.

추론에는 표준 DINOv3 encoder만 남는다. Teacher/ROI mask/loss는 encoder module tree에 추가되지 않는다. RGB는 이미지당 한 번만 인코딩하고 ROI RGB 재인코딩은 없다. P2는 물리GPU0, P3는 물리GPU1, 각각 내부cuda:0으로 학습한다. 기존 A/B는 안전한 checkpoint pause 후 배정하고 Reader/CPU 구간에는 backfill한다.

B0와 같은 Seed0,12k,AdamW,LR5e-5,LLRD0.75,WD0.05,betas0.9/0.999,600warmup+cosine,clip1,동일cosineEMA를 사용한다. 기본16image×4ROI stream에서 같은 제외 목록을 적용하므로 실제 유효 batch 크기도 B0와 같다. Final12k EMA만 평가하며 calibration6k EMA로 초기화하지 않는다.

## 검증 및 사전 고정

`training_protocol.json`, `calibration_source_audit.json`, `source_manifest.json`, `data_manifest.json`, `protected_inputs.json`에 설정·소스·체크포인트·데이터 해시를 기록했다. B0에 별도 training_protocol.json은 없어서 실제 B0/config.json과 초기화/제외/cache 기록을 검증한다. P1 replay는 B0의6000step까지 동일12000 LR/EMA schedule로 실행된 EMA이며 BF16 replay의 미세한 수치 변동은 기존 calibration_replay_match.json에 남아 있다.

P2에서 한 번 보정한 공통 lambda_preserve=0.065171371165. Crop norm=0.326314390, unweighted patch norm=2.503510237, 가중 ratio=0.500000. P3 별도 보정과 λ 탐색은 하지 않는다.
Calibration EMA SHA: `34ed4cfdd7a223b1c874029508e6bd862a12fe344441d5cab58e84dbe8c85238`. 동일 stream0 이미지 IDs: `[261603, 406676, 532689, 394608, 297091, 60602, 327338, 371072, 518966, 224522, 315404, 414851, 226545, 513861, 444533]`.

ROI Union 사전 점검 527images, coverage quantiles={'0': 0.005775401368737221, '0.25': 0.11373036354780197, '0.5': 0.2663639187812805, '0.75': 0.5199605524539948, '0.95': 0.9322500586509703, '1': 1.0}; P2/P3 평균 가중치 절대차<0.01 비율=0.005693. Union이 거의 전체이면 P3가 P2로 수렴하는 한계가 있다.0.75 계수는 변경하지 않는다.
Padding/inside/outside/boundary/정규화 가중치 및 차이는 `roi_union_preflight.json`; 시각화는 `preflight/union_*.png`.

## PACO Official Val + ADE20K

PACO: Official Val2410, GT Box+GT Category Oracle attribute AP, 기존 bilinear ROIAlign24, R2 seed0/init7,10epochs. Object/Part AP 평균 최대 epoch를 선택하고 동률은 이른 epoch를 사용한다. B0/Frozen/B1/P1도 같은 프로토콜의 완료 결과만 재사용한다. Test는 실행하거나 읽지 않는다.

ADE20K: train20210/Official Val2000,150classes, frozen final-LN encoder + Linear768→150, seed0,40k,world2/globalbatch16. 기존 `src.g1eval.ade`를 수정하지 않고 `src.fgtrain.ade`의 encoder loader만 사용한다. P2/P3 각각 새로운 동일 초기화 head를 동일 augmented batch로 학습한다. 최종iteration만 평가하고 Val head selection은 없다.

ADE 학습 전처리는 기존 short-side384×U(0.5,2),384crop,pad0,label255,hflip0.5,ImageNet normalization이다. 평가는 기존 short-side384 가변 aspect grid,16배수 padding,유효 logits crop 후 원래 GT 좌표로 resize한다. Class0은 ignore255,1..150→0..149. 동일 프로토콜 유지를 위해 평가는 square letterbox로 변경하지 않는다.

| Method | Overall AP | Object AP | Part AP | Object Pattern | Part Pattern | Obj Color | Part Color | Obj Material | Part Material | ADE mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 54.212756 | 52.266481 | 56.159032 | 51.943767 | 52.787385 | 42.736270 | 50.395994 | 52.403051 | 57.030898 | 49.367149 |
| B0 | 55.450408 | 54.511306 | 56.389511 | 59.644402 | 55.242753 | 45.182727 | 52.670993 | 51.225419 | 56.587176 | 44.828816 |
| B1 | 55.778975 | 54.327146 | 57.230804 | 58.186199 | 56.376012 | 46.524781 | 53.480163 | 50.875264 | 56.483110 | 45.453114 |
| P1 | 55.898495 | 54.203629 | 57.593362 | 59.042732 | 59.743869 | 45.769497 | 51.310476 | 52.289297 | 55.922847 | 45.236717 |
| P2 | 55.456957 | 53.716002 | 57.197911 | 57.158315 | 59.130266 | 46.231284 | 49.930879 | 52.175615 | 57.198496 | 46.907381 |
| P3 | 55.194704 | 53.022607 | 57.366801 | 56.230059 | 57.193004 | 45.458010 | 52.695472 | 51.242025 | 56.632916 | 46.873151 |

## P2/P3 비교 및 진단

P2−Frozen (pp): `{"overall": 1.2442002123573985, "object": 1.449521188061702, "part": 1.038879236653095, "obj_pattern": 5.214548622443822, "part_pattern": 6.342881580591509, "obj_color": 3.4950144628372755, "part_color": -0.46511480672856464, "obj_material": -0.22743611720085966, "part_material": 0.16759737512929007, "miou": -2.4597680459033384}`

P2−B0 (pp): `{"overall": 0.0065483674576682915, "object": -0.7953037401833498, "part": 0.8084004750986793, "obj_pattern": -2.4860868374463365, "part_pattern": 3.887513400736559, "obj_color": 1.0485574293043385, "part_color": -2.7401144228549867, "obj_material": 0.9501956478473943, "part_material": 0.6113195713007755, "miou": 2.0785648634088787}`

P2−P1 (pp): `{"overall": -0.44153872568188746, "object": -0.4876263672195478, "part": -0.39545108414422714, "obj_pattern": -1.8844164369583822, "part_pattern": -0.6136030350827255, "obj_color": 0.46178728203699393, "part_color": -1.3795974520644805, "obj_material": -0.11368149984471643, "part_material": 1.2756486411156942, "miou": 1.6706639137133266}`

### P2 gradient

| Step | Crop norm | Weighted preservation | Ratio | Gradient cosine | Patch drift | ROI inside drift | ROI outside drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.85281891 | 2.0294161e-10 | 2.3796565e-10 | 0.28616231 | 9.7031512e-09 | 4.0361328e-09 | 9.3350891e-09 |
| 6000 | 0.58015513 | 0.093748671 | 0.16159242 | -0.77203203 | 0.14120878 | 0.13287559 | 0.14985025 |
| 12000 | 0.51515871 | 0.11953563 | 0.23203651 | -0.44659043 | 0.17273253 | 0.18271104 | 0.17821413 |

Patch embedding과 12개 Block의 raw/total norm, Loss, 이미지 ID는 각 `gradient_diagnostics.jsonl`에 기록한다. 초기 Student==Teacher이면 Preservation Gradient가 거의0이므로 이 시점의 Gradient Cosine은 해석하지 않는다. `train_log.jsonl`에 Loss50step 평균과 Global Similarity를 저장한다.

P2: strict=True, params=85641216, shape=[2, 576, 768], forward equality=True, same module tree=True, extra modules=[]. EMA SHA `ba178de597d78f9fc3a5a609f8cdf31d1b1055b6710fc6624b0ed2c59b121b26`.

P3−Frozen (pp): `{"overall": 0.9819474643413599, "object": 0.7561257514303534, "part": 1.2077691772523664, "obj_pattern": 4.286292653838849, "part_pattern": 4.405619599232573, "obj_color": 2.721740067018615, "part_color": 2.299478311325089, "obj_material": -1.1610263928933904, "part_material": -0.39798227214480164, "miou": -2.493997628249673}`

P3−B0 (pp): `{"overall": -0.2557043805583703, "object": -1.4886991768146984, "part": 0.9772904156979507, "obj_pattern": -3.414342806051309, "part_pattern": 1.9502514193776221, "obj_color": 0.2752830334856782, "part_color": 0.024478695198666856, "obj_material": 0.016605372154863574, "part_material": 0.04573992402668381, "miou": 2.044335281062544}`

P3−P1 (pp): `{"overall": -0.703791473697926, "object": -1.1810218038508964, "part": -0.2265611435449557, "obj_pattern": -2.8126724055633545, "part_pattern": -2.550865016441662, "obj_color": -0.3114871137816664, "part_color": 1.3849956659891731, "obj_material": -1.0472717755372472, "part_material": 0.7100689938416025, "miou": 1.636434331366992}`

### P3 gradient

| Step | Crop norm | Weighted preservation | Ratio | Gradient cosine | Patch drift | ROI inside drift | ROI outside drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.85303414 | 2.2703982e-10 | 2.6615561e-10 | 0.34522396 | 9.7031512e-09 | 4.0361328e-09 | 9.3350891e-09 |
| 6000 | 0.59535766 | 0.08615295 | 0.14470789 | -0.75082012 | 0.13453977 | 0.1346011 | 0.1385283 |
| 12000 | 0.5244174 | 0.11150356 | 0.21262369 | -0.45959204 | 0.16368569 | 0.18042232 | 0.16058651 |

Patch embedding과 12개 Block의 raw/total norm, Loss, 이미지 ID는 각 `gradient_diagnostics.jsonl`에 기록한다. 초기 Student==Teacher이면 Preservation Gradient가 거의0이므로 이 시점의 Gradient Cosine은 해석하지 않는다. `train_log.jsonl`에 Loss50step 평균과 Global Similarity를 저장한다.

P3: strict=True, params=85641216, shape=[2, 576, 768], forward equality=True, same module tree=True, extra modules=[]. EMA SHA `5ae5164f3d85c85a72de5eb47435c541c08299016d8887da2209a7fca6a64a87`.

P3−P2 (pp): `{"overall": -0.2622527480160386, "object": -0.6933954366313486, "part": 0.16888994059927143, "obj_pattern": -0.9282559686049723, "part_pattern": -1.9372619813589367, "obj_color": -0.7732743958186603, "part_color": 2.7645931180536536, "obj_material": -0.9335902756925307, "part_material": -0.5655796472740917, "miou": -0.03422958234633455}`

동일한 excluded COCO dev126 이미지와 Full384 Tensor에서 Original/B0/P1/P2/P3를 비교한다. ROI 내부/외부는 정확한 Union의 fractional m/1−m으로 면적 가중하고, 면적0인 그룹은 None으로 기록한다. 이미지별 분포는 `patch_drift.json`에 저장한다.

Measured drift means: `{"Original": {"patch_cosine": 0.9999999920210273, "gram_drift": 0.0, "patch_distance": 7.978972816012246e-09, "roi_inside_distance": 7.972532383829972e-09, "roi_outside_distance": 8.239772065272216e-09, "roi_union_fraction": 0.3551447832927344}, "B0": {"patch_cosine": 0.7479339549465785, "gram_drift": 0.07032157576805542, "patch_distance": 0.2520660450534215, "roi_inside_distance": 0.21675235582959085, "roi_outside_distance": 0.2892337345651218, "roi_union_fraction": 0.3551447832927344}, "P1": {"patch_cosine": 0.7874474843579625, "gram_drift": 0.025186923428601217, "patch_distance": 0.21255251564203748, "roi_inside_distance": 0.20290891968068622, "roi_outside_distance": 0.2299612203524226, "roi_union_fraction": 0.3551447832927344}, "P2": {"patch_cosine": 0.8528936071883119, "gram_drift": 0.037941807215767245, "patch_distance": 0.14710639281168816, "roi_inside_distance": 0.15604533275796306, "roi_outside_distance": 0.15537464234327514, "roi_union_fraction": 0.3551447832927344}, "P3": {"patch_cosine": 0.8593945606123834, "gram_drift": 0.03547681888772382, "patch_distance": 0.14060543938761666, "roi_inside_distance": 0.15659661091391056, "roi_outside_distance": 0.14257672890311196, "roi_union_fraction": 0.3551447832927344}}`

## 결과 해석

P2−P1 ADE=+1.670664 pp.P3−P2 ADE=-0.034230 pp, Object Pattern=-0.928256 pp, Part Pattern=-1.937262 pp.

ADE와 Object/Part Pattern의 변화 방향을 개별적으로 평가한다. P3가 P2의 두 성능을 동시에 개선했다고 일괄 주장할 수 없다. 단일 Seed의 작은 차이에 통계적 의미를 부여하지 않는다.

Feature/Gram Drift 감소만으로 ADE 보존의 성공을 판단하지 않는다. PACO와 ADE 실측이 모두 확보될 때까지 Method의 성공은 미확정이다. PACO Test, P4, 추가 Loss, λ/Mask Sweep, 새로운 Architecture를 자동 실행하지 않는다.

## 구현 중 수정

최초 인계 시 Python3.10의 os.pidfd_open 미지원으로 실행이 중단됐다. 기존 프로젝트의 libc pidfd 호환 함수를 재사용하도록 수정했다. 이 시도에서는 Calibration/학습 업데이트가 없었으며 예약 해제 후 A/B가 재개됐다. 수정 전 Source Manifest와 로그는 implementation_failures에 보존했다. Loss, λ 설계, 데이터 Stream은 변경하지 않았다.
