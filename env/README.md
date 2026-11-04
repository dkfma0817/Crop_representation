# 실험 환경

모든 실험은 같은 Python env 하나로 실행했습니다.

- Python 3.10.19
- torch 2.7.1+cu118, torchvision 0.22.1+cu118
- timm 1.0.27, transformers 4.52.1
- numpy 2.2.6, pandas 2.3.3, Pillow 12.0.0

`requirements.txt`는 원래 서버 env의 `pip freeze` 결과(58개)입니다. 여기에 원래 env 밖에서 불러오던 `lpips 0.1.4`와 `omegaconf 2.3.0`(+ `antlr4-python3-runtime`)을 추가했습니다.

## 설치

```bash
git clone https://github.com/dkfma0817/Crop_representation.git   # 이미 받았다면: cd Crop_representation && git pull
bash Crop_representation/env/setup_env.sh /workspace/Areum/research/envs/dinov2_ord_py310
```

- 경로는 **컨테이너가 바뀌어도 남는 디스크**로 지정하세요. 기존 코드는 위 경로를 그대로 쓰므로, 같은 경로로 만들면 코드를 고칠 필요가 없습니다.
- conda가 없으면 스크립트가 miniforge를 `~/miniforge3`에 설치합니다.
- 정상 설치면 마지막에 `torch 2.7.1+cu118 | cuda True | cudnn 90100 | timm 1.0.27 | transformers 4.52.1`이 출력됩니다.
- cu118 wheel은 CUDA 런타임을 포함하므로 CUDA toolkit은 필요 없습니다. NVIDIA driver 520 이상이면 됩니다.

## 모델 가중치

코드는 `HF_HUB_OFFLINE=1`로 실행합니다. 새 서버에서는 **처음 한 번 이 변수 없이 실행해서** 가중치를 받아야 합니다.

- timm `vit_base_patch14_dinov2.lvd142m`
- `google/siglip2-base-patch16-384`
- torch.hub `facebookresearch/dinov2` `dinov2_vitb14_reg`
- RAEv2 decoder: HF `nyu-visionx/RAEv2-models` @ `9770b7b`, `stage1/imagenet/dinov2b-k1`
- LPIPS용 VGG16 (torchvision)
