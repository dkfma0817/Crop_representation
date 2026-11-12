import hashlib
import json
import os
from pathlib import Path
import torch

MODEL = 'vit_base_patch16_dinov3.lvd1689m'
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/2026-11-10/fgtrain_s0'
OLD768 = ROOT / 'output/2026-11-10/overnight_six_teacher768'
OLDGLOBAL = ROOT / 'output/2026-11-08/hcf_pilot/shared'
EXCLUDE = 'output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encoder(path, dev='cuda', train=False):
    import timm
    model = timm.create_model(MODEL, pretrained=False, img_size=384, dynamic_img_size=True)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if 'enc' in state:
        state = state['enc']
    model.load_state_dict(state, strict=True)
    assert len(model.blocks) == 12 and model.num_features == 768
    return model.to(dev).train(train).requires_grad_(train)


def initialise(run=RUN):
    import timm
    run = Path(run)
    path = run / 'initial_encoder.pt'
    if not path.exists():
        model = timm.create_model(MODEL, pretrained=True, img_size=384, dynamic_img_size=True).eval()
        torch.save(model.state_dict(), path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    result = dict(model=MODEL, initialisation='Original pretrained DINOv3; neither G1-pool nor A0',
                  state_sha256=state_sha(state), parameters=sum(v.numel() for k, v in state.items()), path=str(path))
    write_json(run / 'initialisation.json', result)
    return result


def freeze_export(model, out, sample, reference=None, step=0):
    """Pure standard state dict plus evaluation envelope containing ONLY encoder weights."""
    from src.ivc.ft import run_enc
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out / 'encoder.pt')
    torch.save(dict(enc=state, kind='encoder_only', step=step), out / 'encoder_eval.pt')
    clean = encoder(out / 'encoder.pt', sample.device)
    was = model.training
    model.eval()
    with torch.no_grad():
        before, before_cls = run_enc(model, sample)
        calls = []
        handle = clean.patch_embed.register_forward_hook(lambda m, a, b: calls.append(len(a[0])))
        after, after_cls = run_enc(clean, sample)
        handle.remove()
        assert torch.equal(before, after) and torch.equal(before_cls, after_cls)
        assert after.shape == (len(sample), 576, 768)
        ref = encoder(reference or out / 'encoder.pt', 'cpu')
        assert type(clean) is type(ref)
        assert [(n, type(m).__name__) for n, m in clean.named_modules()] == [(n, type(m).__name__) for n, m in ref.named_modules()]
        assert sum(p.numel() for p in clean.parameters()) == sum(p.numel() for p in ref.parameters())
        assert sum(calls) == len(sample)
    report = dict(strict_loading=True, class_name=type(clean).__module__ + '.' + type(clean).__name__,
                  parameter_count=sum(p.numel() for p in clean.parameters()), original_parameter_count=sum(p.numel() for p in ref.parameters()),
                  patch_shape=list(after.shape), input_shape=list(sample.shape), bit_equal_forward=True,
                  encoder_canvases=sum(calls), input_images=len(sample), roi_reencoding=False,
                  identical_module_tree=True, additional_modules=[], state_sha256=state_sha(state),
                  reference_state_sha256=state_sha(ref.state_dict()),encoder_weights_changed=state_sha(state)!=state_sha(ref.state_dict()),
                  export_files=['encoder.pt', 'encoder_eval.pt'], fine_decoder_instantiated=False)
    write_json(out / 'export_verification.json', report)
    model.train(was)
    return report
