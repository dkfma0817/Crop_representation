"""D2 Fine-only: the trained A1 encoder with the final addition removed.

src.hcf.model.CoarseFine.fuse(A, mem) = A + s_d * Wd(mem["fine"]), applied after ROIAlign of A and of the 48 x 48 Fine map (src.hcf.probe.encode).
Fine-only keeps every operation (forward, Fine LayerNorm, ROIAlign, Wd, s_d, autocast) and returns s_d * Wd(mem["fine"]). Wd and s_d are the checkpoint's trained, frozen values; nothing is added or re-learned.
"""
import types

from src.arch_ablate.model import load_ablation


def fine_contribution(model, mem):
    return model.s_d * model.Wd(mem["fine"])


def load_probe_encoder(path, dev, fine_only=False):
    model, ck = load_ablation(path, dev)
    if fine_only:
        assert ck["interaction"] == "A1", ck.get("interaction")
        model.fuse = types.MethodType(lambda self, A, mem: fine_contribution(self, mem), model)
    return model, ck
