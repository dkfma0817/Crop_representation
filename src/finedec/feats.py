"""Frozen DINOv3 384 features for FA: final patch tokens and X_pre = the input of the final norm (prefix tokens dropped)."""
import torch

from src.tsplit.features import PREFIX


@torch.no_grad()
def student_pre(vit, x, chunk=64):
    fin, pre = [], []
    for i in range(0, len(x), chunk):
        store = []
        h = vit.norm.register_forward_pre_hook(lambda mod, args: store.append(args[0]))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = vit.forward_features(x[i:i + chunk])
        finally:
            h.remove()
        assert len(store) == 1
        fin.append(t[:, PREFIX:].float())
        pre.append(store[0][:, PREFIX:].float())
    return torch.cat(fin), torch.cat(pre)
