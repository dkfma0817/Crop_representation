"""Read-only ADE20K diagnostic with the original eval geometry and frozen heads."""
import argparse
import heapq
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.fgtrain.common import ROOT, RUN, encoder, write_json
from src.g1eval import ade as ade
from src.g1in.common import checksum

OUT = Path(__file__).resolve().parent
NAMES = ('Original', 'B0', 'P1', 'P2')
SEED = 20261111
NC = ade.NC


def sources(p23):
    ade_original = ROOT/'output/2026-11-07/g1_leakcheck_ade20k/ade'
    p1 = ROOT/'output/2026-11-11/fgtrain_p1_s0'
    return {
        'Original': (RUN/'initial_encoder.pt', ade_original, 'original'),
        'B0': (RUN/'B0/export/encoder.pt', RUN/'ade', 'B0'),
        'P1': (p1/'P1/export/encoder.pt', p1/'ade', 'P1'),
        'P2': (p23/'P2/export/encoder.pt', p23/'ade', 'P2'),
    }


class ADEImages(Dataset):
    def __init__(self, split, ids):
        self.items = [ade.listing(split)[i] for i in ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, ap = self.items[i]
        image, label = ade.load_pair(ip, ap)
        width, height = image.size
        scale = ade.CFG['eval_short'] / min(width, height)
        nw, nh = max(1, round(width*scale)), max(1, round(height*scale))
        resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
        x = ade.norm(resized)
        canvas = torch.zeros(3, math.ceil(nh/16)*16, math.ceil(nw/16)*16)
        canvas[:, :nh, :nw] = x
        aligned = np.asarray(Image.fromarray(label.astype(np.uint8)).resize((nw, nh), Image.Resampling.NEAREST))
        return Path(ip).stem, canvas, torch.from_numpy(label), torch.tensor(aligned), nh, nw


def patch_labels(label, ph, pw):
    padded = np.full((ph, pw), ade.IGN, dtype=np.int32)
    padded[:label.shape[0], :label.shape[1]] = label
    pixels = padded.reshape(ph//16, 16, pw//16, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    offsets = (NC+1)*np.arange(len(pixels), dtype=np.int32)[:, None]
    hist = np.bincount((offsets + np.minimum(pixels, NC)).ravel(), minlength=(NC+1)*len(pixels)).reshape(-1, NC+1)[:, :NC]
    count = hist.sum(1)
    cls = hist.argmax(1)
    purity = hist.max(1) / np.maximum(count, 1)
    regime = np.full(len(count), -1, np.int8)
    regime[(count >= 128) & (purity >= .90)] = 0
    regime[(count >= 128) & (purity >= .50) & (purity < .90)] = 1
    return cls, purity, count, regime


def pixel_regions(label):
    valid = label != ade.IGN
    boundary = torch.zeros_like(valid)
    vertical = valid[1:] & valid[:-1] & (label[1:] != label[:-1])
    horizontal = valid[:, 1:] & valid[:, :-1] & (label[:, 1:] != label[:, :-1])
    boundary[:-1] |= vertical
    boundary[1:] |= vertical
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary = F.max_pool2d(boundary[None, None].float(), 5, stride=1, padding=2)[0, 0].bool() & valid
    return valid, boundary, valid & ~boundary


def hist_confusion(pred, label, region):
    return torch.bincount(NC*label[region] + pred[region], minlength=NC*NC).reshape(NC, NC).cpu().numpy()


def case_indices(stored):
    orig_i, orig_u = stored['Original']
    b0_i, b0_u = stored['B0']
    p2_i, p2_u = stored['P2']
    gt_union = (orig_u > 0) & (b0_u > 0) & (p2_u > 0)
    support = gt_union.sum(1)
    subset = np.flatnonzero((support >= 6) & (orig_i.sum(1) >= 20000))
    scores = [((i/np.maximum(u, 1))*gt_union).sum(1)/np.maximum(support, 1) for i, u in [(orig_i, orig_u), (b0_i, b0_u), (p2_i, p2_u)]]
    candidates = [('B0→P2 recovery', scores[2]-scores[1]), ('Original→P2 residual', scores[0]-scores[2]),
                  ('P2 regression versus B0', scores[1]-scores[2])]
    result = {}
    for name, score in candidates:
        index = next(int(i) for i in subset[np.argsort(-score[subset], kind='stable')] if int(i) not in result)
        result[index] = dict(reason=name, common_union_classes=int(support[index]), macro_iou=[float(s[index]*100) for s in scores])
    return result


@torch.inference_mode()
def run_val(models, heads, stored, workers):
    count = len(ade.listing('val'))
    cases = case_indices(stored)
    conf = np.zeros((len(NAMES), 3, NC, NC), np.int64)
    inter = np.zeros((len(NAMES), count, NC), np.int64)
    union = np.zeros_like(inter)
    ids = []
    loader = DataLoader(ADEImages('val', range(count)), batch_size=None, num_workers=workers)
    for i, (iid, x, label, _, nh, nw) in enumerate(loader):
        ids.append(iid)
        x = x[None].cuda()
        label = label.cuda()
        regions = pixel_regions(label)
        save = i in cases
        masks = {}
        for m, name in enumerate(NAMES):
            logits = ade.logits(heads[name], ade.feats(models[name], x))
            logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, :, :nh, :nw]
            pred = F.interpolate(logits, size=label.shape, mode='bilinear', align_corners=False).argmax(1)[0]
            matrix = hist_confusion(pred, label, regions[0])
            conf[m, 0] += matrix
            for r in (1, 2):
                conf[m, r] += hist_confusion(pred, label, regions[r])
            inter[m, i] = matrix.diagonal()
            union[m, i] = matrix.sum(0)+matrix.sum(1)-matrix.diagonal()
            if save:
                masks[name] = pred.cpu().numpy().astype(np.uint8)
        if save:
            cases[i]['image_id'] = iid
            np.savez_compressed(OUT/f'val_case_{iid}.npz', label=label.cpu().numpy().astype(np.uint8), **masks)
        if (i+1) % 100 == 0:
            print(f'ADE Val inference {i+1}/{count}', flush=True)
    assert np.array_equal(conf[:, 0], conf[:, 1]+conf[:, 2])
    np.savez_compressed(OUT/'val_confusion.npz', confusion=conf, intersection=inter, union=union, ids=ids)
    write_json(OUT/'val_cases.json', cases)


def add_sample(heap, key, features, image, patch, purity, cap=64):
    if len(heap) >= cap and key <= heap[0][0]:
        return
    row = (key, image, patch, float(purity), features)
    if len(heap) < cap:
        heapq.heappush(heap, row)
    else:
        heapq.heapreplace(heap, row)


@torch.inference_mode()
def run_train(models, workers, n):
    rng = np.random.default_rng(SEED)
    ids = np.sort(rng.choice(len(ade.listing('train')), n, replace=False))
    sample_rng = np.random.default_rng(SEED+1)
    loader = DataLoader(ADEImages('train', ids), batch_size=None, num_workers=workers)
    cases = {int(n*.13), int(n*.49), int(n*.83)}
    pools = [[[] for _ in range(NC)] for _ in range(2)]
    image_rows, keys = [], []
    regional = np.zeros((len(NAMES), 2, NC, 2), np.float64)
    patch_counts = np.zeros(3, np.int64)
    for i, (iid, x, _, aligned, nh, nw) in enumerate(loader):
        keys.append(iid)
        ph, pw = x.shape[-2:]
        shape = (ph//16, pw//16)
        cls, purity, _, regime = patch_labels(aligned.numpy(), ph, pw)
        # Partial patches get precisely the fraction of real canvas area.
        cov = (np.clip(nh-np.arange(shape[0])*16, 0, 16)[:, None] * np.clip(nw-np.arange(shape[1])*16, 0, 16)[None, :]/256).ravel()
        patch_counts += np.bincount(regime+1, minlength=3)
        x = x[None].cuda()
        f = torch.stack([F.normalize(ade.feats(models[name], x)[0].flatten(1).T, dim=1) for name in NAMES])
        v = torch.as_tensor(cov, dtype=torch.float32, device=x.device)
        cos = torch.einsum('mnd,nd->mn', f, f[0]).clamp(-1, 1)
        cos[0] = 1
        l2 = (2*(1-cos)).sqrt()
        w = f*v.sqrt()[None, :, None]
        ref = w[0]
        # Frobenius identity computes all patch pairs without an N-by-N allocation.
        ref_norm = (ref.T@ref).square().sum()
        gram = torch.stack([(z.T@z).square().sum()+ref_norm-2*(z.T@ref).square().sum() for z in w]).clamp_min(0)/v.sum().square()
        gram[0] = 0
        patch_stats = torch.stack([(cos*v).sum(1)/v.sum(), (l2*v).sum(1)/v.sum(), gram], dim=1).cpu().numpy()
        drift = (1-cos).cpu().numpy()
        for r in (0, 1):
            mask = regime == r
            labels = cls[mask]
            regional[:, r, :, 1] += np.bincount(labels, minlength=NC)[None, :]
            for m in range(len(NAMES)):
                regional[m, r, :, 0] += np.bincount(labels, weights=drift[m, mask], minlength=NC)
        chosen = []
        for r in (0, 1):
            for c in np.unique(cls[regime == r]):
                heap = pools[r][c]
                key = float(sample_rng.random())
                if len(heap) >= 64 and key <= heap[0][0]:
                    continue
                patch = int(sample_rng.choice(np.flatnonzero((regime == r) & (cls == c))))
                chosen.append((r, c, key, patch))
        if chosen:
            values = f[:, [row[3] for row in chosen]].cpu().numpy().transpose(1, 0, 2)
            for (r, c, key, patch), feature in zip(chosen, values):
                add_sample(pools[r][c], key, feature, i, patch, purity[patch])
        if i in cases:
            np.savez_compressed(OUT/f'train_case_{iid}.npz', drift=drift.reshape(len(NAMES), *shape),
                                coverage=cov.reshape(shape), regime=regime.reshape(shape), purity=purity.reshape(shape))
        image_rows.append(patch_stats)
        if (i+1) % 100 == 0:
            print(f'ADE Train features {i+1}/{n}', flush=True)
    feats, metadata = [], []
    for r, classes in enumerate(pools):
        for c, heap in enumerate(classes):
            for _, image, patch, purity, feature in sorted(heap, reverse=True):
                feats.append(feature)
                metadata.append((r, c, image, patch, purity))
    np.savez_compressed(OUT/'train_features.npz', features=np.stack(feats), metadata=metadata, ids=keys,
                        image_stats=np.stack(image_rows), regional=regional, patch_counts=patch_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p23-run', type=Path, required=True)
    parser.add_argument('--train-sample', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    models, heads, stored = {}, {}, {}
    import json
    with torch.inference_mode():
        for name, (ckpt, directory, key) in sources(args.p23_run).items():
            cfg = json.loads((directory/'config.json').read_text())
            models[name] = encoder(ckpt)
            value = checksum(models[name])
            assert abs(value-cfg['checksum'][key]) < 1e-6, (name, value, cfg['checksum'][key])
            head = torch.nn.Linear(768, NC).cuda().eval().requires_grad_(False)
            head.load_state_dict(torch.load(directory/f'head_{key}.pt', map_location='cpu', weights_only=True))
            heads[name] = head
            with np.load(directory/'val_inter_union.npz') as archive:
                stored[name] = archive[f'I_{key}'], archive[f'U_{key}']
            print(f'Loaded {name}, checkpoint checksum {value:.10f}', flush=True)
    if not (OUT/'val_confusion.npz').exists():
        run_val(models, heads, stored, args.workers)
    if not (OUT/'train_features.npz').exists():
        run_train(models, args.workers, args.train_sample)
    print('Inference complete', flush=True)


if __name__ == '__main__':
    main()
