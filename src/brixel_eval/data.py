"""PACO B0/B1/B2 inputs using BRIXEL's official square-resize transform.

B0: full image -> 256 square -> 16x16 backbone tokens (stride 16).
B1: the identical full input -> native 64x64 BRIXEL tokens (stride 4).
    models.py's adapter returns 4*(S//16) cells per side; ConvHead preserves it.
B2: original-pixel integer GT crop -> 256 square -> 16x16 backbone tokens.

An item is a list of image-group indices, as in src.g1x.data.PacoDS. Full
geometry uses x*S/W, y*S/H independently; crop geometry covers the whole
square. geo = [rel_x, rel_y, cell_overlap, log(box_w/stride),
log(box_h/stride)] in row-major order; box4 = [x, y, w, h]/S.
No padding, letterbox, feature interpolation, encoder loading, or training.

Run CPU checks: python -m src.brixel_eval.data [--n-img 8]
"""
import math
import operator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.readout.common import C


def make_transform(S=256):
    """BRIXEL README transform: resize uint8 with default bilinear, then scale."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((S, S), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _stride(S, grid):
    S, grid = operator.index(S), operator.index(grid)
    if S <= 0 or grid <= 0:
        raise ValueError("S and grid must be positive integers")
    return S / grid


def roi_geo(box_c, S=256, grid=16):
    """Canvas-pixel ltrb -> float32 (geo [grid**2,5], box4 [4], cov)."""
    P = _stride(S, grid)
    X0, Y0, X1, Y1 = (float(v) for v in box_c)
    bw, bh = X1 - X0, Y1 - Y0
    if bw <= 0 or bh <= 0:
        raise ValueError(f"ROI must have positive area: {box_c}")
    e = np.arange(grid + 1) * P
    ox = np.clip(np.minimum(e[1:], X1) - np.maximum(e[:-1], X0), 0, None) / P
    oy = np.clip(np.minimum(e[1:], Y1) - np.maximum(e[:-1], Y0), 0, None) / P
    cov = (oy[:, None] * ox[None, :]).reshape(-1).astype(np.float32)
    ci, cj = np.divmod(np.arange(grid * grid), grid)
    geo = np.stack([
        ((cj + 0.5) * P - X0) / bw,
        ((ci + 0.5) * P - Y0) / bh,
        cov,
        np.full(grid * grid, math.log(bw / P)),
        np.full(grid * grid, math.log(bh / P)),
    ], 1).astype(np.float32)
    box4 = np.array([X0 / S, Y0 / S, bw / S, bh / S], np.float32)
    return geo, box4, cov


def full_geo(W, H, ltrb, S=256, grid=16):
    """Integer original-pixel GT box on the anisotropically resized full image."""
    if W <= 0 or H <= 0:
        raise ValueError("Image dimensions must be positive")
    l, t, r, b = (int(v) for v in ltrb)
    return roi_geo((l * S / W, t * S / H, r * S / W, b * S / H), S, grid)


def crop_geo(S=256, grid=16):
    """The tight crop fills the square: cov=1, box4=(0,0,1,1)."""
    return roi_geo((0, 0, S, S), S, grid)


class PacoDS(Dataset):
    """One item = list of image indices; B2 img_index points to ROI crops.

    grid is the actual output cells per side, configurable without changing
    input pixels. Defaults at S=256: B0/B2=16, B1=64. S is exposed for geometry
    checks; all three evaluation conditions use S=256.
    """

    def __init__(self, df, Y, flip=False, mode="B0", S=256, grid=None):
        mode = mode.upper()
        if mode not in ("B0", "B1", "B2"):
            raise ValueError(f"Unknown condition: {mode}")
        self.S = operator.index(S)
        self.grid = operator.index(grid) if grid is not None else self.S // (4 if mode == "B1" else 16)
        self.P = _stride(self.S, self.grid)
        self.groups = C.image_groups(df)
        self.p, self.box = df.path.values, df[["l", "t", "r", "b"]].values.astype(int)
        self.WH = df[["W", "H"]].values.astype(int)
        self.lv = (df.level.values == "part").astype(np.int64)
        self.Y, self.flip, self.mode = torch.as_tensor(Y), flip, mode
        self.n_roi = np.array([len(g) for g in self.groups])
        self.transform = make_transform(self.S)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        if not len(idx):
            raise ValueError("An image-index batch must be nonempty")
        xs, img_index, rows_all, groups = [], [], [], []
        per = {k: [] for k in ("geo", "box4", "cov")}
        for k, gi in enumerate(idx):
            rows = self.groups[gi]
            with Image.open(self.p[rows[0]]) as source:
                img = source.convert("RGB")
            W, H = img.size
            assert (W, H) == tuple(self.WH[rows[0]]), self.p[rows[0]]
            # Exactly one draw per image iff flip is enabled, as in PacoDS.
            f = bool(self.flip and torch.rand(1).item() < 0.5)
            if f:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.mode != "B2":
                xs.append(self.transform(img))
            for r in rows:
                bx = C.flip_box(tuple(self.box[r]), W) if f else tuple(self.box[r])
                if self.mode == "B2":
                    if bx[2] <= bx[0] or bx[3] <= bx[1]:
                        raise ValueError(f"ROI must have positive area: {bx}")
                    xs.append(self.transform(img.crop(bx)))
                    values = crop_geo(self.S, self.grid)
                    img_index.append(len(xs) - 1)
                else:
                    values = full_geo(W, H, bx, self.S, self.grid)
                    img_index.append(k)
                for key, value in zip(("geo", "box4", "cov"), values):
                    per[key].append(value)
            rows_all.append(rows)
            groups.extend([k] * len(rows))
        rows_all = np.concatenate(rows_all)
        out = {key: torch.from_numpy(np.stack(values)) for key, values in per.items()}
        out.update(
            x=torch.stack(xs), side=self.S,
            img_index=torch.as_tensor(img_index, dtype=torch.long),
            rows=torch.from_numpy(rows_all), lv=torch.from_numpy(self.lv[rows_all]),
            Y=self.Y[rows_all], group=torch.as_tensor(groups, dtype=torch.long),
        )
        return out


def iter_roi_reader(reader, feats, b, roi_batch_size=4, **kwargs):
    """Yield (ROI slice, reader(feats, sliced_batch)) without expanding all ROIs.

    Compatible with src.readout.models.Reader / src.ivc.common.R0: feats is a
    dict of image token tensors [B_img, grid**2, D] (B_crop for B2). Pass native
    BRIXEL maps as {12: maps.flatten(2).transpose(1, 2)}. Image features, x,
    and side are shared across chunks; img_index values keep global indexing.

    Use torch.no_grad() for evaluation. For frozen-feature reader training,
    backward each yielded loss before advancing, using the whole batch's loss
    denominator. Collecting differentiable outputs retains every chunk's graph
    and defeats the activation-memory bound. This iterator never gathers a
    [total_ROIs, grid**2, D] tensor; gathering occurs inside the reader.
    """
    roi_batch_size = operator.index(roi_batch_size)
    if roi_batch_size <= 0:
        raise ValueError("roi_batch_size must be positive")
    roi_keys = {"geo", "cov", "box4", "rows", "img_index", "lv", "Y", "group"}
    for start in range(0, len(b["rows"]), roi_batch_size):
        sl = slice(start, start + roi_batch_size)
        chunk = {key: value[sl] if key in roi_keys else value for key, value in b.items()}
        yield sl, reader(feats, chunk, **kwargs)


def _check_geometry():
    """Independent scalar rectangle intersections, centres, logs, and area sums."""
    count = 0
    for S, grid in ((256, 16), (256, 64), (256, 32), (384, 24)):
        P = S / grid
        for W, H in ((641, 317), (317, 641), (513, 513), (1, 7)):
            for bx in ((0, 0, W, H), (0, 0, 1, 1), (W - 1, H - 1, W, H),
                       (W // 3, H // 4, max(W // 3 + 1, W * 2 // 3), max(H // 4 + 1, H * 3 // 4))):
                geo, box4, cov = full_geo(W, H, bx, S, grid)
                l, t, r, b = bx
                x0, y0, x1, y1 = l / W * S, t / H * S, r / W * S, b / H * S
                bw, bh = x1 - x0, y1 - y0
                expected = []
                for i in range(grid):
                    for j in range(grid):
                        area = max(0., min((j + 1) * P, x1) - max(j * P, x0))
                        area *= max(0., min((i + 1) * P, y1) - max(i * P, y0))
                        expected.append([((j + .5) * P - x0) / bw, ((i + .5) * P - y0) / bh,
                                         area / (P * P), math.log(bw / P), math.log(bh / P)])
                np.testing.assert_allclose(geo, expected, rtol=1e-6, atol=1e-5)
                np.testing.assert_allclose(box4, [l / W, t / H, (r - l) / W, (b - t) / H], atol=1e-7)
                np.testing.assert_array_equal(cov, geo[:, 2])
                np.testing.assert_allclose(cov.sum(dtype=np.float64) * P * P, bw * bh, rtol=1e-6)
                assert cov.min() >= 0 and cov.max() <= 1
                count += 1
        geo, box4, cov = crop_geo(S, grid)
        np.testing.assert_array_equal(cov, np.ones(grid * grid, np.float32))
        np.testing.assert_array_equal(box4, [0, 0, 1, 1])
        np.testing.assert_allclose(geo[:, 3:], math.log(grid), rtol=1e-6)
    # Legacy equality is applicable only where square resizing and letterbox
    # coincide: square full images, and square crops. Nonsquare inputs differ.
    from src.g1x.data import full_r2_geo, roi_geom
    from src.ivc.bbdata import crop_geo as legacy_crop_geo
    for bx in ((0, 0, 513, 513), (7, 23, 501, 389), (512, 0, 513, 1)):
        old = full_r2_geo(roi_geom(513, 513, bx, 384, 16), 384, 16)
        for actual, expected in zip(full_geo(513, 513, bx, 384, 24), old):
            np.testing.assert_array_equal(actual, expected)
    g = roi_geom(641, 317, (7, 11, 108, 112), 384, 16)
    for actual, expected in zip(crop_geo(384, 24), legacy_crop_geo(g["crop_lb"], 384, 16)):
        np.testing.assert_array_equal(actual, expected)
    return count


def _check_batch(df, Y, idx):
    """Exact official pixels, condition alignment, flip draws, crop mapping."""
    official = v2.Compose([v2.ToImage(), v2.Resize((256, 256), antialias=True),
                           v2.ToDtype(torch.float32, scale=True),
                           v2.Normalize((.485, .456, .406), (.229, .224, .225))])
    groups = C.image_groups(df)
    rows = np.concatenate([groups[gi] for gi in idx])
    group = torch.as_tensor(np.concatenate([np.full(len(groups[gi]), k) for k, gi in enumerate(idx)]))
    b1 = None
    for flip in (False, True):
        for seed in (0, 1):
            torch.manual_seed(seed)
            flips = [bool(flip and torch.rand(1).item() < .5) for _ in idx]
            expected_state = torch.random.get_rng_state()
            batches = {}
            for mode in ("B0", "B1", "B2"):
                ds = PacoDS(df, Y, flip, mode)
                np.testing.assert_array_equal(ds.n_roi, [len(g) for g in groups])
                torch.manual_seed(seed)
                batch = ds[idx]
                assert torch.equal(torch.random.get_rng_state(), expected_state), "flip RNG"
                batches[mode] = batch
                assert torch.equal(batch["rows"], torch.as_tensor(rows))
                assert torch.equal(batch["Y"], torch.as_tensor(Y)[rows])
                assert torch.equal(batch["group"], group)
                assert torch.equal(batch["lv"], torch.as_tensor((df.level.values[rows] == "part").astype(np.int64)))
                assert batch["geo"].shape == (len(rows), ds.grid ** 2, 5)
                assert batch["cov"].shape == (len(rows), ds.grid ** 2)
                assert batch["side"] == 256 and batch["x"].dtype == torch.float32
            b0, b1, b2 = (batches[m] for m in ("B0", "B1", "B2"))
            assert torch.equal(b0["x"], b1["x"]), "B0/B1 pixels"
            assert torch.equal(b0["box4"], b1["box4"]), "B0/B1 box4"
            assert torch.equal(b0["img_index"], group) and torch.equal(b1["img_index"], group)
            assert torch.equal(b2["img_index"], torch.arange(len(rows)))
            assert torch.equal(b2["cov"], torch.ones_like(b2["cov"]))
            assert torch.equal(b2["box4"], torch.tensor([0., 0., 1., 1.]).expand(len(rows), -1))
            ri = 0
            for k, gi in enumerate(idx):
                with Image.open(df.path.values[groups[gi][0]]) as source:
                    img = source.convert("RGB")
                if flips[k]:
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                assert torch.equal(b0["x"][k], official(img)), "official full transform"
                for row in groups[gi]:
                    bx = tuple(int(v) for v in df.iloc[row][["l", "t", "r", "b"]])
                    if flips[k]:
                        bx = C.flip_box(bx, img.width)
                    assert torch.equal(b2["x"][ri], official(img.crop(bx))), "official crop transform"
                    expected = full_geo(*img.size, bx, 256, 16)
                    for key, value in zip(("geo", "box4", "cov"), expected):
                        assert torch.equal(b0[key][ri], torch.from_numpy(value)), key
                    ri += 1
    return b1


def _check_reader(b):
    """Native 4096-token microbatches agree in values and frozen-feature gradients."""
    class TinyReader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.7))
            self.limit = len(b["rows"])

        def forward(self, feats, chunk):
            assert len(chunk["rows"]) <= self.limit
            assert chunk["x"] is b["x"]
            assert feats[12].shape[1] == 4096
            tok = feats[12][chunk["img_index"]]
            mean = (tok * chunk["cov"][..., None]).sum(1) / chunk["cov"].sum(1, keepdim=True)
            return dict(logits=mean * self.weight + chunk["box4"][:, :1], extra={})

    feats = {12: torch.arange(len(b["x"]) * 4096 * 3, dtype=torch.float32).reshape(len(b["x"]), 4096, 3) / 4096}
    reader = TinyReader()
    whole = reader(feats, b)["logits"]
    whole.square().mean().backward()
    expected_grad = reader.weight.grad.clone()
    for size in (1, 2, 4, 5, len(b["rows"]) + 1):
        reader.limit = size
        reader.zero_grad(set_to_none=True)
        outputs = []
        for sl, out in iter_roi_reader(reader, feats, b, size):
            assert torch.equal(out["logits"].detach(), whole[sl].detach())
            (out["logits"].square().sum() / whole.numel()).backward()
            outputs.append(out["logits"].detach())
        assert torch.equal(torch.cat(outputs), whole.detach())
        torch.testing.assert_close(reader.weight.grad, expected_grad)


def check(n_img=0):
    """CPU only; optional real PACO val images, never an encoder or training job."""
    import tempfile
    from pathlib import Path

    import pandas as pd

    with torch.random.fork_rng(devices=[]):
        count = _check_geometry()
        with tempfile.TemporaryDirectory(prefix="brixel_data_check_") as tmp:
            records = []
            rng = np.random.default_rng(5)
            for iid, (W, H) in enumerate(((65, 33), (31, 67), (45, 45))):
                path = Path(tmp) / f"{iid}.png"
                Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(path)
                for j, bx in enumerate(((0, 0, W, H), (1, 2, W - 3, H - 1), (W - 1, 0, W, 1))):
                    records.append(dict(image_id=iid, path=str(path), W=W, H=H,
                                        l=bx[0] + .25, t=bx[1] + .25, r=bx[2] + .25, b=bx[3] + .25,
                                        level="part" if j % 2 else "object"))
            # Interleaved manifest, nontrivial image order and repeated indices.
            df = pd.DataFrame(records).iloc[[0, 3, 6, 1, 4, 7, 2, 5, 8]].reset_index(drop=True)
            Y = rng.integers(-1, 2, (len(df), 59), dtype=np.int8)
            b = _check_batch(df, Y, [2, 0, 2, 1])
            _check_reader(b)
            custom = PacoDS(df, Y, mode="B1", grid=32)[[0]]
            assert custom["geo"].shape == (3, 1024, 5)
        real_rois = 0
        if n_img:
            from src.readout.common import load_data
            df, Y = load_data(splits=("val",), limit=n_img)["val"]
            b = _check_batch(df, Y, list(range(len(C.image_groups(df)))))
            real_rois = len(b["rows"])
    print(f"BRIXEL data CPU checks OK: {count} analytic geometries; exact official transforms; "
          "B0/B1 identical x/box4; B2 crops/cov/box4; flip RNG/order/labels; "
          "grid override; square-compatible legacy @384/grid24; "
          f"4096-token ROI microbatch values/gradients; PACO val={n_img} images/{real_rois} ROIs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-img", type=int, default=0, help="Also check this many real PACO val images")
    check(parser.parse_args().n_img)
