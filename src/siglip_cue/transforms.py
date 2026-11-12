"""Local appearance change of one ROI in native image coordinates: only the luminance detail band is edited, inside the box.

All parameters are fixed before training and drawn from a seeded generator, so the same (step, stream column) always produces
the same ROI, operation, strength and donor patch. Pixels outside the box are copied byte for byte.

    Y  = luma of the ROI, L = G_sigma(Y), H = Y - L        (L keeps the object's shape and shading)
    swap      Y' = Y + mask * (scale * H_donor - H)        (detail band of a real patch from another COCO image)
    boost     Y' = Y + mask * alpha * H
    suppress  Y' = Y - mask * beta * H
    RGB' = RGB + (Y' - Y)                                  (equal offset on R, G, B: chroma is unchanged except where clipped)

mask = feather (zero outside the box, ramps up inside it) x edge guard (keeps the strongest low-pass contours).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

SEED = 20261109
MIN_SIDE = 16
OPS = ('swap', 'boost', 'suppress')
OP_P = (.5, .25, .25)
SIGMA_RANGE = (1.2, 4.0)
SWAP_RANGE = (.7, 1.1)
BOOST_RANGE = (.75, 1.5)
SUPPRESS_RANGE = (.6, 1.)
MAX_DELTA = 24.
DONOR_SOFT = 2.
TARGET_FLOOR = 2.
TARGET_CAP = 8.
LUMA = np.array([.299, .587, .114], np.float32)


def item_rng(step, column):
    return np.random.default_rng([SEED, int(step), int(column)])


def luma(rgb):
    return rgb @ LUMA


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def feather(w, h, f):
    x = np.minimum(np.arange(w) + .5, w - np.arange(w) - .5) / f
    y = np.minimum(np.arange(h) + .5, h - np.arange(h) - .5) / f
    return smoothstep(y)[:, None] * smoothstep(x)[None]


def feather_width(w, h):
    return float(np.clip(.12 * min(w, h), 2, 12))


def edge_guard(y, sigma):
    low = gaussian_filter(y, 2 * sigma, mode='nearest')
    g = np.hypot(np.gradient(low, axis=0), np.gradient(low, axis=1))
    tau = max(float(g.mean() + 2 * g.std()), 1e-3)
    return 1 - smoothstep(g / tau)


def band(y, sigma):
    return y - gaussian_filter(y, sigma, mode='nearest')


def robust_rms(x):
    """1.4826 MAD: the amplitude of the texture, not of the few strong contours in the patch."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def donor_band(donor, w, h, sigma, rng):
    """Detail band of a w x h patch of the donor image; reflected when the donor is smaller than the box."""
    dh, dw = donor.shape[:2]
    if dw < w or dh < h:
        donor = np.pad(donor, ((0, max(0, h - dh)), (0, max(0, w - dw)), (0, 0)), mode='reflect')
        dh, dw = donor.shape[:2]
    x = int(rng.integers(0, dw - w + 1))
    y = int(rng.integers(0, dh - h + 1))
    return band(luma(donor[y:y + h, x:x + w].astype(np.float32)), sigma)


def apply(image, box, donor, rng, region='full'):
    """image: (H, W, 3) uint8. box: integer ltrb. Returns the edited image and a metadata dict.
    region: 'full' (the training setting); 'ring' / 'interior' restrict the edit to the feather band / the area inside it (controls)."""
    l, t, r, b = box
    w, h = r - l, b - t
    assert min(w, h) >= MIN_SIDE and 0 <= l < r <= image.shape[1] and 0 <= t < b <= image.shape[0], box
    rgb = image[t:b, l:r].astype(np.float32)
    y = luma(rgb)
    op = int(rng.choice(len(OPS), p=OP_P))
    sigma = float(np.exp(rng.uniform(*np.log(SIGMA_RANGE))))
    sigma = min(sigma, min(w, h) / 4)
    high = band(y, sigma)
    if OPS[op] == 'swap':
        hd = donor_band(donor, w, h, sigma, rng)
        strength = float(rng.uniform(*SWAP_RANGE))
        rms_d = max(float(hd.std()), 1e-3)
        hd = DONOR_SOFT * np.tanh(hd / (DONOR_SOFT * rms_d))                     # unit-variance donor detail; strong donor edges are compressed, not binarized
        amp = min(max(robust_rms(high), TARGET_FLOOR), TARGET_CAP)               # amplitude of the target's own texture: a smooth surface stays nearly smooth
        delta = strength * amp * hd - high
    elif OPS[op] == 'boost':
        strength = float(rng.uniform(*BOOST_RANGE))
        delta = strength * high
    else:
        strength = float(rng.uniform(*SUPPRESS_RANGE))
        delta = -strength * high
    f = feather_width(w, h)
    ramp = feather(w, h, f)
    mask = ramp.copy()
    if region == 'ring':
        mask = mask * (ramp < 1)
    elif region == 'interior':
        mask = mask * (ramp >= 1)
    else:
        assert region == 'full'
    mask = mask * edge_guard(y, sigma)
    delta = np.clip(mask * delta, -MAX_DELTA, MAX_DELTA)
    out = np.clip(rgb + delta[..., None], 0, 255).round().astype(np.uint8)
    result = image.copy()
    result[t:b, l:r] = out
    changed = out.astype(np.float32) - rgb
    chroma = changed - luma(changed)[..., None]
    meta = dict(op=op, sigma=sigma, strength=strength, feather=f, luma_abs=float(np.abs(luma(changed)).mean()), chroma_abs=float(np.abs(chroma).mean()))
    return result, meta


def color_control(image, box, rng):
    """Sanity reference only (never used in training): a chroma-only edit of the same ROI with the same feather."""
    l, t, r, b = box
    w, h = r - l, b - t
    rgb = image[t:b, l:r].astype(np.float32)
    y = luma(rgb)[..., None]
    gain = rng.uniform(.5, 1.5, 3)
    mask = feather(w, h, feather_width(w, h))[..., None]
    out = np.clip(rgb + mask * (rgb - y) * (gain - 1), 0, 255).round().astype(np.uint8)
    result = image.copy()
    result[t:b, l:r] = out
    return result
