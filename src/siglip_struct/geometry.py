"""Physical cell overlap and ragged real-token selection. No resampling."""
import torch
from torch.nn.utils.rnn import pad_sequence

from src.siglip_pm.distill import cell_w


def centers(g, device):
    q = (torch.arange(g, device=device, dtype=torch.float32)+.5)*(384/g)
    y, x = torch.meshgrid(q, q, indexing='ij')
    return torch.stack([x.flatten(), y.flatten()], -1)


def box_from_bilinear(b):
    """Invert the existing sampler affine geometry, including its random flip."""
    p = (b['bil_grid'].float()+1)*192
    r = b['geo'][..., :2].float()
    w = (p[:, 1, 0]-p[:, 0, 0])/(r[:, 1, 0]-r[:, 0, 0])
    h = (p[:, 24, 1]-p[:, 0, 1])/(r[:, 24, 1]-r[:, 0, 1])
    x, y = p[:, 0, 0]-r[:, 0, 0]*w, p[:, 0, 1]-r[:, 0, 1]*h
    return torch.stack([x, y, x+w, y+h], -1)


def valid_from_canvas(x):
    """Unused: valid regions must come from letterbox geometry, never pixel color."""
    raise RuntimeError('Use the recorded letterbox valid rectangle')


def native_tokens(o, encoder, image_index, boxes, valid_rect, margin=0.):
    """Project native fine384 with existing frozenWd; preserve every selected cell.

    Type is explicit and recoverable from geometry's native stride8 vs16. R2's
    five-input geometry MLP and complete parameterization remain unchanged.
    """
    coarse, fine = o['coarse'], o['mem']['fine']
    if 'fine_projected' in o:
        projected = o['fine_projected']
    else:
        with torch.autocast(coarse.device.type, dtype=torch.bfloat16, enabled=coarse.device.type=='cuda'):
            projected = encoder.Wd(fine).to(coarse.dtype)
    by_type = [coarse, projected]
    maps = [centers(24, boxes.device), centers(48, boxes.device)]
    valid = [cell_w(valid_rect, 24, 16), cell_w(valid_rect, 48, 8)]
    widths = (boxes[:, 2:]-boxes[:, :2]).clamp_min(1e-4)
    expanded = torch.cat([boxes[:, :2]-widths*margin, boxes[:, 2:]+widths*margin], -1)
    all_cov = [cell_w(boxes, 24, 16), cell_w(boxes, 48, 8)]
    selected_cov = [cell_w(expanded, 24, 16), cell_w(expanded, 48, 8)]
    tokens, geometry, kinds = [], [], []
    for r in range(len(boxes)):
        ri = int(image_index[r])
        feats, geo, types = [], [], []
        for k, stride in enumerate([16, 8]):
            keep = (selected_cov[k][r]>0) & (valid[k][ri]>0)
            f, pos = by_type[k][ri, keep], maps[k][keep]
            cov = all_cov[k][r, keep]
            sizes = torch.log(widths[r]/stride).expand(len(f), -1)
            g = torch.cat([(pos-boxes[r, :2])/widths[r], cov[:, None], sizes], -1)
            feats.append(f); geo.append(g)
            types.append(torch.full((len(f),), k, device=boxes.device, dtype=torch.long))
        tokens.append(torch.cat(feats)); geometry.append(torch.cat(geo)); kinds.append(torch.cat(types))
    lengths = torch.tensor([len(v) for v in tokens], device=boxes.device)
    assert bool((lengths>0).all()), 'Every valid GT ROI must select a native cell'
    tok = pad_sequence(tokens, batch_first=True)
    geo = pad_sequence(geometry, batch_first=True)
    typ = pad_sequence(kinds, batch_first=True, padding_value=-1)
    mask = torch.arange(tok.shape[1], device=boxes.device)[None]<lengths[:, None]
    return tok, geo, typ, mask


def query_geometry(boxes):
    p = (torch.arange(4, device=boxes.device, dtype=torch.float32)+.5)/4
    y, x = torch.meshgrid(p, p, indexing='ij')
    rel = torch.stack([x.flatten(), y.flatten()], -1)[None].expand(len(boxes), -1, -1)
    wh = (boxes[:, 2:]-boxes[:, :2]).clamp_min(1e-4)
    absolute = (boxes[:, None, :2]+rel*wh[:, None])/384
    qinput = torch.cat([rel, absolute, (boxes/384)[:, None].expand(-1, 16, -1)], -1)
    geo = torch.cat([rel, torch.ones_like(rel[..., :1]), torch.log(wh/16)[:, None].expand(-1, 16, -1)], -1)
    return qinput, geo
