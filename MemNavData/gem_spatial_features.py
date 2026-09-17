"""Experimental first-write spatial features; never modifies LingBot outputs.

Feature cosine below is an external descriptor affinity, not the model's
attention probability. No image synthesis, cache insertion, or pose update.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def patch_coordinates(raw_wh, size=518, patch=14):
    width, height = map(int, raw_wh)
    if min(width, height) <= 0 or size % patch:
        raise ValueError('Invalid camera raster or patch size')
    scale = size / max(width, height)
    nw, nh = [round(v * scale / patch) * patch for v in (width, height)]
    left, top = (size-nw)//2, (size-nh)//2
    y, x = np.meshgrid(np.arange(size//patch), np.arange(size//patch), indexing='ij')
    uv = np.stack((x.ravel(), y.ravel()), -1).astype(np.float64)*patch + (patch-1)/2
    valid = ((uv[:, 0] >= left) & (uv[:, 0] < left+nw)
             & (uv[:, 1] >= top) & (uv[:, 1] < top+nh))
    ids = np.flatnonzero(valid)
    raw = (uv[valid] - [left, top]) * [width/nw, height/nh]
    return ids, uv[valid], raw


def mutual_cosine(reference, current):
    """Unthresholded mutual nearest neighbours; argmax ties use lowest index."""
    a, b = reference.float(), current.float()
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Feature rows must have the same descriptor dimension')
    if (not torch.isfinite(a).all() or not torch.isfinite(b).all()
            or (a.norm(dim=-1) == 0).any() or (b.norm(dim=-1) == 0).any()):
        raise ValueError('Invalid spatial descriptor')
    if not len(a) or not len(b):
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    score = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
    j = score.argmax(1)
    i = score.argmax(0)
    rows = torch.arange(len(a), device=a.device)
    keep = rows[i[j] == rows]
    cols = j[keep]
    affinity = score[keep, cols].clamp(-1, 1)
    return (keep.cpu().numpy(), cols.cpu().numpy(), affinity.cpu().numpy())


@dataclass
class SpatialCapture:
    """DINO patch baseline and two preregistered GCA block outputs."""
    model: object
    layers: tuple = (17, 23)

    def __post_init__(self):
        self.enabled = False
        self.values = {}
        self.cls = None
        self.handles = [self.model.aggregator.patch_embed.register_forward_hook(self._dino)]
        for index in self.layers:
            self.handles.append(self.model.aggregator.global_blocks[index].register_forward_hook(
                self._global_hook(index)))

    def _dino(self, module, args, output):
        self.cls = output['x_norm_clstoken'].detach().float().cpu()
        if self.enabled:
            self.values['dino_patch'] = output['x_norm_patchtokens'].detach()

    def _global_hook(self, index):
        def capture(module, args, output):
            if self.enabled:
                self.values[f'gca_{index+1}'] = output.detach()
        return capture

    def pop(self, count, patch_ids):
        ids = torch.as_tensor(patch_ids, device=next(self.model.parameters()).device)
        result = {}
        for name, value in self.values.items():
            value = value.reshape(count, -1, value.shape[-1])
            if name != 'dino_patch':
                value = value[:, self.model.aggregator.num_special_tokens:]
            result[name] = value[:, ids].to(dtype=torch.bfloat16, device='cpu').contiguous()
        self.values = {}
        if set(result) != {'dino_patch', *[f'gca_{i+1}' for i in self.layers]}:
            raise RuntimeError('Incomplete spatial feature capture')
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.values = {}
        self.cls = None
