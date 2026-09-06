"""History-only depth/patch alignment for the train-only relation ablation."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F


def padded_image_mask(raw_hw: tuple[int, int], image_size=518, patch_size=14):
    """Match LingBot load_fn's rounded resize THEN white padding, not square crop."""
    height, width = raw_hw
    if min(height, width) <= 0:
        raise ValueError("positive image dimensions required")
    if width >= height:
        new_w = image_size
        new_h = round(height * image_size / width / patch_size) * patch_size
    else:
        new_h = image_size
        new_w = round(width * image_size / height / patch_size) * patch_size
    top, left = (image_size - new_h) // 2, (image_size - new_w) // 2
    mask = torch.zeros(image_size, image_size, dtype=torch.bool)
    mask[top:top + new_h, left:left + new_w] = True
    return mask


def pooled_fraction(pixel_mask: torch.Tensor, grid_size=8, patch_size=14):
    fine = F.avg_pool2d(pixel_mask.float()[None, None], patch_size, patch_size)
    return F.adaptive_avg_pool2d(fine, (grid_size, grid_size))[0, 0]


def intrinsics_from_pose_fov(pose9, image_hw=(518, 518)):
    pose = np.asarray(pose9, dtype=np.float64)
    height, width = image_hw
    if pose.shape != (9,) or not np.isfinite(pose).all():
        raise ValueError("a finite historical predicted pose9 is required")
    if not all(0 < fov < math.pi for fov in pose[7:9]):
        raise ValueError("predicted FoV must be in (0, pi)")
    return np.asarray([[width / 2 / math.tan(pose[8] / 2), 0., width / 2],
                       [0., height / 2 / math.tan(pose[7] / 2), height / 2],
                       [0., 0., 1.]], dtype=np.float64)


def depth_to_patch_geometry(depth, confidence, intrinsic, pixel_mask,
                            metric_scale, grid_size=8, patch_size=14):
    """XYZ is camera [right,down,forward]; output positions use [forward,left].

    Average 14x14 pixels into DINO patches first, then use the exact 37-to-8
    adaptive pooling support. Averaging the 518 image directly to 8 would use
    different support intervals at the adaptive-pool boundaries.
    """
    depth = torch.as_tensor(depth, dtype=torch.float32).cpu()
    confidence = torch.as_tensor(confidence, dtype=torch.float32).cpu()
    mask = torch.as_tensor(pixel_mask, dtype=torch.bool).cpu()
    k = torch.as_tensor(intrinsic, dtype=torch.float32)
    if depth.ndim != 2 or depth.shape != mask.shape or confidence.shape != depth.shape:
        raise ValueError("aligned depth, confidence, and content mask required")
    if not math.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError("missing causal metric scale is not replaced by a constant")
    valid = mask & torch.isfinite(depth) & (depth > 1e-6) & torch.isfinite(confidence)
    if not valid.any():
        raise ValueError("no valid historical depth")
    height, width = depth.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    safe_depth = torch.where(valid, depth, 0.)
    xyz = torch.stack(((xx-k[0, 2])*safe_depth/k[0, 0],
                       (yy-k[1, 2])*safe_depth/k[1, 1], safe_depth), dim=0)
    # Numerator and validity use the same linear pooling operators.
    fine_sum = F.avg_pool2d(xyz[None], patch_size, patch_size)
    fine_fraction = F.avg_pool2d(valid.float()[None, None], patch_size, patch_size)
    coarse_sum = F.adaptive_avg_pool2d(fine_sum, (grid_size, grid_size))
    fraction = F.adaptive_avg_pool2d(fine_fraction, (grid_size, grid_size))
    coarse = coarse_sum / fraction.clamp_min(1e-12)
    median_depth = float(depth[valid].median())
    normalized = (coarse / median_depth)[0].permute(1, 2, 0).reshape(-1, 3)
    valid_cells = (fraction[0, 0] >= .5).flatten()
    if not valid_cells.any():
        raise ValueError("no content-majority coarse patch")
    # Independent camera projection identity: lift and project the valid pixels.
    px = k[0, 0] * xyz[0, valid] / xyz[2, valid] + k[0, 2]
    py = k[1, 1] * xyz[1, valid] / xyz[2, valid] + k[1, 2]
    projection_error = torch.maximum((px-xx[valid]).abs(), (py-yy[valid]).abs()).max()
    return {
        "xyz_normalized": normalized.numpy(),
        "valid_mask": valid_cells.numpy(),
        "valid_fraction": fraction[0, 0].flatten().numpy(),
        "xyz_patch37_raw": (fine_sum / fine_fraction.clamp_min(1e-12))[0].permute(1,2,0).numpy(),
        "valid_fraction37": fine_fraction[0, 0].numpy(),
        "depth_median_raw": median_depth,
        "normalization_m": median_depth * metric_scale,
        "roundtrip_projection_max_px": float(projection_error),
    }
