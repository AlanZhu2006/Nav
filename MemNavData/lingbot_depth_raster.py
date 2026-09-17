"""Undo LingBot's image padding before the unchanged NavDP depth encoder.

This is a pixel-coordinate transform, not a learned correction, metric-scale
estimator or navigation gate. Historical geometry remains in LingBot's raster;
only the dense observation readout is returned to its source RGB raster.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PadRaster:
    source_height: int
    source_width: int
    padded_size: int
    resized_height: int
    resized_width: int
    top: int
    left: int

    @property
    def crop(self):
        return (slice(self.top, self.top + self.resized_height),
                slice(self.left, self.left + self.resized_width))


def lingbot_pad_raster(source_hw, *, image_size=518, patch_size=14):
    """Exact dimensions for the vendored load_fn mode='pad' resize rule.

    source_hw is measured from the decoded frame (after any EXIF transpose),
    not inferred from predicted depth or chosen using the navigation outcome.
    """
    height, width = map(int, source_hw)
    if min(height, width, image_size, patch_size) <= 0:
        raise ValueError("Positive source, image and patch dimensions required")
    if width >= height:
        new_width = image_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size
    else:
        new_height = image_size
        new_width = round(width * (new_height / height) / patch_size) * patch_size
    if min(new_width, new_height) < 1 or max(new_width, new_height) > image_size:
        raise ValueError("Input is unsupported by this LingBot pad contract")
    return PadRaster(height, width, image_size, new_height, new_width,
                     (image_size - new_height) // 2, (image_size - new_width) // 2)


def to_source_rgb_raster(depth, raster):
    """Crop nonexistent image rays and resize, preserving metric depth units.

    Bilinear sampling follows NavDP's existing observation resize convention.
    No values from the artificial padding are mixed into the retained crop.
    No rescale/clamp, metric sensor, pose, map, target or role is consulted.
    """
    values = np.asarray(depth)
    if values.shape != (raster.padded_size, raster.padded_size):
        raise ValueError("Depth raster does not match its producer layout")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Decoded metric depth must be a floating array")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative metric depth")
    valid = values[raster.crop]
    return cv2.resize(valid, (raster.source_width, raster.source_height),
                      interpolation=cv2.INTER_LINEAR)
