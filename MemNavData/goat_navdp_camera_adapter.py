"""Calibrated GOAT-to-NavDP camera adapter.

GOAT's policy camera is portrait (360x640, 42 degree HFOV), while frozen
NavDP was trained on a landscape 480x270, approximately 68 degree camera.
GOAT ImageGoals are square 512x512 renders with per-goal HFOVs ranging from
60 to 120 degrees.  This module defines one observable, pinhole-only adapter:

* render current RGB-D with a dedicated NavDP-shaped sensor; and
* reproject each published GOAT goal image into that same camera intrinsic.

No simulator pose, goal location, depth, or success metric is used.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import cv2
import numpy as np


NAVDP_CAMERA_WIDTH = 480
NAVDP_CAMERA_HEIGHT = 270
NAVDP_CAMERA_HFOV_DEG = 68.0
NAVDP_RGB_SENSOR_UUID = "navdp_rgb"
NAVDP_RGB_SENSOR_TYPE = "NavDPCanonicalRGBSensor"

# Recorded in the frozen NavDP/InternData-N1 trajectory parquet files.  The
# simulator-facing canonical intrinsic below uses square pixels, as Habitat
# does; this receipt is retained to make the small fy discrepancy explicit.
NAVDP_TRAINING_INTRINSIC = np.asarray(
    [
        [355.81463623, 0.0, 240.0],
        [0.0, 351.68701172, 135.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def canonical_navdp_intrinsic() -> np.ndarray:
    focal = (NAVDP_CAMERA_WIDTH / 2.0) / math.tan(
        math.radians(NAVDP_CAMERA_HFOV_DEG) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, NAVDP_CAMERA_WIDTH / 2.0],
            [0.0, focal, NAVDP_CAMERA_HEIGHT / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validated_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(
            "GOAT goal image must be HxWx3 or HxWx4, got {}".format(
                array.shape))
    return np.asarray(array[..., :3], dtype=np.uint8)


def _validated_intrinsic(intrinsic: Any, label: str) -> np.ndarray:
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("{} intrinsic must be a finite 3x3 matrix".format(
            label))
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("{} focal lengths must be positive".format(label))
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("{} intrinsic must be invertible".format(label))
    return matrix


def reproject_goal_to_navdp_camera(
    goal_rgb: Any,
    source_intrinsic: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reproject a calibrated GOAT goal into NavDP's canonical pinhole view.

    A pure rotation-free camera-intrinsic homography is exact for the shared
    optical center and orientation.  Wider goals are centrally cropped;
    narrower goals retain their available pixels and expose black invalid
    borders instead of inventing content.
    """
    source = _validated_rgb(goal_rgb)
    source_k = _validated_intrinsic(source_intrinsic, "source")
    target_k = canonical_navdp_intrinsic()
    source_to_target = target_k.dot(np.linalg.inv(source_k))

    target = cv2.warpPerspective(
        source,
        source_to_target,
        (NAVDP_CAMERA_WIDTH, NAVDP_CAMERA_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid_source = np.full(source.shape[:2], 255, dtype=np.uint8)
    valid_target = cv2.warpPerspective(
        valid_source,
        source_to_target,
        (NAVDP_CAMERA_WIDTH, NAVDP_CAMERA_HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    receipt = {
        "adapter": "calibrated_pinhole_reprojection",
        "source_size": [int(source.shape[1]), int(source.shape[0])],
        "target_size": [NAVDP_CAMERA_WIDTH, NAVDP_CAMERA_HEIGHT],
        "source_intrinsic": source_k.tolist(),
        "target_intrinsic": target_k.tolist(),
        "training_intrinsic": NAVDP_TRAINING_INTRINSIC.tolist(),
        "valid_fraction": float(np.count_nonzero(valid_target)) /
        float(valid_target.size),
    }
    return np.asarray(target, dtype=np.uint8), receipt
