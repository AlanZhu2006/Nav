"""Small, dependency-light pose helpers used by live MemNav alignment."""

import numpy as np


# Same rotation-specific basis correction as MemNavTrainer._C_rot.  Translation
# uses a different axis map; do not reuse the aux-pose conversion here.
ROTATION_BASIS = np.diag([-1.0, -1.0, 1.0])


def lingbot_relative_yaw(relative_rotation):
    """Axis-corrected current-camera -> goal-image yaw in radians."""
    rotation = np.asarray(relative_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"relative rotation must be 3x3, got {rotation.shape}")
    corrected = ROTATION_BASIS @ rotation @ ROTATION_BASIS.T
    yaw = float(np.arctan2(corrected[0, 2], corrected[2, 2]))
    return (yaw + np.pi) % (2.0 * np.pi) - np.pi
