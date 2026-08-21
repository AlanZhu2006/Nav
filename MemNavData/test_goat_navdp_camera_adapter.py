import math
import unittest

import numpy as np

from MemNavData.goat_navdp_camera_adapter import (
    NAVDP_CAMERA_HEIGHT,
    NAVDP_CAMERA_WIDTH,
    canonical_navdp_intrinsic,
    reproject_goal_to_navdp_camera,
)


def _intrinsic(width, height, hfov_deg):
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class GoatNavDPCameraAdapterTest(unittest.TestCase):
    def test_identity_camera_reprojection_preserves_pixels(self):
        row = np.arange(NAVDP_CAMERA_WIDTH, dtype=np.uint16) % 256
        image = np.repeat(row[None, :, None], NAVDP_CAMERA_HEIGHT, axis=0)
        image = np.repeat(image, 3, axis=2).astype(np.uint8)

        adapted, receipt = reproject_goal_to_navdp_camera(
            image, canonical_navdp_intrinsic())

        np.testing.assert_array_equal(adapted, image)
        self.assertEqual(
            receipt["target_size"],
            [NAVDP_CAMERA_WIDTH, NAVDP_CAMERA_HEIGHT],
        )
        self.assertEqual(receipt["valid_fraction"], 1.0)

    def test_wide_goat_goal_crops_to_fully_valid_canonical_view(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[251:262, 251:262] = (255, 100, 20)

        adapted, receipt = reproject_goal_to_navdp_camera(
            image, _intrinsic(512, 512, 120.0))

        self.assertEqual(
            adapted.shape,
            (NAVDP_CAMERA_HEIGHT, NAVDP_CAMERA_WIDTH, 3),
        )
        self.assertEqual(receipt["valid_fraction"], 1.0)
        center = adapted[
            NAVDP_CAMERA_HEIGHT // 2 - 2:NAVDP_CAMERA_HEIGHT // 2 + 3,
            NAVDP_CAMERA_WIDTH // 2 - 2:NAVDP_CAMERA_WIDTH // 2 + 3,
        ]
        self.assertGreater(int(center[..., 0].max()), 200)

    def test_narrow_goat_goal_exposes_missing_fov_without_fabrication(self):
        image = np.full((512, 512, 3), 200, dtype=np.uint8)

        adapted, receipt = reproject_goal_to_navdp_camera(
            image, _intrinsic(512, 512, 60.0))

        self.assertLess(receipt["valid_fraction"], 1.0)
        self.assertGreater(receipt["valid_fraction"], 0.5)
        self.assertEqual(int(adapted[:, 0].max()), 0)
        self.assertEqual(int(adapted[:, -1].max()), 0)
        self.assertEqual(
            int(adapted[NAVDP_CAMERA_HEIGHT // 2,
                        NAVDP_CAMERA_WIDTH // 2, 0]),
            200,
        )

    def test_rejects_non_calibrated_intrinsic(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            reproject_goal_to_navdp_camera(
                np.zeros((10, 10, 3), dtype=np.uint8),
                np.zeros((3, 3), dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
