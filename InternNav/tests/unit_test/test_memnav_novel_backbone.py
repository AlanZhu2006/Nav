import unittest

import torch

from internnav.model.encoder.navdp_backbone import expand_patch_embed_to_six_channels


class MemNavNovelBackboneTest(unittest.TestCase):
    def test_six_channel_expansion_preserves_pretrained_rgb_response(self):
        torch.manual_seed(7)
        rgb_projection = torch.nn.Conv2d(
            3, 5, kernel_size=3, stride=2, padding=1, bias=True
        )
        fused_projection = expand_patch_embed_to_six_channels(rgb_projection)
        image = torch.randn(2, 3, 17, 19)

        expected = rgb_projection(image)
        actual = fused_projection(torch.cat([image, image], dim=1))

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            fused_projection.weight[:, :3], 0.5 * rgb_projection.weight
        )
        torch.testing.assert_close(
            fused_projection.weight[:, 3:], 0.5 * rgb_projection.weight
        )

    def test_expansion_rejects_non_rgb_projection(self):
        with self.assertRaisesRegex(ValueError, 'three-channel'):
            expand_patch_embed_to_six_channels(torch.nn.Conv2d(4, 8, 3))


if __name__ == '__main__':
    unittest.main()
