import unittest

import torch
import torch.nn.functional as F

from internnav.model.basemodel.memnav.retrieval_head import RetrievalHead


class MemNavRetrievalHeadTest(unittest.TestCase):
    def test_shared_initial_projection_preserves_a_match(self):
        torch.manual_seed(0)
        head = RetrievalHead(dino_dim=32, proj_dim=16)
        self.assertTrue(torch.equal(head.proj_goal.weight, head.proj_mem.weight))
        goal = torch.randn(2, 32)
        memory = torch.randn(2, 7, 32)
        memory[:, 3] = goal
        mask = torch.ones(2, 7, dtype=torch.bool)
        match, _, _, _ = head(goal, memory, mask)
        torch.testing.assert_close(match, torch.tensor([3, 3]))

    def test_gate_uses_raw_cosine_and_handles_empty_candidates(self):
        torch.manual_seed(1)
        head = RetrievalHead(dino_dim=8, proj_dim=4)
        goal = torch.randn(2, 8)
        memory = torch.randn(2, 5, 8)
        mask = torch.tensor([
            [True, False, True, False, False],
            [False, False, False, False, False],
        ])
        _, gate_logit, logits, feature = head(goal, memory, mask)
        raw = torch.einsum(
            'bd,bld->bl', F.normalize(goal.float(), dim=-1),
            F.normalize(memory.float(), dim=-1),
        )
        expected = raw[0, [0, 2]].max()
        torch.testing.assert_close(feature, torch.tensor([expected, -1.0]))
        normalized = (feature - head.gate_center) / head.gate_width
        torch.testing.assert_close(
            gate_logit, head.gate_slope * normalized + head.gate_bias
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_raw_match_respects_candidate_mask(self):
        head = RetrievalHead(dino_dim=3, proj_dim=2)
        goal = torch.tensor([[1.0, 0.0, 0.0]])
        memory = torch.tensor([[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]])
        candidate = torch.tensor([[False, True]])
        match, cosine = head.raw_match(goal, memory, candidate)
        self.assertEqual(match.tolist(), [1])
        torch.testing.assert_close(cosine, torch.tensor([[1.0, 0.8]]))

    def test_gate_starts_at_calibrated_threshold_with_healthy_gradients(self):
        head = RetrievalHead(
            dino_dim=8,
            proj_dim=4,
            gate_center=0.94,
            gate_width=0.04,
            gate_slope_init=1.6,
        )
        self.assertAlmostEqual(
            float(head.effective_gate_threshold.detach()), 0.94, places=6
        )
        features = torch.tensor([0.96, 0.88])
        logits = head.gate_slope * (
            (features - head.gate_center) / head.gate_width
        ) + head.gate_bias
        loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([1.0, 0.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(head.gate_log_slope.grad))
        self.assertTrue(torch.isfinite(head.gate_bias.grad))
        self.assertGreater(abs(float(head.gate_bias.grad)), 1e-3)

    def test_legacy_affine_gate_migrates_without_changing_logits(self):
        torch.manual_seed(2)
        source = RetrievalHead(dino_dim=8, proj_dim=4)
        legacy = source.state_dict()
        for key in ('gate_log_slope', 'gate_bias', 'gate_center', 'gate_width'):
            legacy.pop(key)
        legacy['gate_a'] = torch.tensor(9.75)
        legacy['gate_b'] = torch.tensor(-7.85)

        restored = RetrievalHead(dino_dim=8, proj_dim=4)
        restored.load_state_dict(legacy, strict=True)
        goal = torch.randn(3, 8)
        memory = torch.randn(3, 6, 8)
        mask = torch.ones(3, 6, dtype=torch.bool)
        _, migrated_logits, _, feature = restored(goal, memory, mask)
        torch.testing.assert_close(migrated_logits, 9.75 * feature - 7.85)
        self.assertAlmostEqual(
            float(restored.effective_gate_threshold.detach()), 7.85 / 9.75, places=6
        )


if __name__ == '__main__':
    unittest.main()
