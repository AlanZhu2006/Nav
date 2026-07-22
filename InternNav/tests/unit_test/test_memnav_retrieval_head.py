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

    def test_raw_residual_starts_exactly_raw_and_has_immediate_gradients(self):
        torch.manual_seed(4)
        head = RetrievalHead(
            dino_dim=12,
            proj_dim=6,
            rank_mode='raw_residual',
            raw_temp_init=0.01,
        )
        goal = torch.randn(2, 12)
        memory = torch.randn(2, 5, 12)
        candidate = torch.tensor([
            [True, True, False, True, False],
            [False, True, True, True, True],
        ])
        match, _, logits, _ = head(goal, memory, candidate)
        raw = head.raw_cosine(goal, memory)
        floor = torch.finfo(raw.dtype).min
        expected = (raw / 0.01).masked_fill(~candidate, floor)
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(
            match, raw.masked_fill(~candidate, floor).argmax(-1)
        )
        self.assertEqual(float(head.residual_weight_abs_mean.detach()), 0.0)

        loss = -logits[candidate].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(head.raw_log_temp.grad))
        self.assertTrue(torch.isfinite(head.residual_weights.grad).all())
        self.assertGreater(float(head.residual_weights.grad.abs().sum()), 0.0)

    def test_raw_temporal_starts_exactly_raw_and_has_immediate_gradients(self):
        torch.manual_seed(5)
        head = RetrievalHead(
            dino_dim=12,
            proj_dim=6,
            rank_mode='raw_temporal',
            raw_temp_init=0.01,
            temporal_topk=3,
            temporal_residual_max=0.05,
        )
        goal = torch.randn(2, 12)
        memory = torch.randn(2, 6, 12)
        candidate = torch.tensor([
            [True, True, False, True, True, False],
            [False, True, True, True, True, True],
        ])
        match, _, logits, _ = head(goal, memory, candidate)
        raw = head.raw_cosine(goal, memory)
        floor = torch.finfo(raw.dtype).min
        expected = (raw / 0.01).masked_fill(~candidate, floor)
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(
            match, raw.masked_fill(~candidate, floor).argmax(-1)
        )
        self.assertEqual(float(head.residual_weight_abs_mean.detach()), 0.0)

        positive = torch.tensor([
            [True, False, False, False, False, False],
            [False, False, True, False, False, False],
        ])
        numerator = logits.masked_fill(~positive, floor).logsumexp(-1)
        denominator = logits.masked_fill(~candidate, floor).logsumexp(-1)
        (denominator - numerator).mean().backward()
        self.assertTrue(torch.isfinite(head.raw_log_temp.grad))
        self.assertTrue(torch.isfinite(head.temporal_weights.grad).all())
        self.assertGreater(float(head.temporal_weights.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(head.temporal_bias.grad))
        self.assertIsNone(head.proj_goal.weight.grad)
        self.assertIsNone(head.proj_mem.weight.grad)

    def test_temporal_reranker_can_demote_an_isolated_raw_peak(self):
        head = RetrievalHead(
            dino_dim=2,
            proj_dim=2,
            rank_mode='raw_temporal',
            temporal_topk=6,
            temporal_residual_max=0.05,
        )
        cosine = torch.tensor([0.94, 0.95, 0.94, 0.96, 0.70, 0.70])
        memory = torch.stack((cosine, (1.0 - cosine.square()).sqrt()), -1)[None]
        goal = torch.tensor([[1.0, 0.0]])
        candidate = torch.ones(1, 6, dtype=torch.bool)
        raw_match, _ = head.raw_match(goal, memory, candidate)
        self.assertEqual(raw_match.tolist(), [3])

        # Feature 8 is the mean neighbour-minus-centre delta.  A positive
        # coefficient penalizes an isolated peak more than a supported plateau.
        with torch.no_grad():
            head.temporal_weights[8] = 10.0
        match, _, _, _ = head(goal, memory, candidate)
        self.assertEqual(match.tolist(), [1])

        shortlist = head.raw_topk_mask(
            head.raw_cosine(goal, memory), candidate
        )
        self.assertEqual(int(shortlist.sum()), 6)

    def test_legacy_rank_state_gets_complete_zero_residual_migration(self):
        source = RetrievalHead(dino_dim=8, proj_dim=4)
        legacy = source.state_dict()
        for key in (
            'rank_mode_code', 'raw_log_temp',
            'residual_weights', 'residual_max',
        ):
            legacy.pop(key)

        restored = RetrievalHead(
            dino_dim=8, proj_dim=4, rank_mode='raw_residual',
            raw_temp_init=0.02,
        )
        restored.load_state_dict(legacy, strict=True)
        torch.testing.assert_close(
            restored.raw_log_temp.exp(), torch.tensor(0.02)
        )
        torch.testing.assert_close(
            restored.residual_weights, torch.zeros(4)
        )
        self.assertEqual(restored.rank_mode, 'raw_residual')

        default_config = RetrievalHead(dino_dim=8, proj_dim=4)
        default_config.load_state_dict(restored.state_dict(), strict=True)
        self.assertEqual(default_config.rank_mode, 'raw_residual')

        partial = source.state_dict()
        partial.pop('residual_max')
        with self.assertRaisesRegex(RuntimeError, 'residual_max'):
            restored.load_state_dict(partial, strict=True)

    def test_version_one_rank_state_adds_temporal_namespace_but_corruption_fails(self):
        source = RetrievalHead(
            dino_dim=8, proj_dim=4, rank_mode='raw_residual'
        )
        version_one = source.state_dict()
        temporal_names = (
            'temporal_weights', 'temporal_bias',
            'temporal_topk', 'temporal_residual_max',
        )
        for key in temporal_names:
            version_one.pop(key)
        restored = RetrievalHead(dino_dim=8, proj_dim=4)
        restored.load_state_dict(version_one, strict=True)
        self.assertEqual(restored.rank_mode, 'raw_residual')
        torch.testing.assert_close(
            restored.temporal_weights,
            torch.zeros(RetrievalHead.TEMPORAL_FEATURE_DIM),
        )

        corrupt = RetrievalHead(
            dino_dim=8, proj_dim=4, rank_mode='raw_temporal'
        ).state_dict()
        for key in temporal_names:
            corrupt.pop(key)
        with self.assertRaisesRegex(RuntimeError, 'temporal_'):
            restored.load_state_dict(corrupt, strict=True)

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
