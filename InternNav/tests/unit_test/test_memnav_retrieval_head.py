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
        torch.testing.assert_close(gate_logit, head.gate_a * feature + head.gate_b)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()
