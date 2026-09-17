"""Protect the experimental causal domain and the unchanged full-history arm."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from NavDP.baselines.memnav.gem import sparse
from MemNavData.gem_retrieval_domain import eligible_indices, install, validate_config


class RetrievalDomainTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config_path = self.root / "domain.json"
        self.goal = b"frozen-test-goal"
        self.key = hashlib.md5(self.goal).hexdigest()
        self.config = dict(arm="recent_seven", history_frames=100,
                           recent_indices=[48, 56, 64, 72, 80, 88, 96], goal_key=self.key)
        self.write_config()
        self.memory = sparse.SparseReadout()
        self.memory.shortlists = {}
        self.memory.goal_start_frames = {self.key: 100}
        self.originals = install(sparse, self.config_path, self.root / "audit.jsonl")
        self.scores = torch.zeros(101, device="cpu")
        self.scores[8] = 100.0
        self.scores[49] = 99.0
        self.scores[100] = 200.0
        for order, index in enumerate(self.config["recent_indices"]):
            self.scores[index] = 0.9 - order * 0.1

    def tearDown(self):
        sparse.SparseReadout.shortlist_from_scores, sparse.SparseReadout.read_sparse = self.originals
        self.temp.cleanup()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def call(self):
        return self.memory.shortlist_from_scores(self.key, 99, 100, self.scores)

    def test_exclusion_precedes_temporal_suppression(self):
        result = self.call()
        self.assertEqual([r["anchor"] for r in result], self.config["recent_indices"])
        self.assertNotIn(8, [r["anchor"] for r in result])
        self.assertNotIn(49, [r["anchor"] for r in result])
        self.assertNotIn(100, [r["anchor"] for r in result])

    def test_full_history_delegates_without_changing_candidates(self):
        self.config["arm"] = "full_history"
        self.write_config()
        reference = sparse.SparseReadout()
        reference.shortlists = {}
        expected = self.originals[0](reference, self.key, 99, 100, self.scores)
        self.assertEqual(self.call(), expected)
        self.assertIn(8, [r["anchor"] for r in expected])

    def test_cached_query_cannot_change_domain(self):
        self.call()
        self.config["arm"] = "full_history"
        self.write_config()
        with self.assertRaisesRegex(RuntimeError, "domain changed"):
            self.call()

    def test_cached_candidates_remain_frozen(self):
        expected = self.call()
        self.scores[96] = 500.0
        self.assertEqual(self.call(), expected)

    def test_relocalization_cannot_bypass_domain(self):
        self.call()
        with self.assertRaisesRegex(RuntimeError, "bypassed"):
            self.memory.read_sparse(self.goal, [{"anchor": 8, "score": 100.0}])

    def test_other_goal_cannot_reuse_configuration(self):
        with self.assertRaisesRegex(RuntimeError, "another goal"):
            self.memory.shortlist_from_scores("other-goal", 99, 100, self.scores)

    def test_future_and_malformed_indices_are_rejected(self):
        for indices in ([48, 56, 64, 72, 80, 88, 100], [48, 56, 64, 72, 80, 88, 88],
                        [True, 56, 64, 72, 80, 88, 96]):
            with self.assertRaises(ValueError):
                validate_config(dict(self.config, recent_indices=indices))
        with self.assertRaises(ValueError):
            eligible_indices(self.config, 100, 100)


if __name__ == "__main__":
    unittest.main()
