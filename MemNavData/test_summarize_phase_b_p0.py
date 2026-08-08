import csv
import json
from pathlib import Path
import tempfile
import unittest

from MemNavData.summarize_phase_b_p0 import load_phase_b_audit


class PhaseBP0SummaryAuditTest(unittest.TestCase):
    SHA = "a" * 64

    def _fixture(self, root: Path, *, changed_membership: bool = False) -> Path:
        arm = root / "learned_rank_geometry"
        arm.mkdir()
        (arm / "summary.json").write_text(json.dumps({
            "phase_b_p0_transport_valid": True,
            "phase_b_ranker": {
                "checkpoint_sha256": self.SHA,
                "deployment_approved": False,
                "allow_unapproved": True,
                "activation_semantics": (
                    "diagnostic_only_geometry_gate_unchanged"),
            },
        }))
        fields = [
            "episode", "phase_b_rank_request_count",
            "phase_b_rank_success_count", "phase_b_uncached_rank_count",
            "phase_b_rank_fallback_count",
            "phase_b_activation_violation_count",
            "phase_b_order_change_count", "phase_b_uncached_ranking_ms",
        ]
        with (arm / "metric.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "episode": "episode_0000",
                "phase_b_rank_request_count": 2,
                "phase_b_rank_success_count": 2,
                "phase_b_uncached_rank_count": 1,
                "phase_b_rank_fallback_count": 0,
                "phase_b_activation_violation_count": 0,
                "phase_b_order_change_count": 1,
                "phase_b_uncached_ranking_ms": 12.5,
            })
        used = [3, 2] if changed_membership else [2, 1]
        plan = {
            "step": 0,
            "router_phase_b_requested": True,
            "router_phase_b_cached": False,
            "router_phase_b_success": True,
            "router_phase_b_activation_used": False,
            "router_candidate_order_dino": [1, 2],
            "router_candidate_order_used": used,
        }
        cached = dict(plan, step=8, router_phase_b_cached=True)
        (arm / "episode_0000_plans.json").write_text(json.dumps({
            "legA": [], "legB": [plan, cached],
        }))
        return root

    def test_cached_confirmation_is_not_double_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            counters, sets = load_phase_b_audit(
                self._fixture(Path(directory)), self.SHA)
            self.assertEqual(counters["requests"], 2)
            self.assertEqual(counters["uncached_candidate_sets"], 1)
            self.assertEqual(counters["uncached_ranking_ms"], 12.5)
            self.assertEqual(len(sets), 1)

    def test_shortlist_membership_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._fixture(
                Path(directory), changed_membership=True)
            with self.assertRaisesRegex(RuntimeError, "shortlist membership"):
                load_phase_b_audit(root, self.SHA)


if __name__ == "__main__":
    unittest.main()
