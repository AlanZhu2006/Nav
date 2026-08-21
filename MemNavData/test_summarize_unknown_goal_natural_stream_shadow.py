import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.summarize_unknown_goal_natural_stream_shadow import summarize


def plan(pool=1, active=False, takeover=False):
    trials = [{
        "anchor": 7,
        "score": 0.93,
        "matches": 30,
        "inliers": 20,
        "inlier_ratio": 2 / 3,
    } for _ in range(pool)]
    return {
        "router_active": active,
        "revisit_adapter_takeover": takeover,
        "router_candidate_pool_size": pool,
        "router_candidates_considered": len(trials),
        "router_candidate_trials": trials,
        "step": 0,
        "frame_idx": 8,
    }


def payload(*, active=False, takeover=False):
    plans = {
        "legA": [plan(active=active)],
        "legB": [plan(takeover=takeover)],
        "legC": [plan()],
    }
    plans["rollout_traces"] = {
        leg: [{"step": 0, "x": 0.0, "y": 0.0, "z": 0.0,
               "yaw": 0.0, "jpg_sha256": "a" * 64}]
        for leg in ("legA", "legB", "legC")
    }
    plans["memory_traces"] = {
        "legA": [{"frame_idx": 7, "step": 7, "x": 0.0, "z": 0.0, "yaw": 0.0},
                 {"frame_idx": 8, "step": 8, "x": 0.0, "z": 0.0, "yaw": 0.0}],
        "legB": [{"frame_idx": 9, "step": 0, "x": 0.0, "z": 0.0, "yaw": 0.0}],
        "legC": [{"frame_idx": 10, "step": 0, "x": 0.0, "z": 0.0, "yaw": 0.0}],
    }
    return plans


class StreamShadowSummaryTests(unittest.TestCase):
    def write(self, root: Path, payload):
        (root / "episode_0000_plans.json").write_text(json.dumps(payload))

    def test_complete_nonintervening_trace_passes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.write(root, payload())
            report = summarize(root)
            self.assertTrue(report["contract_pass"])
            self.assertEqual(report["geometry_trials"], 3)

    def test_takeover_fails(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.write(root, payload(takeover=True))
            with self.assertRaisesRegex(ValueError, "takeover"):
                summarize(root)

    def test_partial_candidate_verification_fails(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bad = plan(1)
            bad["router_candidate_pool_size"] = 2
            data = payload()
            data["legB"] = [bad]
            self.write(root, data)
            with self.assertRaisesRegex(ValueError, "full candidate pool"):
                summarize(root)

    def test_missing_trace_fails(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = payload()
            data["memory_traces"]["legA"] = [
                {"frame_idx": 8, "step": 8, "x": 0.0, "z": 0.0, "yaw": 0.0}]
            self.write(root, data)
            with self.assertRaisesRegex(ValueError, "map to saved natural traces"):
                summarize(root)

    def test_explicit_censored_downstream_leg_is_opt_in(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = payload()
            data["legC"] = []
            data["rollout_traces"]["legC"] = []
            data["memory_traces"]["legC"] = []
            self.write(root, data)
            with self.assertRaisesRegex(ValueError, "legC memory trace"):
                summarize(root)
            report = summarize(root, allow_censored_legs=True)
            self.assertTrue(report["contract_pass"])
            self.assertEqual(report["censored_episode_legs"]["legC"], 1)

    def test_zero_candidates_with_censored_downstream_is_valid(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = payload()
            data["legA"] = [plan(0)]
            for leg in ("legB", "legC"):
                data[leg] = []
                data["rollout_traces"][leg] = []
                data["memory_traces"][leg] = []
            self.write(root, data)
            report = summarize(root, allow_censored_legs=True)
            self.assertTrue(report["contract_pass"])
            self.assertFalse(report["memory_support_evidence_observed"])
            self.assertEqual(report["candidate_plans"], 0)
            self.assertEqual(report["geometry_trials"], 0)


if __name__ == "__main__":
    unittest.main()
