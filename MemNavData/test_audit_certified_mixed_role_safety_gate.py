import csv
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.audit_certified_mixed_role_safety_gate import (
    audit,
    certificate_lifecycle,
)


def plan(*, cached, accepted, takeover, reason):
    return {
        "certified_relocalization_cached": cached,
        "certified_relocalization_accepted": accepted,
        "certified_relocalization_reason": reason,
        "revisit_adapter_takeover": takeover,
    }


class CertifiedMixedRoleSafetyGateTest(unittest.TestCase):
    def test_lifecycle_separates_uncached_decision_from_cached_reuse(self):
        value = certificate_lifecycle([
            plan(cached=False, accepted=True, takeover=True,
                 reason="certificate_accepted"),
            plan(cached=True, accepted=True, takeover=True,
                 reason="certificate_accepted"),
        ])
        self.assertEqual(value["requests"], 2)
        self.assertEqual(value["uncached"], 1)
        self.assertEqual(value["accepted_uncached"], 1)
        self.assertEqual(value["takeovers"], 2)

    def _write_arm(self, root: Path, arm: str, *, false_takeover=False):
        arm_root = root / arm
        arm_root.mkdir(parents=True)
        certified = arm == "certified"
        summary = {
            "episodes": 1,
            "server_backend": "hybrid_pose",
            "hybrid_route": ("certified_relocalization" if certified else "phase"),
            "policy_backends": (
                {"A": "navdp_auto", "B": "navdp_auto", "C": "navdp_auto"}
                if certified else
                {"A": "navdp", "B": "navdp", "C": "navdp_mix"}
            ),
            "role_labels": {
                "A": "initial_imagegoal", "B": "novel", "C": "revisit"
            },
            "multigoal_contract": "multileg_v4_role_paired_20260812",
            "contract_valid_episodes": 1,
        }
        (arm_root / "summary.json").write_text(json.dumps(summary))
        metric = {
            "episode": "episode_0000", "seed": "7",
            "multigoal_contract_ok": "1",
            "role_sequence": json.dumps([
                "initial_imagegoal", "novel", "revisit"
            ]),
        }
        for leg, reached, steps, length, distance in (
            ("A", 1, 8, 1.0, 0.9),
            ("B", 1, 16, 2.0, 0.8),
            ("C", 1, 24, 3.0, 0.7),
        ):
            metric.update({
                f"reached_{leg}": str(reached), f"steps_{leg}": str(steps),
                f"len_{leg}": str(length), f"final_dist_{leg}": str(distance),
                f"router_active_plans_{leg}": "0",
            })
        if certified:
            metric["router_active_plans_C"] = "1"
            if false_takeover:
                metric["router_active_plans_B"] = "1"
        with (arm_root / "metric.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric))
            writer.writeheader(); writer.writerow(metric)
        empty = {"legA": [], "legB": [], "legC": []}
        plans = {
            "legA": [], "legB": [], "legC": [],
            "rollout_traces": {key: list(value) for key, value in empty.items()},
            "memory_traces": {key: list(value) for key, value in empty.items()},
        }
        if certified:
            plans["legA"] = [plan(cached=False, accepted=False, takeover=False,
                                       reason="no_causal_candidate")]
            plans["legB"] = [plan(
                cached=False, accepted=false_takeover, takeover=false_takeover,
                reason=("certificate_accepted" if false_takeover
                        else "precheck_fundamental_inliers"),
            )]
            plans["legC"] = [plan(cached=False, accepted=True, takeover=True,
                                       reason="certificate_accepted")]
        (arm_root / "episode_0000_plans.json").write_text(json.dumps(plans))

    def _fixture(self, root: Path, *, false_takeover=False):
        (root / "scenes" / "00_scene").mkdir(parents=True)
        (root / "run_contract.json").write_text(json.dumps({
            "protocol": "certified_mixed_role_safety_gate_v1_20260813",
            "deterministic_plan_seeds": True,
            "blind_data_read": False,
        }))
        scene = root / "scenes" / "00_scene"
        self._write_arm(scene, "known_c_reference")
        self._write_arm(scene, "certified", false_takeover=false_takeover)

    def test_full_audit_accepts_exact_reject_then_revisit_takeover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            report = audit(root)
            self.assertTrue(report["audit_ok"])
            self.assertEqual(report["novel_certificate_accepts"], 0)
            self.assertEqual(report["revisit_positive_controls"], 1)

    def test_false_novel_takeover_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root, false_takeover=True)
            with self.assertRaisesRegex(RuntimeError, "false certificate accept"):
                audit(root)


if __name__ == "__main__":
    unittest.main()

