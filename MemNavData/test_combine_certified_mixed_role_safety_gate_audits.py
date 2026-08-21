import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.audit_certified_mixed_role_safety_gate import SCHEMA_VERSION
from MemNavData.combine_certified_mixed_role_safety_gate_audits import combine


def receipt(scene):
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "strict-v4 implementation/causal safety gate; not an SR estimate",
        "audit_ok": True,
        "independent_certificate_reasons": {"no_causal_candidate": 1},
        "records": [{
            "scene": scene, "episode": "episode_0000", "seed": 7,
            "prefix_exact": {"A": {"all_exact": True},
                             "B": {"all_exact": True}},
            "certificate": {
                leg: {"accepted_requests": 0, "takeovers": 0,
                      "runtime_failures": 0, "plans": 1}
                for leg in ("A", "B", "C")
            },
            "revisit_positive_control_eligible": False,
            "revisit_positive_control_activated": False,
            "certified_success": {"A": True, "B": False, "C": False},
        }],
    }


class CombineSafetyGateAuditsTest(unittest.TestCase):
    def test_combines_disjoint_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, scene in enumerate(("a", "b")):
                path = Path(tmp) / f"{index}.json"
                path.write_text(json.dumps(receipt(scene)))
                paths.append(path)
            result = combine(paths, expected_scenes=2)
            self.assertEqual(result["scenes"], 2)
            self.assertEqual(result["novel_legs_audited"], 4)
            self.assertTrue(result["all_novel_prefixes_exact"])
            self.assertEqual(
                result["independent_certificate_reasons"],
                {"no_causal_candidate": 2},
            )

    def test_rejects_duplicate_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(2):
                path = Path(tmp) / f"{index}.json"
                path.write_text(json.dumps(receipt("same")))
                paths.append(path)
            with self.assertRaisesRegex(RuntimeError, "duplicate episode"):
                combine(paths)


if __name__ == "__main__":
    unittest.main()
