#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from independent_verify_hm3d_fullmono_lifelong_natural_b import (
    AUDIT_SCHEMA,
    compare_summary,
    recount_fragments,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IndependentNaturalBAuditVerifierTest(unittest.TestCase):
    def write_fragment(self, root: Path, index: int, scene: str,
                       candidate_count: int) -> Path:
        candidates = []
        for slot in range(candidate_count):
            candidates.append({
                "candidate_slot": slot,
                "candidate_identity": f"e0__natural_b_{slot:02d}",
                "support_band": "unsupported_novel",
                "query_geodesic_m": 3.0,
                "paired_revisit_separation_m": 4.0,
                "max_online_a_covis": 0.01 * (slot + 1),
                "assigned_direction_stratum": "front",
                "initial_path_direction_relative_to_a_end_deg": 30.0,
                "goal_yaw_contract": "identity_hash_eight_world_yaw_bins",
            })
        path = root / f"{index:02d}_{scene}" / "natural_b_audit.json"
        path.parent.mkdir(parents=True)
        payload = {
            "schema_version": AUDIT_SCHEMA,
            "scene": scene,
            "scene_index": index,
            "protocol_sha256": "a" * 64,
            "source_materialized_A_histories": 1,
            "controlled_revisit_constructible_histories": 1,
            "natural_B_constructible_recipients": int(bool(candidates)),
            "natural_B_candidate_histories": len(candidates),
            "query_policy_outcomes_read": False,
            "navigation_outcomes_read": False,
            "evaluation_authorized": False,
            "recipients": [{
                "scene": scene,
                "episode": "e0",
                "status": "constructible" if candidates
                else "no_natural_B_candidate",
                "candidate_count": len(candidates),
                "candidates": candidates,
            }],
        }
        path.write_text(json.dumps(payload) + "\n")
        path.with_name(path.name + ".sha256").write_text(
            f"{sha256(path)}  {path.name}\n"
        )
        return path

    def test_recounts_fragments_and_matches_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self.write_fragment(root, 0, "s0", 2),
                self.write_fragment(root, 1, "s1", 1),
            ]
            recount = recount_fragments(paths, 2)
            self.assertEqual(recount["natural_B_candidate_histories"], 3)
            self.assertEqual(
                recount["natural_B_constructible_scene_clusters"], 2
            )
            summary = {
                key: value for key, value in recount.items()
                if key != "protocol_sha256"
            }
            summary.update({
                "construction_contract": {
                    "maximum_candidates_per_controlled_revisit_history": 4,
                    "minimum_candidate_planar_separation_m": 2.0,
                    "A_to_B_geodesic_m": [2.0, 9.0],
                    "B_to_C_geodesic_m": [2.0, 9.0],
                    "B_max_online_A_covis_exclusive": 0.10,
                    "same_scene_navmesh": True,
                    "goal_rendered_at_frozen_camera_height": True,
                    "cross_online_history_donor_required": False,
                },
                "v3_source_gate_reference": {
                    "minimum_candidate_histories": 96,
                    "minimum_scene_clusters": 15,
                    "met": False,
                    "evaluation_authority_conferred": False,
                },
                "query_policy_outcomes_read": False,
                "navigation_outcomes_read": False,
                "evaluation_authorized": False,
            })
            compare_summary(recount, summary)

    def test_rejects_novel_covis_at_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_fragment(root, 0, "s0", 1)
            payload = json.loads(path.read_text())
            payload["recipients"][0]["candidates"][0][
                "max_online_a_covis"
            ] = 0.10
            path.write_text(json.dumps(payload) + "\n")
            path.with_name(path.name + ".sha256").write_text(
                f"{sha256(path)}  {path.name}\n"
            )
            with self.assertRaisesRegex(RuntimeError, "Novel contract"):
                recount_fragments([path], 1)


if __name__ == "__main__":
    unittest.main()
