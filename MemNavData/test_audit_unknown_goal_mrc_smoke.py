import json
import unittest

import pandas as pd

from MemNavData.audit_unknown_goal_mrc_smoke import validate_contract


class UnknownGoalMrcSmokeAuditTest(unittest.TestCase):
    @staticmethod
    def fixtures():
        sessions = [f"train/scene_{index % 20:02d}/ep_{index}/goal"
                    for index in range(24)]
        hypothesis = {
            "anchor": 20,
            "cloud_overlap_f1": 0.5,
            "depth_scale_raw": 2.0,
            "goal_pose": [0.0] * 9,
            "goal_refine_rotation_deg": 1.0,
            "goal_refine_translation_raw": 0.1,
            "offset": 0,
            "predicted_relative_xy_m": [1.0, 0.0],
        }
        rows = []
        for index, session in enumerate(sessions):
            hypotheses = []
            for offset in (-4, 0, 4):
                item = dict(hypothesis)
                item["offset"] = offset
                item["anchor"] = 20 + offset
                hypotheses.append(item)
            rows.append({
                "session_id": session,
                "scene": f"scene_{index % 20:02d}",
                "candidate_frame": 20,
                "causal_decision_frame": 30,
                "causal_split_role": "train",
                "candidate_selection_origin": "deployment_topk",
                "n_hypotheses": 3,
                "neighbor_offsets": "-4;0;4",
                "metric_scale_source": "external_causal_first_prefix_v1",
                "goal_pose_translation_dispersion_norm": 0.1,
                "goal_pose_rotation_dispersion_deg": 1.0,
                "cloud_overlap_f1_mean": 0.5,
                "goal_refine_translation_norm_median": 0.1,
                "goal_refine_rotation_deg_median": 1.0,
                "hypotheses_json": json.dumps(hypotheses),
            })
        manifest = {
            "schema_version": "unknown_goal_mrc_v0_smoke_sessions_v1",
            "selected_session_count": 24,
            "selected_scene_count": 20,
            "source_teacher_sha256": "teacher",
            "sessions": sessions,
        }
        report = {
            "config": {
                "selection_mode": "deployment",
                "top_k": 1,
                "adaptive_neighbor_radius": 4,
                "adaptive_neighbor_count": 3,
                "adaptive_neighbor_policy": "maximin_spacing_v1",
                "full_replay": True,
            },
            "provenance": {
                "teacher_csv_sha256": "teacher",
                "elapsed_seconds": 120.0,
            },
        }
        progress = {
            "status": "complete",
            "completed_sessions": 24,
            "cuda_memory": {"peak_allocated_gib": 12.0},
        }
        return pd.DataFrame(rows), report, progress, manifest

    def test_contract_passes_without_label_metrics(self):
        receipt = validate_contract(*self.fixtures())
        self.assertEqual(
            receipt["status"],
            "contract_smoke_passed_not_effectiveness_evidence")
        self.assertTrue(receipt["label_metrics_intentionally_omitted"])
        self.assertNotIn("auc", receipt)
        self.assertEqual(receipt["seconds_per_session"], 5.0)

    def test_contract_rejects_variable_view_count(self):
        rows, report, progress, manifest = self.fixtures()
        rows.loc[0, "n_hypotheses"] = 2
        with self.assertRaisesRegex(RuntimeError, "exactly three"):
            validate_contract(rows, report, progress, manifest)

    def test_train40_challenge_scope_accepts_explicit_larger_count(self):
        rows, report, progress, manifest = self.fixtures()
        manifest["schema_version"] = (
            "train40_certificate_challenge_manifest_v1")
        receipt = validate_contract(
            rows, report, progress, manifest, expected_session_count=24)
        self.assertEqual(
            receipt["status"],
            "train40_challenge_contract_passed_not_closed_loop")


if __name__ == "__main__":
    unittest.main()
