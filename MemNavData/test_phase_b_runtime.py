"""Contract tests for the ranking-only Phase-B online runtime."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from MemNavData.phase_b_feature_schema import (
    CHECKPOINT_MODEL_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    EXTERNAL_CAUSAL_SCALE_SOURCE,
    FEATURE_DIMENSION,
    FEATURE_NAMES,
    FEATURE_NAMES_SHA256,
    FEATURE_SCHEMA_VERSION,
    METRIC_SCALE_SOURCES,
)
from MemNavData.phase_b_model import LingBotNativeLocalizer
from MemNavData.phase_b_runtime import (
    PhaseBEnsembleRanker,
    PhaseBRuntimeError,
    build_feature_vector,
)
from MemNavData.train_lingbot_native_localizer import build_feature_matrix


def deployment_row(dino: float = 0.90) -> dict[str, object]:
    return {
        "dino_cosine": dino,
        "metric_scale_m_per_raw": 3.2,
        "metric_scale_source": EXTERNAL_CAUSAL_SCALE_SOURCE,
        "depth_scale_raw": 0.7,
        "cloud_overlap_f1_center": 0.4,
        "anchor_goal_distance_norm_center": 0.8,
        "goal_refine_translation_norm_median": 0.1,
        "goal_refine_rotation_deg_median": 2.0,
        "goal_depth_confidence_mean": 0.7,
        "candidate_depth_confidence_mean": 0.8,
        "predicted_relative_xy_m": np.array([1.2, -0.3]),
        "external_scale_valid_frame_ratio": 1.0,
        "external_scale_relative_h_iqr": 0.1,
        "external_scale_clamped": 0.0,
    }


def checkpoint_artifact(*, deployment_approved: bool) -> dict[str, object]:
    model = LingBotNativeLocalizer(FEATURE_DIMENSION, hidden_dim=8, dropout=0.1)
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "deployment_approved": deployment_approved,
        "deployment_input_contract_approved": True,
        "model_kind": CHECKPOINT_MODEL_KIND,
        "input_dim": FEATURE_DIMENSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "metric_scale_source_categories": list(METRIC_SCALE_SOURCES),
        "normalization_mean": [0.0] * FEATURE_DIMENSION,
        "normalization_scale": [1.0] * FEATURE_DIMENSION,
        "config": {"hidden_dim": 8, "dropout": 0.1},
        "states": [model.state_dict()],
        "external_causal_scale_coverage": {
            "approved": True,
            "train_exact_coverage_approved": True,
            "development_exact_coverage_approved": True,
        },
        "train_artifact_identity_sha256": "0" * 64,
        "development_artifact_identity_sha256": "1" * 64,
        "train_audit_sha256": "2" * 64,
        "development_audit_sha256": "3" * 64,
    }


class FeatureParityTest(unittest.TestCase):
    def test_runtime_vector_is_exactly_the_training_allowlist(self) -> None:
        row = deployment_row()
        frame_row = {
            **{key: value for key, value in row.items()
               if key != "predicted_relative_xy_m"},
            "predicted_relative_xy_m_center_json": json.dumps(
                np.asarray(row["predicted_relative_xy_m"]).tolist()),
            "target_relative_xy_m_center_json": "[0.0, 0.0]",
        }
        expected, names, _predicted, _target = build_feature_matrix(
            pd.DataFrame([frame_row]))
        actual = build_feature_vector(row)
        self.assertEqual(names, list(FEATURE_NAMES))
        self.assertTrue(np.array_equal(actual, expected[0]))

    def test_non_finite_or_invalid_external_quality_fails_closed(self) -> None:
        with self.assertRaises(PhaseBRuntimeError):
            build_feature_vector(deployment_row(float("nan")))
        with self.assertRaises(PhaseBRuntimeError):
            build_feature_vector(dict(
                deployment_row(), external_scale_valid_frame_ratio=0.0))


class CheckpointAndRankingTest(unittest.TestCase):
    def _save(self, directory: str, approved: bool) -> Path:
        path = Path(directory) / "phase_b.pt"
        torch.save(
            checkpoint_artifact(deployment_approved=approved), path)
        return path

    def test_unapproved_checkpoint_requires_explicit_experiment_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, approved=False)
            with self.assertRaisesRegex(
                    PhaseBRuntimeError, "not deployment-approved"):
                PhaseBEnsembleRanker(path)
            ranker = PhaseBEnsembleRanker(path, allow_unapproved=True)
            self.assertFalse(ranker.status()["deployment_approved"])
            self.assertTrue(ranker.status()["allow_unapproved"])

    def test_rank_is_a_permutation_and_never_controls_activation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ranker = PhaseBEnsembleRanker(
                self._save(directory, approved=True))
            result = ranker.rank([
                deployment_row(0.91), deployment_row(0.89),
                deployment_row(0.87),
            ])
            self.assertEqual(sorted(result["order"]), [0, 1, 2])
            self.assertFalse(result["activation_uses_model_score"])
            self.assertAlmostEqual(
                sum(result["rank_probability"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
