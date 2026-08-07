import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from MemNavData.audit_lingbot_native_localizer_artifact import (
    AUDIT_NAME,
    CHECKPOINT_NAME,
    CSV_NAME,
    PROGRESS_NAME,
    REPORT_NAME,
    audit_artifact,
    canonical_json,
    sha256,
    validate_formal_selection_policy,
)
from MemNavData.train_lingbot_native_localizer import (
    CHECKPOINT_MODEL_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_DIMENSION,
    FEATURE_NAMES_SHA256,
    FEATURE_SCHEMA_VERSION,
    METRIC_SCALE_SOURCES,
    LingBotNativeLocalizer,
    Prediction,
    apply_pose_gain,
    build_feature_matrix,
    load_pinned_artifact_audit,
    model_loss,
    pack_exact_sessions,
    pose_metrics,
    predict,
    select_pose_gain,
    stratified_scene_split,
    train_model,
    validate_phase_b_checkpoint_artifact,
)
from MemNavData.external_causal_scale_contract import (
    EXTERNAL_CAUSAL_ROW_COLUMNS,
    EXTERNAL_CAUSAL_SCALE_SOURCE,
    validate_external_causal_frame,
)


def synthetic_rows() -> pd.DataFrame:
    records = []
    definitions = (
        ("scene_a", "scene_a/session_positive", 8, 0.8, 1, True, False,
         [1.0, 0.1], [1.1, 0.0]),
        ("scene_a", "scene_a/session_positive", 16, 0.0, 0, True, False,
         [2.0, 0.2], [1.1, 0.0]),
        ("scene_b", "scene_b/session_no_match", 8, 0.1, 0, False, True,
         [0.4, -0.1], [2.0, 0.0]),
        ("scene_b", "scene_b/session_no_match", 16, 0.0, 0, False, True,
         [0.7, 0.1], [2.0, 0.0]),
    )
    for (scene, session, frame, covis, label, has_positive, no_match,
         predicted, target) in definitions:
        records.append({
            "session_id": session,
            "scene": scene,
            "episode": "episode_0000",
            "kind": "cross_episode_train",
            "query_path": f"/{scene}/query/16.jpg",
            "candidate_path": f"/{scene}/candidate/{frame}.jpg",
            "candidate_frame": frame,
            "label": label,
            "session_has_positive": has_positive,
            "session_is_strict_no_match": no_match,
            "session_max_covis": 0.8 if has_positive else 0.1,
            "teacher_covis": covis,
            "dino_cosine": 0.9 - frame / 1000.0,
            "metric_scale_m_per_raw": 2.5,
            "metric_scale_source": "cached_ground_anchored",
            "external_scale_valid_frame_ratio": 1.0,
            "external_scale_relative_h_iqr": 0.0,
            "external_scale_clamped": 0,
            "n_hypotheses": 1,
            "neighbor_offsets": "0",
            "depth_scale_raw": 0.8,
            "goal_pose_translation_dispersion_raw": np.nan,
            "goal_pose_translation_dispersion_norm": np.nan,
            "goal_pose_rotation_dispersion_deg": np.nan,
            "cloud_overlap_f1_center": 0.5 if label == 1 else 0.02,
            "cloud_overlap_f1_mean": 0.5 if label == 1 else 0.02,
            "cloud_overlap_f1_median": 0.5 if label == 1 else 0.02,
            "anchor_goal_distance_norm_center": 1.0,
            "goal_refine_translation_norm_median": 0.01,
            "goal_refine_rotation_deg_median": 1.0,
            "relative_position_error_m_center": 0.2,
            "relative_position_error_m_median": 0.2,
            "relative_position_direction_error_deg_center": 5.0,
            "relative_position_direction_error_deg_median": 5.0,
            "relative_distance_error_m_center": 0.1,
            "relative_rotation_error_deg_center": 1.0,
            "relative_rotation_error_deg_median": 1.0,
            "predicted_relative_xy_m_center_json": json.dumps(predicted),
            "target_relative_xy_m_center_json": json.dumps(target),
            "goal_pose9_center_json": json.dumps([0.0] * 9),
            "goal_depth_confidence_mean": 4.0,
            "candidate_depth_confidence_mean": 5.0,
            "hypotheses_json": "[]",
        })
    return pd.DataFrame(records)


def externalized_rows() -> pd.DataFrame:
    frame = synthetic_rows()
    frame["metric_scale_source"] = EXTERNAL_CAUSAL_SCALE_SOURCE
    for row_index, row in frame.iterrows():
        sample_suffix = "positive" if "positive" in row["session_id"] else "novel"
        values = {
            "causal_manifest_sample_id": f"train/{row['scene']}/{sample_suffix}",
            "causal_split_role": "train",
            "causal_source_episode_id": f"{row['scene']}/episode_0000",
            "causal_goal_source_episode_id": f"{row['scene']}/episode_goal",
            "causal_goal_variant": "counterfactual",
            "causal_goal_role": "B",
            "causal_state_name": "goal_b_t0",
            "causal_decision_frame": 32,
            "causal_prefix_sha256": "1" * 64,
            "causal_navdp_fifo_sha256": "2" * 64,
            "causal_goal_sha256": "3" * 64,
            "causal_manifest_sha256": "4" * 64,
            "causal_manifest_schema_version": (
                "nlsr_v2_multistage_expert_candidate_manifest_v1"),
            "external_scale_artifact_sha256": "5" * 64,
            "external_scale_record_sha256": "6" * 64,
            "external_scale_prefix_end_frame_exclusive": 8,
            "external_scale_cam_pose_prefix_sha256": "7" * 64,
            "external_scale_rgb_prefix_content_sequence_sha256": "8" * 64,
            "external_scale_producer_sha256": "9" * 64,
            "external_scale_configuration_sha256": "a" * 64,
            "external_scale_lingbot_commit": "b" * 40,
            "external_scale_weights_sha256": "c" * 64,
            "external_scale_stream_source_sha256": "d" * 64,
            "external_scale_valid_frame_ratio": 0.75,
            "external_scale_relative_h_iqr": 0.1,
            "external_scale_clamped": 0,
        }
        for column, value in values.items():
            frame.loc[row_index, column] = value
        frame.loc[row_index, "session_id"] = values[
            "causal_manifest_sample_id"]
    self_check = set(EXTERNAL_CAUSAL_ROW_COLUMNS) - set(frame.columns)
    if self_check:
        raise AssertionError(f"external fixture lacks {sorted(self_check)}")
    return frame


class LingBotNativeArtifactAuditTest(unittest.TestCase):
    def test_formal_selection_policy_separates_train_teacher_from_development(self):
        teacher = pd.DataFrame([
            {"scene": "s", "session_id": "positive", "kind": "formal",
             "split_role": "train", "teacher_covis": 0.05},
            {"scene": "s", "session_id": "positive", "kind": "formal",
             "split_role": "train", "teacher_covis": 0.80},
            {"scene": "s", "session_id": "no_match", "kind": "formal",
             "split_role": "train", "teacher_covis": 0.10},
        ])
        train_rows = teacher.copy()
        train_rows["candidate_selection_origin"] = [
            "deployment_topk", "teacher_forced_positive", "deployment_topk"]
        train_signature = {"compute_config": {
            "selection_mode": "train_augmented", "top_k": 1,
            "kind": "formal", "positive_threshold": 0.5,
        }}
        summary = validate_formal_selection_policy(
            train_rows, teacher, train_signature, expected_role="train")
        self.assertTrue(summary["approved"])
        self.assertEqual(summary["positive_session_recall"], 1.0)
        self.assertEqual(summary["maximum_rows_per_session"], 2)

        development_rows = train_rows.loc[
            train_rows["candidate_selection_origin"].eq(
                "deployment_topk")].copy()
        development_rows["split_role"] = "development"
        development_teacher = teacher.copy()
        development_teacher["split_role"] = "development"
        development_signature = {"compute_config": {
            "selection_mode": "deployment", "top_k": 1,
            "kind": "formal", "positive_threshold": 0.5,
        }}
        summary = validate_formal_selection_policy(
            development_rows, development_teacher, development_signature,
            expected_role="development")
        self.assertTrue(summary["approved"])
        self.assertEqual(summary["positive_session_recall"], 0.0)

        leaked = development_rows.copy()
        leaked.loc[leaked.index[0], "candidate_selection_origin"] = (
            "teacher_forced_positive")
        with self.assertRaisesRegex(ValueError, "train-only"):
            validate_formal_selection_policy(
                leaked, development_teacher, development_signature,
                expected_role="development")

    def build_artifact(self, root: Path, *, external: bool = False):
        run = root / "run"
        run.mkdir()
        rows = externalized_rows() if external else synthetic_rows()
        rows.to_csv(run / CSV_NAME, index=False)
        teacher = rows[[
            "session_id", "scene", "episode", "kind", "query_path",
            "candidate_path", "candidate_frame", "teacher_covis",
            "dino_cosine",
        ]].copy()
        teacher_path = root / "teacher.csv"
        teacher.to_csv(teacher_path, index=False)
        split_path = root / "split.json"
        split_path.write_text(json.dumps({
            "train": ["scene_a", "scene_b"],
            "development": ["scene_c"],
            "final_reserved": ["scene_d"],
        }))
        signature = {
            "schema_version": 1,
            "seed_manifest_sha256": "seed",
            "total_seeds": 4,
            "total_sessions": 2,
            "compute_config": {
                "positive_threshold": 0.5,
                "negative_threshold": 0.2,
                "metric_scale_mode": (
                    EXTERNAL_CAUSAL_SCALE_SOURCE
                    if external else "legacy_runtime_or_cached"),
            },
            "provenance": {
                "source_commit": "abc123",
                "lingbot_commit": "lingbot",
                "lingbot_weight_sha256": "weight",
                "teacher_csv_sha256": sha256(teacher_path),
                "split_manifest_sha256": sha256(split_path),
            },
        }
        if external:
            signature["provenance"]["external_causal_scale"] = {
                "mode": EXTERNAL_CAUSAL_SCALE_SOURCE,
                "manifest_sha256": "4" * 64,
                "artifact_sha256": "5" * 64,
            }
        signature_json = canonical_json(signature)
        connection = sqlite3.connect(run / CHECKPOINT_NAME)
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE rows (
                seed_index INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE completed_sessions (
                session_id TEXT PRIMARY KEY,
                first_seed_index INTEGER NOT NULL,
                last_seed_index INTEGER NOT NULL,
                expected_seed_count INTEGER NOT NULL,
                row_count INTEGER NOT NULL
            );
        """)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [("schema_version", "1"), ("signature_json", signature_json)])
        for index, record in enumerate(rows.to_dict("records"), 1):
            # Pandas NaN is intentionally valid in checkpoint row payloads,
            # matching the collector's dispersion fields for offset=0.
            connection.execute(
                "INSERT INTO rows VALUES (?, ?, ?)",
                (index, record["session_id"], json.dumps(record)))
        sessions = []
        for session_id, group in rows.groupby("session_id", sort=False):
            indices = [int(index) + 1 for index in group.index]
            sessions.append((
                str(session_id), min(indices), max(indices),
                len(indices), len(indices),
            ))
        connection.executemany(
            "INSERT INTO completed_sessions VALUES (?, ?, ?, ?, ?)",
            sessions)
        connection.commit()
        connection.close()
        signature_sha = hashlib.sha256(
            signature_json.encode("utf-8")).hexdigest()
        (run / PROGRESS_NAME).write_text(json.dumps({
            "status": "complete",
            "completed_sessions": 2,
            "total_sessions": 2,
            "completed_seeds": 4,
            "total_seeds": 4,
            "saved_rows": 4,
            "signature_sha256": signature_sha,
        }))
        (run / REPORT_NAME).write_text(json.dumps({
            "n_rows": 4,
            "n_sessions": 2,
            "provenance": signature["provenance"],
        }))
        return run, teacher_path, split_path

    def test_strict_artifact_audit_accepts_consistent_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, teacher, split = self.build_artifact(root)
            report = audit_artifact(
                run, teacher, split,
                expected_role="train",
                expected_scenes=2,
                expected_sessions=2,
                expected_seeds=4,
                expected_rows=4,
                expected_source_commit="abc123",
                expected_teacher_sha256=sha256(teacher),
                expected_split_sha256=sha256(split),
            )
            self.assertTrue(report["training_artifact_approved"])
            self.assertEqual(report["counts"]["saved_rows"], 4)
            self.assertEqual(report["teacher_alignment"]["positive_rows"], 1)
            audit_path = run / AUDIT_NAME
            sidecar_path = Path(f"{audit_path}.sha256")
            self.assertTrue(audit_path.is_file())
            digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            self.assertEqual(
                sidecar_path.read_text(), f"{digest}  {AUDIT_NAME}\n")
            repeated = audit_artifact(
                run, teacher, split,
                expected_role="train",
                expected_scenes=2,
                expected_sessions=2,
                expected_seeds=4,
                expected_rows=4,
                expected_source_commit="abc123",
                expected_teacher_sha256=sha256(teacher),
                expected_split_sha256=sha256(split),
            )
            self.assertEqual(repeated, report)
            sidecar_path.write_text("0" * 64 + f"  {AUDIT_NAME}\n")
            with self.assertRaisesRegex(RuntimeError, "sidecar differs"):
                audit_artifact(
                    run, teacher, split,
                    expected_role="train",
                    expected_scenes=2,
                    expected_sessions=2,
                    expected_seeds=4,
                    expected_rows=4,
                    expected_source_commit="abc123",
                    expected_teacher_sha256=sha256(teacher),
                    expected_split_sha256=sha256(split),
                )

    def test_artifact_audit_rejects_progress_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, teacher, split = self.build_artifact(root)
            progress = json.loads((run / PROGRESS_NAME).read_text())
            progress["saved_rows"] = 3
            (run / PROGRESS_NAME).write_text(json.dumps(progress))
            with self.assertRaisesRegex(RuntimeError, "row mismatch"):
                audit_artifact(
                    run, teacher, split,
                    expected_role="train",
                    raise_on_failure=True,
                )
            failed = json.loads((run / AUDIT_NAME).read_text())
            self.assertFalse(failed["training_artifact_approved"])

    def test_artifact_audit_rejects_csv_payload_divergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, teacher, split = self.build_artifact(root)
            rows = pd.read_csv(run / CSV_NAME)
            rows.loc[0, "goal_depth_confidence_mean"] += 1.0
            rows.to_csv(run / CSV_NAME, index=False)
            with self.assertRaisesRegex(RuntimeError, "content mismatch"):
                audit_artifact(
                    run, teacher, split,
                    expected_role="train",
                    raise_on_failure=True,
                )

    def test_artifact_audit_rejects_self_declared_external_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, teacher, split = self.build_artifact(root, external=True)
            with self.assertRaisesRegex(
                    RuntimeError, "external causal-scale contract"):
                audit_artifact(
                    run, teacher, split,
                    expected_role="train",
                )


class LingBotNativeTrainerTest(unittest.TestCase):
    def packed(self):
        frame = synthetic_rows()
        features, names, predicted, target = build_feature_matrix(frame)
        packed = pack_exact_sessions(
            features,
            frame["session_id"].to_numpy(dtype=str),
            frame["scene"].to_numpy(dtype=str),
            frame["teacher_covis"].to_numpy(dtype=np.float64),
            predicted,
            target,
            frame["session_has_positive"].to_numpy(dtype=bool),
            frame["session_is_strict_no_match"].to_numpy(dtype=bool),
            positive_threshold=0.5,
            negative_threshold=0.2,
        )
        return packed, names

    def test_feature_allow_list_excludes_ground_truth(self):
        packed, names = self.packed()
        self.assertEqual(packed.features.shape[-1], len(names))
        self.assertEqual(len(names), FEATURE_DIMENSION)
        for name in names:
            self.assertNotIn("target_", name)
            self.assertNotIn("error", name)
            self.assertNotIn("teacher", name)

    def test_external_scale_has_a_dedicated_one_hot_not_other(self):
        frame = externalized_rows()
        features, names, _predicted, _target = build_feature_matrix(frame)
        external_index = names.index(
            f"metric_scale_source={EXTERNAL_CAUSAL_SCALE_SOURCE}")
        other_index = names.index("metric_scale_source=other")
        np.testing.assert_array_equal(features[:, external_index], 1.0)
        np.testing.assert_array_equal(features[:, other_index], 0.0)
        self.assertEqual(names[-8:-3], [
            *(f"metric_scale_source={source}" for source in METRIC_SCALE_SOURCES),
            "metric_scale_source=other",
        ])
        self.assertEqual(names[-3:], [
            "external_scale_valid_frame_ratio",
            "external_scale_relative_h_iqr",
            "external_scale_clamped",
        ])

    def test_external_scale_row_contract_reports_real_coverage(self):
        coverage = validate_external_causal_frame(externalized_rows())
        self.assertTrue(coverage["approved"])
        self.assertEqual(coverage["external_rows"], 4)
        self.assertEqual(coverage["external_sessions"], 2)
        self.assertEqual(coverage["external_samples"], 2)

    def test_external_scale_rows_cannot_mix_legacy_or_invalid_quality(self):
        mixed = externalized_rows()
        mixed.loc[mixed.index[0], "metric_scale_source"] = "pooled_fallback"
        with self.assertRaisesRegex(RuntimeError, "mixes external and legacy"):
            validate_external_causal_frame(mixed)
        malformed = externalized_rows()
        malformed.loc[
            malformed.index[0], "external_scale_relative_h_iqr"] = np.nan
        with self.assertRaisesRegex(RuntimeError, "relative h_iqr"):
            validate_external_causal_frame(malformed)

    def test_legacy_sixteen_and_seventeen_dimensional_checkpoints_are_rejected(self):
        frame = externalized_rows()
        _features, names, _predicted, _target = build_feature_matrix(frame)
        artifact = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_kind": CHECKPOINT_MODEL_KIND,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "input_dim": FEATURE_DIMENSION,
            "feature_names": names,
            "feature_names_sha256": FEATURE_NAMES_SHA256,
            "metric_scale_source_categories": list(METRIC_SCALE_SOURCES),
            "normalization_mean": [0.0] * FEATURE_DIMENSION,
            "normalization_scale": [1.0] * FEATURE_DIMENSION,
            "deployment_input_contract_approved": True,
            "external_causal_scale_coverage": {
                "approved": True,
                "train_exact_coverage_approved": True,
                "development_exact_coverage_approved": True,
            },
            "train_artifact_identity_sha256": "d" * 64,
            "development_artifact_identity_sha256": "e" * 64,
            "train_audit_sha256": "f" * 64,
            "development_audit_sha256": "0" * 64,
        }
        validate_phase_b_checkpoint_artifact(artifact)
        for dimension in (16, 17):
            legacy = dict(artifact)
            legacy["input_dim"] = dimension
            with self.assertRaisesRegex(RuntimeError, "input dimension"):
                validate_phase_b_checkpoint_artifact(legacy)
        obsolete = dict(artifact)
        obsolete["checkpoint_schema_version"] = 1
        with self.assertRaisesRegex(RuntimeError, "obsolete"):
            validate_phase_b_checkpoint_artifact(obsolete)

    def test_artifact_audit_requires_external_file_sha_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(json.dumps({
                "training_artifact_approved": True,
                "role": "train",
            }), encoding="utf-8")
            pinned = sha256(path)
            audit, actual = load_pinned_artifact_audit(
                path, expected_sha256=pinned, expected_role="train")
            self.assertTrue(audit["training_artifact_approved"])
            self.assertEqual(actual, pinned)

            # The JSON can still self-declare approval, but it is no longer
            # the externally pinned audit receipt after this mutation.
            path.write_text(json.dumps({
                "training_artifact_approved": True,
                "role": "development",
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "audit SHA256 changed"):
                load_pinned_artifact_audit(
                    path, expected_sha256=pinned, expected_role="train")
            with self.assertRaisesRegex(ValueError, "lowercase/complete"):
                load_pinned_artifact_audit(
                    path, expected_sha256="bad", expected_role="train")

    def test_pack_uses_dustbin_and_pose_only_on_selected_positive(self):
        packed, _ = self.packed()
        self.assertEqual(packed.target[0, -1].item(), 0.0)
        self.assertEqual(packed.target[1, -1].item(), 1.0)
        self.assertEqual(int(packed.pose_mask[0].sum()), 1)
        self.assertEqual(int(packed.pose_mask[1].sum()), 0)

    def test_model_is_candidate_permutation_equivariant(self):
        torch.manual_seed(0)
        model = LingBotNativeLocalizer(5, hidden_dim=8, dropout=0.0).eval()
        features = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        logits, no_match, mean, variance = model(features, mask)
        order = torch.tensor([2, 0, 1])
        p_logits, p_no_match, p_mean, p_variance = model(
            features[:, order], mask[:, order])
        self.assertTrue(torch.allclose(logits[:, order], p_logits, atol=1e-6))
        self.assertTrue(torch.allclose(no_match, p_no_match, atol=1e-6))
        self.assertTrue(torch.allclose(mean[:, order], p_mean, atol=1e-6))
        self.assertTrue(torch.allclose(
            variance[:, order], p_variance, atol=1e-6))

    def test_new_pose_head_starts_as_raw_lingbot_pose(self):
        torch.manual_seed(0)
        model = LingBotNativeLocalizer(5, hidden_dim=8, dropout=0.0).eval()
        features = torch.randn(2, 3, 5)
        mask = torch.ones(2, 3, dtype=torch.bool)
        _logits, _no_match, residual, _variance = model(features, mask)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_multitask_loss_has_finite_gradients(self):
        packed, names = self.packed()
        model = LingBotNativeLocalizer(
            len(names), hidden_dim=16, dropout=0.0)
        loss, components = model_loss(
            model, packed, torch.arange(len(packed.session_ids)),
            pose_weight=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(np.isfinite(value) for value in components.values()))
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()))
        for module in (
                model.rank_head, model.no_match_head,
                model.pose_mean_head, model.pose_log_variance_head):
            self.assertTrue(any(
                parameter.grad is not None
                and float(parameter.grad.norm()) > 0.0
                for parameter in module.parameters()))

    def test_strict_no_match_trains_verifier_but_not_listwise_rank(self):
        packed, names = self.packed()
        model = LingBotNativeLocalizer(
            len(names), hidden_dim=16, dropout=0.0)
        loss, components = model_loss(
            model, packed, torch.tensor([1]), pose_weight=1.0)
        loss.backward()
        self.assertEqual(components["rank_loss"], 0.0)
        self.assertGreater(components["candidate_validity_loss"], 0.0)
        self.assertGreater(components["no_match_loss"], 0.0)
        self.assertTrue(any(
            parameter.grad is not None and float(parameter.grad.norm()) > 0.0
            for parameter in model.rank_head.parameters()))
        self.assertTrue(any(
            parameter.grad is not None and float(parameter.grad.norm()) > 0.0
            for parameter in model.no_match_head.parameters()))

    def test_shortlist_miss_is_not_relabelled_as_global_novel(self):
        frame = synthetic_rows().loc[
            lambda rows: rows["scene"].eq("scene_b")].copy()
        frame["session_has_positive"] = True
        frame["session_is_strict_no_match"] = False
        features, _names, predicted, target = build_feature_matrix(frame)
        packed = pack_exact_sessions(
            features,
            frame["session_id"].to_numpy(dtype=str),
            frame["scene"].to_numpy(dtype=str),
            frame["teacher_covis"].to_numpy(dtype=np.float64),
            predicted,
            target,
            frame["session_has_positive"].to_numpy(dtype=bool),
            frame["session_is_strict_no_match"].to_numpy(dtype=bool),
            positive_threshold=0.5,
            negative_threshold=0.2,
        )
        self.assertEqual(packed.selected_match_target.item(), 0.0)
        self.assertEqual(packed.target[0, -1].item(), 1.0)
        self.assertEqual(packed.no_match_target.item(), 0.0)
        self.assertTrue(packed.no_match_supervision_mask.item())

    def test_pose_metrics_distinguish_raw_and_corrected_tail(self):
        packed, _ = self.packed()
        corrected = packed.predicted_xy.numpy().copy()
        corrected[packed.pose_mask.numpy()] = packed.target_xy.numpy()[
            packed.pose_mask.numpy()]
        prediction = Prediction(
            probability=packed.target.numpy(),
            candidate_validity=packed.candidate_target.numpy(),
            no_match_probability=packed.no_match_target.numpy(),
            corrected_xy=corrected,
            variance_xy=np.full_like(corrected, 0.1),
        )
        metrics = pose_metrics(packed, prediction)
        self.assertGreater(metrics["raw_translation_error_median_m"], 0.0)
        self.assertAlmostEqual(
            metrics["corrected_translation_error_p90_m"], 0.0)

    def test_pose_gain_falls_back_to_raw_when_residual_worsens_tail(self):
        packed, _ = self.packed()
        raw = packed.predicted_xy.numpy()
        deliberately_worse = raw.copy()
        deliberately_worse[packed.pose_mask.numpy()] += np.asarray(
            [10.0, 10.0], dtype=np.float32)
        prediction = Prediction(
            probability=packed.target.numpy(),
            candidate_validity=packed.candidate_target.numpy(),
            no_match_probability=packed.no_match_target.numpy(),
            corrected_xy=deliberately_worse,
            variance_xy=np.full_like(deliberately_worse, 0.1),
        )
        gain, metrics = select_pose_gain(packed, prediction)
        selected = apply_pose_gain(packed, prediction, gain)
        self.assertEqual(gain, 0.0)
        self.assertTrue(np.array_equal(selected.corrected_xy, raw))
        self.assertAlmostEqual(
            metrics["corrected_translation_error_p90_m"],
            metrics["raw_translation_error_p90_m"],
        )

    def test_short_training_selects_a_finite_checkpoint(self):
        packed, names = self.packed()
        model, epoch, metrics = train_model(
            packed, packed,
            input_dim=len(names),
            hidden_dim=16,
            dropout=0.0,
            weight_decay=1e-4,
            learning_rate=3e-4,
            batch_size=2,
            epochs=5,
            patience=5,
            pose_weight=1.0,
            pose_tail_weight=0.5,
            pose_tail_fraction=0.2,
            seed=0,
            device=torch.device("cpu"),
            positive_threshold=0.5,
        )
        self.assertEqual(epoch, 5)
        self.assertIsNotNone(metrics)
        prediction = predict(model, packed, torch.device("cpu"))
        self.assertTrue(np.isfinite(prediction.probability).all())
        self.assertTrue(np.isfinite(prediction.corrected_xy).all())
        self.assertTrue(np.isfinite(prediction.variance_xy).all())

    def test_scene_split_preserves_positive_and_novel_classes(self):
        scenes = np.asarray(["a", "b", "c", "d", "e", "f"])
        covisibility = np.asarray([0.8, 0.7, 0.0, 0.1, 0.3, 0.4])
        strict_no_match = np.asarray([False, False, True, True, False, False])
        core, tune = stratified_scene_split(
            scenes, covisibility, strict_no_match,
            positive_threshold=0.5, validation_count=2)
        self.assertEqual(len(core), 4)
        self.assertEqual(len(tune), 2)
        for chosen in (core, tune):
            mask = np.asarray([scene in chosen for scene in scenes])
            self.assertTrue(np.any(covisibility[mask] >= 0.5))
            self.assertTrue(np.any(strict_no_match[mask]))


if __name__ == "__main__":
    unittest.main()
