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
)
from MemNavData.train_lingbot_native_localizer import (
    LingBotNativeLocalizer,
    Prediction,
    apply_pose_gain,
    build_feature_matrix,
    model_loss,
    pack_exact_sessions,
    pose_metrics,
    predict,
    select_pose_gain,
    stratified_scene_split,
    train_model,
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


class LingBotNativeArtifactAuditTest(unittest.TestCase):
    def build_artifact(self, root: Path):
        run = root / "run"
        run.mkdir()
        rows = synthetic_rows()
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
            },
            "provenance": {
                "source_commit": "abc123",
                "lingbot_commit": "lingbot",
                "lingbot_weight_sha256": "weight",
                "teacher_csv_sha256": sha256(teacher_path),
                "split_manifest_sha256": sha256(split_path),
            },
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
        connection.executemany(
            "INSERT INTO completed_sessions VALUES (?, ?, ?, ?, ?)",
            [
                ("scene_a/session_positive", 1, 2, 2, 2),
                ("scene_b/session_no_match", 3, 4, 2, 2),
            ])
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
            self.assertTrue((run / AUDIT_NAME).is_file())

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
        for name in names:
            self.assertNotIn("target_", name)
            self.assertNotIn("error", name)
            self.assertNotIn("teacher", name)

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
