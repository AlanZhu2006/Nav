import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from MemNavData.nlsr_set_ranker import (
    NLSRLossConfig,
    NLSRRankerConfig,
    NLSRRankerOutput,
    NLSRSetRanker,
    feature_spec_from_dataset,
    load_portable_checkpoint,
    state_dict_sha256,
    vectorize_candidate_sets,
)
from MemNavData.novel_candidate_set_schema_v2 import (
    validate_candidate_dataset,
    validate_candidate_set,
)
from MemNavData.test_novel_candidate_set_schema_v2 import (
    record as synthetic_record,
)
from MemNavData.train_nlsr_set_ranker import (
    CALIBRATION_NAME,
    COVERAGE_POLICY_ADVISORY_UNAVAILABLE,
    COVERAGE_POLICY_REQUIRED,
    FINAL_CHECKPOINT_NAME,
    MANIFEST_NAME,
    METRICS_NAME,
    PROVENANCE_NAME,
    RESUME_NAME,
    NLSRTrainingError,
    TrainingConfig,
    atomic_torch_save,
    calibrate_zero_bad_threshold,
    canonical_json_bytes,
    class_weights_from_train,
    deterministic_torch_bytes,
    evaluate_calibrated_audit,
    evaluate_loss,
    finite_sample_lcb_quantile,
    fit_development_calibration,
    load_canonical_candidate_rows,
    resume_payload_sha256,
    run_training,
    sha256_file,
    structural_policy_decisions,
    verify_final_bundle,
    zero_failure_upper_bound,
)


def harmful_record(*, scene, role):
    value = synthetic_record(
        scene=scene,
        role=role,
        episode=f"{scene}/episode-0",
        positive=False,
    )
    labels = value["candidates"][1]["labels"]
    labels.update({
        "geodesic_progress_h8_m": 0.0,
        "geodesic_progress_h24_m": 0.0,
        "advantage_h24_m": -0.5,
        "harm": True,
        "useful": False,
        "reachable": True,
        "collision_h8": False,
        "regression_h24": True,
        "proposal_proxy_progress_m": -0.5,
        "proposal_proxy_reachable": True,
        "proposal_proxy_positive": False,
        "proposal_proxy_label_valid": True,
    })
    validate_candidate_set(value)
    return value


def coverage_miss_record(*, scene, role):
    value = synthetic_record(
        scene=scene,
        role=role,
        episode=f"{scene}/episode-0",
        positive=False,
    )
    value["set_labels"].update({
        "candidate_universe_has_positive": True,
        "candidate_coverage_miss": True,
        "coverage_label_valid": True,
    })
    validate_candidate_set(value)
    return value


def candidate_dataset():
    rows = [
        synthetic_record(
            scene="train-a", role="train",
            episode="train-a/episode-0", positive=True),
        harmful_record(scene="train-b", role="train"),
        synthetic_record(
            scene="train-c", role="train",
            episode="train-c/episode-0", positive=False),
        coverage_miss_record(scene="train-d", role="train"),
        synthetic_record(
            scene="development-a", role="development",
            episode="development-a/episode-0", positive=True),
        harmful_record(scene="development-b", role="development"),
    ]
    validate_candidate_dataset(rows)
    return rows


def write_artifact(path: Path, records, *, jsonl=False):
    if jsonl:
        payload = b"\n".join(canonical_json_bytes(row) for row in records)
    else:
        payload = canonical_json_bytes(records)
    path.write_bytes(payload + b"\n")


def tiny_config(**overrides):
    values = {
        "seed": 3,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 2,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 5.0,
        "tune_scene_fraction": 0.25,
        "minimum_improvement": 1e-6,
        "advantage_alpha": 0.2,
        "risk_alpha": 0.2,
        "target_harm_upper": 0.5,
        "target_coverage_miss_upper": 0.5,
        "minimum_advantage_lcb_m": 0.0,
        "minimum_advantage_calibration_scenes": 1,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingConfig(**values)


def invalid_residual_record(*, scene="audit-invalid"):
    value = synthetic_record(
        scene=scene,
        role="development",
        episode=f"{scene}/episode-0",
        positive=True,
    )
    labels = value["candidates"][1]["labels"]
    labels.update({
        "geodesic_progress_h8_m": 0.0,
        "geodesic_progress_h24_m": 0.0,
        "advantage_h24_m": 0.0,
        "harm": False,
        "useful": False,
        "reachable": False,
        "collision_h8": False,
        "regression_h24": False,
        "rollout_label_valid": False,
    })
    value["set_labels"].update({
        "candidate_set_has_positive": False,
        "candidate_universe_has_positive": False,
        "candidate_coverage_miss": False,
        "coverage_label_valid": False,
        "oracle_best_candidate_id": "dustbin",
    })
    validate_candidate_set(value)
    return value


class FixedResidualModel(torch.nn.Module):
    """A deterministic model whose residual always passes model-only gates."""

    def __init__(
        self, *, residual_mean=1.0, residual_rank=2.0,
        residual_harm_logit=-10.0, coverage_logit=-10.0,
    ):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.residual_mean = float(residual_mean)
        self.residual_rank = float(residual_rank)
        self.residual_harm_logit = float(residual_harm_logit)
        self.coverage_logit = float(coverage_logit)

    def forward_batch(self, batch):
        shape = batch.valid_mask.shape
        device = batch.candidate_features.device
        mean = torch.zeros(shape, device=device)
        log_scale = torch.full(shape, -4.0, device=device)
        harm = torch.full(shape, -10.0, device=device)
        rank = torch.zeros(shape, device=device)
        mean = mean.masked_fill(batch.residual_mask, self.residual_mean)
        rank = rank.masked_fill(batch.residual_mask, self.residual_rank)
        harm = harm.masked_fill(batch.residual_mask, self.residual_harm_logit)
        return NLSRRankerOutput(
            advantage_mean=mean,
            advantage_log_scale=log_scale,
            harm_logit=harm,
            rank_score=rank,
            coverage_logit=torch.full(
                (shape[0],), self.coverage_logit, device=device),
            valid_mask=batch.valid_mask,
            native_mask=batch.native_mask,
            dustbin_mask=batch.dustbin_mask,
            residual_mask=batch.residual_mask,
            selectable_mask=batch.selectable_mask,
        )


def permissive_calibration():
    return {
        "advantage": {"one_sided_normalized_quantile": 0.0},
        "harm": {"threshold": 0.5},
        "coverage_miss_abstain": {"threshold": 0.5},
        "coverage_policy": COVERAGE_POLICY_REQUIRED,
    }


class CanonicalArtifactTest(unittest.TestCase):
    def test_canonical_json_and_jsonl_round_trip(self):
        rows = candidate_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "candidate_sets.json"
            jsonl_path = root / "candidate_sets.jsonl"
            write_artifact(json_path, rows)
            write_artifact(jsonl_path, rows, jsonl=True)
            json_rows, json_audit = load_canonical_candidate_rows(json_path)
            jsonl_rows, jsonl_audit = load_canonical_candidate_rows(jsonl_path)
        self.assertEqual(json_rows, rows)
        self.assertEqual(jsonl_rows, rows)
        self.assertEqual(
            json_audit["dataset_content_sha256"],
            jsonl_audit["dataset_content_sha256"])
        self.assertNotEqual(json_audit["file_sha256"], jsonl_audit["file_sha256"])

    def test_noncanonical_or_blank_jsonl_fails_closed(self):
        rows = candidate_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate_sets.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(NLSRTrainingError, "not canonical"):
                load_canonical_candidate_rows(path)
            jsonl = Path(directory) / "candidate_sets.jsonl"
            jsonl.write_bytes(
                canonical_json_bytes(rows[0]) + b"\n\n"
                + canonical_json_bytes(rows[1]))
            with self.assertRaisesRegex(NLSRTrainingError, "blank"):
                load_canonical_candidate_rows(jsonl)

    def test_class_weights_ignore_development_labels(self):
        rows = candidate_dataset()
        train = [
            row for row in rows
            if row["provenance"]["split_role"] == "train"]
        config, report = class_weights_from_train(train)
        changed = copy.deepcopy(rows)
        changed[-1] = synthetic_record(
            scene="development-b", role="development",
            episode="development-b/episode-0", positive=True)
        changed_train = [
            row for row in changed
            if row["provenance"]["split_role"] == "train"]
        changed_config, changed_report = class_weights_from_train(changed_train)
        self.assertEqual(config, changed_config)
        self.assertEqual(report, changed_report)
        self.assertEqual(report["source_split_role"], "train")
        self.assertEqual(report["harm"]["positive"], 1)
        self.assertEqual(report["coverage_miss"]["positive"], 1)


class CalibrationPrimitiveTest(unittest.TestCase):
    def test_finite_sample_lcb_uses_conservative_rank(self):
        quantile, rank = finite_sample_lcb_quantile(
            [-1.0, 0.0, 0.5, 1.0], alpha=0.2)
        self.assertEqual(rank, 4)
        self.assertEqual(quantile, 1.0)

    def test_grouped_zero_bad_threshold_and_bound(self):
        result = calibrate_zero_bad_threshold(
            [0.1, 0.2, 0.8],
            [False, False, True],
            [True, True, True],
            ["group-a", "group-b", "group-c"],
            alpha=0.2,
            target_upper=0.6,
        )
        self.assertEqual(result["threshold"], 0.2)
        self.assertEqual(result["activated_rows"], 2)
        self.assertEqual(result["bad_units"], 0)
        self.assertTrue(result["statistically_supported"])
        self.assertAlmostEqual(
            result["one_sided_upper"], zero_failure_upper_bound(2, 0.2))


class DeploymentDecisionIsolationTest(unittest.TestCase):
    def test_gt_validity_cannot_change_structural_decision(self):
        valid = synthetic_record(
            scene="audit-valid", role="development",
            episode="audit-valid/episode-0", positive=True)
        invalid = invalid_residual_record()
        spec = feature_spec_from_dataset([valid, invalid])
        valid_batch = vectorize_candidate_sets([valid], feature_spec=spec)
        invalid_batch = vectorize_candidate_sets([invalid], feature_spec=spec)
        model = FixedResidualModel()
        valid_output = model.forward_batch(valid_batch)
        invalid_output = model.forward_batch(invalid_batch)
        valid_decision = structural_policy_decisions(
            valid_output, valid_batch, permissive_calibration(), tiny_config())
        invalid_decision = structural_policy_decisions(
            invalid_output, invalid_batch,
            permissive_calibration(), tiny_config())
        self.assertEqual(
            valid_decision[0]["selected_candidate_id"], "frontier-1")
        self.assertEqual(
            invalid_decision[0]["selected_candidate_id"], "frontier-1")
        self.assertEqual(
            valid_decision[0]["reason"], invalid_decision[0]["reason"])

    def test_selected_invalid_rollout_is_unevaluable_not_rewritten(self):
        row = invalid_residual_record()
        spec = feature_spec_from_dataset([row])
        metrics = evaluate_calibrated_audit(
            FixedResidualModel(), [row], spec, tiny_config(),
            permissive_calibration())
        policy = metrics["calibrated_policy"]
        self.assertEqual(policy["activations"], 1)
        self.assertEqual(policy["evaluable_activations"], 0)
        self.assertEqual(policy["unevaluable_activations"], 1)
        trace = policy["decision_trace"][0]
        self.assertEqual(trace["selected_candidate_id"], "frontier-1")
        self.assertFalse(trace["label_evaluable"])

    def test_advisory_unavailable_skips_only_the_coverage_gate(self):
        row = invalid_residual_record()
        spec = feature_spec_from_dataset([row])
        batch = vectorize_candidate_sets([row], feature_spec=spec)
        calibration = permissive_calibration()
        calibration["coverage_policy"] = COVERAGE_POLICY_ADVISORY_UNAVAILABLE
        calibration["coverage_miss_abstain"]["threshold"] = -1.0

        high_coverage_risk = FixedResidualModel(coverage_logit=10.0)
        decision = structural_policy_decisions(
            high_coverage_risk.forward_batch(batch), batch, calibration,
            tiny_config(),
        )[0]
        self.assertEqual(decision["selected_candidate_id"], "frontier-1")
        self.assertEqual(
            decision["coverage_policy"], COVERAGE_POLICY_ADVISORY_UNAVAILABLE
        )

        required = copy.deepcopy(calibration)
        required["coverage_policy"] = COVERAGE_POLICY_REQUIRED
        required["coverage_miss_abstain"]["threshold"] = 0.5
        blocked = structural_policy_decisions(
            high_coverage_risk.forward_batch(batch), batch, required,
            tiny_config(),
        )[0]
        self.assertEqual(blocked["selected_candidate_id"], "native")
        self.assertEqual(blocked["reason"], "coverage_risk_abstain")

    def test_advisory_unavailable_keeps_advantage_harm_and_rank_fail_closed(self):
        row = invalid_residual_record()
        spec = feature_spec_from_dataset([row])
        batch = vectorize_candidate_sets([row], feature_spec=spec)
        calibration = permissive_calibration()
        calibration["coverage_policy"] = COVERAGE_POLICY_ADVISORY_UNAVAILABLE
        calibration["coverage_miss_abstain"]["threshold"] = -1.0
        models = (
            FixedResidualModel(residual_mean=-1.0, coverage_logit=10.0),
            FixedResidualModel(residual_harm_logit=10.0, coverage_logit=10.0),
            FixedResidualModel(residual_rank=-1.0, coverage_logit=10.0),
        )
        for model in models:
            with self.subTest(model=model):
                decision = structural_policy_decisions(
                    model.forward_batch(batch), batch, calibration,
                    tiny_config(),
                )[0]
                self.assertEqual(decision["selected_candidate_id"], "native")
                self.assertEqual(
                    decision["reason"], "no_structurally_eligible_residual"
                )

    def test_absent_or_unknown_coverage_policy_fails_closed(self):
        row = invalid_residual_record()
        spec = feature_spec_from_dataset([row])
        batch = vectorize_candidate_sets([row], feature_spec=spec)
        output = FixedResidualModel().forward_batch(batch)
        for policy in (None, "optional"):
            calibration = permissive_calibration()
            if policy is None:
                calibration.pop("coverage_policy")
            else:
                calibration["coverage_policy"] = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(
                NLSRTrainingError, "coverage_policy"
            ):
                structural_policy_decisions(
                    output, batch, calibration, tiny_config()
                )

    def test_nonfinite_model_output_cannot_bypass_advisory_coverage(self):
        row = invalid_residual_record()
        spec = feature_spec_from_dataset([row])
        batch = vectorize_candidate_sets([row], feature_spec=spec)
        calibration = permissive_calibration()
        calibration["coverage_policy"] = COVERAGE_POLICY_ADVISORY_UNAVAILABLE
        output = FixedResidualModel(coverage_logit=10.0).forward_batch(batch)
        output.rank_score[0, 1] = float("nan")
        with self.assertRaisesRegex(NLSRTrainingError, "non-finite rank score"):
            structural_policy_decisions(output, batch, calibration, tiny_config())

    def test_missing_coverage_labels_calibrate_to_advisory_not_deployment(self):
        rows = [
            invalid_residual_record(scene="calibration-a"),
            invalid_residual_record(scene="calibration-b"),
        ]
        spec = feature_spec_from_dataset(rows)
        calibration = fit_development_calibration(
            FixedResidualModel(), rows, spec, tiny_config()
        )
        self.assertEqual(
            calibration["coverage_policy"],
            COVERAGE_POLICY_ADVISORY_UNAVAILABLE,
        )
        self.assertEqual(calibration["coverage_miss_abstain"]["valid_rows"], 0)
        self.assertFalse(calibration["deployment_approved"])


class EvaluationAggregationTest(unittest.TestCase):
    def test_component_aggregation_is_batch_size_invariant(self):
        rows = [
            row for row in candidate_dataset()
            if row["provenance"]["split_role"] == "train"]
        spec = feature_spec_from_dataset(rows)
        packed = vectorize_candidate_sets(rows, feature_spec=spec)
        model = NLSRSetRanker(spec, NLSRRankerConfig(init_seed=11))
        loss_config = NLSRLossConfig(
            harm_positive_weight=3.0,
            coverage_positive_weight=2.0,
        )
        reference = evaluate_loss(
            model, packed, loss_config, 1, torch.device("cpu"))
        for batch_size in (2, 4, 99):
            actual = evaluate_loss(
                model, packed, loss_config, batch_size,
                torch.device("cpu"))
            for key in reference:
                self.assertAlmostEqual(
                    actual[key], reference[key], places=6,
                    msg=f"component {key} changed at batch_size={batch_size}")


class TrainingWorkflowTest(unittest.TestCase):
    def test_preflight_and_complete_outputs_are_content_bound(self):
        rows = candidate_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate_sets.json"
            output = root / "output"
            write_artifact(artifact, rows)
            preflight = run_training(
                artifact, output, tiny_config(), preflight_only=True)
            self.assertEqual(preflight.status, "preflight_passed")
            self.assertEqual(preflight.report["records"]["train"], 4)
            self.assertEqual(preflight.report["records"]["development"], 2)
            self.assertTrue(set(preflight.report["scenes"]["core_scenes"])
                            .isdisjoint(
                                preflight.report["scenes"]["tune_scenes"]))

            result = run_training(artifact, output, tiny_config())
            self.assertEqual(result.status, "training_complete")
            for name in (
                    FINAL_CHECKPOINT_NAME, METRICS_NAME, CALIBRATION_NAME,
                    PROVENANCE_NAME, MANIFEST_NAME):
                self.assertTrue((output / name).is_file())
            manifest = json.loads((output / MANIFEST_NAME).read_text())
            for name, descriptor in manifest["files"].items():
                self.assertEqual(
                    descriptor["sha256"], sha256_file(output / name))
            self.assertEqual(
                result.report["metrics_sha256"], sha256_file(output / METRICS_NAME))
            payload = torch.load(
                output / FINAL_CHECKPOINT_NAME,
                map_location="cpu", weights_only=True)
            restored = load_portable_checkpoint(
                payload, expected_records=rows)
            self.assertEqual(
                state_dict_sha256(restored.state_dict()),
                result.report["checkpoint_state_dict_sha256"])
            provenance = json.loads((output / PROVENANCE_NAME).read_text())
            self.assertTrue(
                provenance["development_used_only_after_epoch_freeze"])
            self.assertTrue(
                provenance["class_weights_derived_only_from_train"])
            self.assertTrue(result.report["bundle_readback_verified"])
            calibration = json.loads(
                (output / CALIBRATION_NAME).read_text())
            self.assertTrue(set(calibration["calibration_scenes"])
                            .isdisjoint(calibration["audit_scenes"]))
            self.assertFalse(calibration["deployment_approved"])
            self.assertFalse(result.report["deployment_approved"])

    def test_interrupted_resume_matches_uninterrupted_weights(self):
        rows = candidate_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate_sets.json"
            write_artifact(artifact, rows)
            resumed_output = root / "resumed"
            full_output = root / "full"
            partial = run_training(
                artifact, resumed_output, tiny_config(), epoch_budget=1)
            self.assertEqual(partial.status, "training_incomplete")
            resumed = run_training(
                artifact, resumed_output, tiny_config(), resume=True)
            full = run_training(artifact, full_output, tiny_config())
            self.assertEqual(resumed.status, "training_complete")
            self.assertEqual(full.status, "training_complete")
            self.assertEqual(
                resumed.report["checkpoint_state_dict_sha256"],
                full.report["checkpoint_state_dict_sha256"])
            self.assertEqual(
                (resumed_output / METRICS_NAME).read_bytes(),
                (full_output / METRICS_NAME).read_bytes())
            self.assertEqual(
                (resumed_output / CALIBRATION_NAME).read_bytes(),
                (full_output / CALIBRATION_NAME).read_bytes())
            self.assertEqual(
                (resumed_output / FINAL_CHECKPOINT_NAME).read_bytes(),
                (full_output / FINAL_CHECKPOINT_NAME).read_bytes())

    def test_resume_rejects_feature_label_or_config_drift(self):
        rows = candidate_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate_sets.json"
            output = root / "output"
            write_artifact(artifact, rows)
            partial = run_training(
                artifact, output, tiny_config(), epoch_budget=1)
            self.assertEqual(partial.status, "training_incomplete")

            changed = copy.deepcopy(rows)
            changed[0]["candidates"][1]["features"][
                "subgoal_forward_m"] += 0.125
            write_artifact(artifact, changed)
            with self.assertRaisesRegex(NLSRTrainingError, "contract drifted"):
                run_training(
                    artifact, output, tiny_config(), resume=True)

            write_artifact(artifact, rows)
            with self.assertRaisesRegex(NLSRTrainingError, "contract drifted"):
                run_training(
                    artifact, output,
                    tiny_config(learning_rate=1e-3), resume=True)

            state_path = output / RESUME_NAME
            state = torch.load(
                state_path, map_location="cpu", weights_only=True)
            state["optimizer_state"]["param_groups"][0]["lr"] = 0.123
            torch.save(state, state_path)
            with self.assertRaisesRegex(NLSRTrainingError, "payload hash"):
                run_training(artifact, output, tiny_config(), resume=True)

    def test_resume_rejects_self_consistent_phase_history_corruption(self):
        rows = candidate_dataset()
        corruptions = {
            "next_epoch": lambda state: state.update({"next_epoch": 0}),
            "history_epoch": lambda state: state["selection_history"][0]
            .update({"epoch": 2}),
            "best_epoch": lambda state: state.update({"best_epoch": -1}),
            "best_state_epoch_binding": lambda state: state[
                "selection_history"][0].update({
                    "model_state_sha256": "0" * 64}),
            "premature_complete": lambda state: state.update({
                "phase": "complete", "selected_epochs": 1,
            }),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact = root / "candidate_sets.json"
                output = root / "output"
                write_artifact(artifact, rows)
                partial = run_training(
                    artifact, output, tiny_config(), epoch_budget=1)
                self.assertEqual(partial.status, "training_incomplete")
                state_path = output / RESUME_NAME
                state = torch.load(
                    state_path, map_location="cpu", weights_only=True)
                corrupt(state)
                state["resume_payload_sha256"] = resume_payload_sha256(state)
                atomic_torch_save(state_path, state)
                with self.assertRaises(NLSRTrainingError):
                    run_training(
                        artifact, output, tiny_config(), resume=True)

    def test_selection_and_refit_class_weights_have_distinct_sources(self):
        rows = candidate_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate_sets.json"
            write_artifact(artifact, rows)
            first = run_training(
                artifact, root / "unused", tiny_config(),
                preflight_only=True).report
            tune_scene = first["scenes"]["tune_scenes"][0]
            changed = copy.deepcopy(rows)
            index = next(
                index for index, row in enumerate(changed)
                if row["provenance"]["scene_id"] == tune_scene)
            changed[index] = harmful_record(scene=tune_scene, role="train")
            validate_candidate_dataset(changed)
            write_artifact(artifact, changed)
            second = run_training(
                artifact, root / "unused", tiny_config(),
                preflight_only=True).report
        self.assertEqual(
            first["class_balance"]["selection_core"],
            second["class_balance"]["selection_core"])
        self.assertNotEqual(
            first["class_balance"]["refit_all_train"],
            second["class_balance"]["refit_all_train"])
        self.assertNotEqual(
            first["training_contract_sha256"],
            second["training_contract_sha256"])

    def test_single_development_scene_fails_closed(self):
        rows = [
            row for row in candidate_dataset()
            if row["provenance"]["scene_id"] != "development-b"]
        validate_candidate_dataset(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate_sets.json"
            output = root / "output"
            write_artifact(artifact, rows)
            result = run_training(artifact, output, tiny_config())
            calibration = json.loads(
                (output / CALIBRATION_NAME).read_text())
            metrics = json.loads((output / METRICS_NAME).read_text())
        self.assertEqual(result.status, "training_complete")
        self.assertFalse(calibration["scene_disjoint_audit"])
        self.assertFalse(calibration["shadow_eligible"])
        self.assertEqual(calibration["harm"]["threshold"], -1.0)
        self.assertEqual(
            metrics["development"]["calibrated_policy"]["activations"], 0)

    def test_bundle_verifier_rejects_post_write_tampering(self):
        rows = candidate_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate_sets.json"
            output = root / "output"
            config = tiny_config()
            write_artifact(artifact, rows)
            result = run_training(artifact, output, config)
            verify_final_bundle(
                output, rows, result.report["training_contract_sha256"],
                config)
            metrics_path = output / METRICS_NAME
            metrics_path.write_bytes(metrics_path.read_bytes() + b" ")
            with self.assertRaisesRegex(NLSRTrainingError, "hash mismatch"):
                verify_final_bundle(
                    output, rows,
                    result.report["training_contract_sha256"], config)

    def test_fixed_torch_serialization_is_byte_deterministic(self):
        payload = {
            "state": {"weight": torch.arange(8, dtype=torch.float32)},
            "epoch": 3,
        }
        self.assertEqual(
            deterministic_torch_bytes(payload),
            deterministic_torch_bytes(copy.deepcopy(payload)))


if __name__ == "__main__":
    unittest.main()
