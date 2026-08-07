import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from MemNavData.build_memory_graph_candidate_artifact import (
    FIRST_SUBGOAL_SEMANTICS,
    REVERSE_ROUTE_PURPOSE,
    REVERSE_ROUTE_SCHEMA_VERSION,
    REVERSE_ROUTE_SEMANTICS,
    MemoryGraphCandidateBuildError,
    ReverseRoutePins,
    build_memory_graph_candidate_artifact,
    write_artifact,
)
from MemNavData.phase_b_deployment_inference_contract import (
    CANDIDATE_SELECTION_SEMANTICS,
    CHECKPOINT_MODEL_KIND,
    DEPLOYMENT_APPROVAL_KEYS,
    DEPLOYMENT_APPROVAL_PURPOSE,
    DEPLOYMENT_APPROVAL_SCHEMA_VERSION,
    ENSEMBLE_SEMANTICS,
    InferencePins,
    METRIC_SCALE_SOURCE,
    PHASE_B_CHECKPOINT_SCHEMA_VERSION,
    PHASE_B_FEATURE_NAMES,
    PHASE_B_FEATURE_SCHEMA_VERSION,
    POSE_CONVENTION,
    PRIVILEGED_INPUT_POLICY,
    PURPOSE,
    PhaseBInferenceContractError,
    SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_phase_b_deployment_inference,
)
from MemNavData.external_causal_scale_contract import (
    CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
)
from MemNavData.phase_b_feature_schema import (
    FEATURE_DIMENSION,
    FEATURE_NAMES_SHA256,
    METRIC_SCALE_SOURCES,
)
from MemNavData.train_lingbot_native_localizer import LingBotNativeLocalizer


FLOW_SHA = "1" * 64
INFERENCE_PRODUCER_SHA = "5" * 64
DINO_CHECKPOINT_SHA = "6" * 64
DINO_PRODUCER_SHA = "7" * 64
LINGBOT_COMMIT = "8" * 40
LINGBOT_WEIGHTS_SHA = "9" * 64
LINGBOT_STREAM_SHA = "a" * 64
REVERSE_PRODUCER_SHA = "b" * 64


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def sample(
    *,
    sample_id,
    role,
    decision,
    prefix_sha,
    fifo_sha,
    goal_sha,
):
    return {
        "sample_id": sample_id,
        "split_role": "train",
        "scene": "scene",
        "state_source": "expert",
        "source_episode": "episode_0000",
        "source_episode_id": "scene/episode_0000",
        "goal_episode": "episode_0000",
        "goal_source_episode_id": "scene/episode_0000",
        "goal_variant": "factual",
        "goal_role": role,
        "state_name": "goal_b_t0" if role == "B" else "goal_c_t2",
        "decision_frame": decision,
        "state_frame": {"content_sha256": "c" * 64},
        "causal_prefix": {
            "frame_count": decision,
            "causal_prefix_sha256": prefix_sha,
        },
        "navdp_fifo": {"fifo_sha256": fifo_sha},
        "goal": {"content_sha256": goal_sha},
    }


def manifest_fixture():
    return {
        "schema_version": "nlsr_v2_multistage_expert_candidate_manifest_v1",
        "flow_cache_routing": {
            "mode": "provenance_pinned_multi_root",
            "artifact_sha256": FLOW_SHA,
        },
        "samples": [
            sample(
                sample_id="train/scene/episode_0000/goal_b_t0/factual",
                role="B",
                decision=16,
                prefix_sha="d" * 64,
                fifo_sha="e" * 64,
                goal_sha="f" * 64,
            ),
            sample(
                sample_id="train/scene/episode_0000/goal_c_t2/factual",
                role="C",
                decision=32,
                prefix_sha="0" * 64,
                fifo_sha="1" * 64,
                goal_sha="2" * 64,
            ),
        ],
    }


def inference_configuration():
    return {
        "candidate_shortlist_size": 2,
        "temporal_minimum_gap_frames": 4,
        "neighbor_offsets": [-1, 0, 1],
        "match_threshold": 0.5,
        "ensemble_member_count": 3,
        "candidate_selection_semantics": CANDIDATE_SELECTION_SEMANTICS,
        "ensemble_semantics": ENSEMBLE_SEMANTICS,
    }


def physical_inputs(root: Path, manifest):
    configuration_sha = digest(inference_configuration())
    sample_ids = sorted(row["sample_id"] for row in manifest["samples"])
    decisions = sorted({int(row["decision_frame"])
                        for row in manifest["samples"]})
    scale_record = {
        "scene": "scene",
        "episode": "episode_0000",
        "split_role": "train",
        "sample_ids": sample_ids,
        "decision_frames": decisions,
        "earliest_decision_frame": decisions[0],
        "prefix_end_frame_exclusive": 8,
        "camera_height_m": 0.5,
        "episode_frame_count": 64,
        "rgb_prefix": {
            "frame_count": 8,
            "path_sequence_sha256": "2" * 64,
            "content_sequence_sha256": "3" * 64,
        },
        "cam_pose_prefix_sha256": "4" * 64,
        "cam_pose_prefix_dtype": "<f4",
        "cache_schema_version": 2,
        "precompute_signature": "test-precompute-v1",
        "valid": True,
        "invalid_reason": None,
        "ground_h_est_raw": 0.25,
        "metric_scale_m_per_raw": 2.0,
        "debug": {
            "n_points": 128,
            "n_frames": 8,
            "n_valid": 6,
            "h_est": 0.25,
            "h_iqr": 0.025,
        },
    }
    scale_configuration = {
        "prefix_frame_cap": 64,
        "num_scale_frames": 8,
        "bias_correction": 1.0,
        "scale_min": 0.8,
        "scale_max": 6.0,
    }
    scale_artifact = {
        "schema_version": CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
        "purpose": "test physical causal scale",
        "provenance": {
            "input_manifest_sha256": digest(manifest),
            "input_manifest_schema_version": manifest["schema_version"],
            "producer_source_sha256": "2" * 64,
            "configuration_sha256": digest(scale_configuration),
            "estimator": {
                "kind": "frozen_lingbot_compute_metric_scale_prefix",
                "lingbot_commit": LINGBOT_COMMIT,
                "weights_sha256": LINGBOT_WEIGHTS_SHA,
                "lingbot_stream_source_sha256": LINGBOT_STREAM_SHA,
            },
        },
        "configuration": scale_configuration,
        "records": [scale_record],
        "summary": {"future_frames_consumed": 0},
    }
    scale_path = root / "causal_scale.json"
    scale_path.write_bytes(canonical_json_bytes(scale_artifact))
    scale_sha = sha256_file(scale_path)

    model = LingBotNativeLocalizer(FEATURE_DIMENSION, hidden_dim=8, dropout=0.0)
    scale_coverage_common = {
        "approved": True,
        "exact_manifest_sample_coverage_approved": True,
        "manifest_sha256": [digest(manifest)],
        "scale_artifact_sha256": [scale_sha],
        "producer_source_sha256": "2" * 64,
        "configuration_sha256": digest(scale_configuration),
        "lingbot_commit": LINGBOT_COMMIT,
        "weights_sha256": LINGBOT_WEIGHTS_SHA,
        "stream_source_sha256": LINGBOT_STREAM_SHA,
        "manifest_schema_version": manifest["schema_version"],
        "source": METRIC_SCALE_SOURCE,
    }
    checkpoint = {
        "checkpoint_schema_version": PHASE_B_CHECKPOINT_SCHEMA_VERSION,
        "deployment_approved": False,
        "deployment_input_contract_approved": True,
        "model_kind": CHECKPOINT_MODEL_KIND,
        "input_dim": FEATURE_DIMENSION,
        "feature_schema_version": PHASE_B_FEATURE_SCHEMA_VERSION,
        "feature_names": list(PHASE_B_FEATURE_NAMES),
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "metric_scale_source_categories": list(METRIC_SCALE_SOURCES),
        "normalization_mean": [0.0] * FEATURE_DIMENSION,
        "normalization_scale": [1.0] * FEATURE_DIMENSION,
        "config": {
            "hidden_dim": 8,
            "dropout": 0.0,
            "match_threshold": 0.5,
            "pose_gain": 1.0,
            "positive_threshold": 0.5,
        },
        "states": [copy.deepcopy(model.state_dict()) for _ in range(3)],
        "external_causal_scale_coverage": {
            "approved": True,
            "train_exact_coverage_approved": True,
            "development_exact_coverage_approved": True,
            "train": {
                **scale_coverage_common,
                "split_roles": ["train"],
            },
            "development": {
                **scale_coverage_common,
                "split_roles": ["development"],
            },
        },
        "train_artifact_identity_sha256": "c" * 64,
        "development_artifact_identity_sha256": "d" * 64,
        "train_audit_sha256": "a" * 64,
        "development_audit_sha256": "b" * 64,
    }
    checkpoint_path = root / "phase_b.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    approval = {
        "schema_version": DEPLOYMENT_APPROVAL_SCHEMA_VERSION,
        "purpose": DEPLOYMENT_APPROVAL_PURPOSE,
        "deployment_approved": True,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_schema_version": PHASE_B_CHECKPOINT_SCHEMA_VERSION,
        "model_kind": CHECKPOINT_MODEL_KIND,
        "feature_schema_version": PHASE_B_FEATURE_SCHEMA_VERSION,
        "input_dim": FEATURE_DIMENSION,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "train_artifact_identity_sha256": "c" * 64,
        "development_artifact_identity_sha256": "d" * 64,
        "train_audit_sha256": "a" * 64,
        "development_audit_sha256": "b" * 64,
        "deployment_input_contract_approved": True,
        "train_exact_coverage_approved": True,
        "development_exact_coverage_approved": True,
        "inference_configuration_sha256": configuration_sha,
        "closed_loop_evidence_artifact_sha256": "e" * 64,
    }
    assert frozenset(approval) == DEPLOYMENT_APPROVAL_KEYS
    approval_path = root / "deployment_approval.json"
    approval_path.write_bytes(canonical_json_bytes(approval))
    approval_sha = sha256_file(approval_path)
    pins = InferencePins(
        flow_route_artifact_sha256=FLOW_SHA,
        causal_scale_artifact_path=scale_path,
        causal_scale_artifact_sha256=scale_sha,
        phase_b_checkpoint_path=checkpoint_path,
        phase_b_checkpoint_sha256=checkpoint_sha,
        deployment_approval_artifact_path=approval_path,
        deployment_approval_artifact_sha256=approval_sha,
        inference_producer_source_sha256=INFERENCE_PRODUCER_SHA,
        inference_configuration_sha256=configuration_sha,
        dino_encoder_checkpoint_sha256=DINO_CHECKPOINT_SHA,
        dino_feature_producer_sha256=DINO_PRODUCER_SHA,
        lingbot_commit=LINGBOT_COMMIT,
        lingbot_weights_sha256=LINGBOT_WEIGHTS_SHA,
        lingbot_stream_source_sha256=LINGBOT_STREAM_SHA,
    )
    return pins, scale_record


def candidate(*, rank, anchor, scale, validity, usable, dino=0.8):
    raw_xz = [0.25, 1.5]
    scaled_xz = [scale * raw_xz[0], scale * raw_xz[1]]
    raw_forward_left = [scaled_xz[1], -scaled_xz[0]]
    # Exact LingBot confidence scores are positive but not probabilities.
    goal_confidence = 4.0
    candidate_confidence = 5.0
    features = [
        dino,
        scale,
        1.0,
        0.7,
        0.5,
        0.1,
        2.0,
        goal_confidence,
        candidate_confidence,
        raw_forward_left[0],
        raw_forward_left[1],
        math.hypot(*raw_forward_left),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.75,
        0.1,
        0.0,
    ]
    return {
        "candidate_rank": rank,
        "anchor_frame_index": anchor,
        "goal_append_prefix_end_frame_exclusive": anchor + 1,
        "goal_append_input_sha256": hex(anchor % 16)[2:] * 64,
        "dino_cosine": dino,
        "phase_b_input_features": features,
        "rank_probability": 1.0,
        "candidate_validity_probability": validity,
        "set_probability": usable,
        "native_relative_translation_xz_raw": raw_xz,
        "scaled_relative_translation_xz_m": scaled_xz,
        "raw_relative_forward_left_m": raw_forward_left,
        "corrected_relative_forward_left_m": [
            raw_forward_left[0] + 0.1,
            raw_forward_left[1] + 0.1,
        ],
        "translation_variance_forward_left_m2": [0.04, 0.09],
        "goal_depth_confidence_mean": goal_confidence,
        "candidate_depth_confidence_mean": candidate_confidence,
    }


def inference_record(
    sample_row, *, anchor, global_no_match, validity, selected, scale_record,
):
    usable = (1.0 - global_no_match) * validity
    scale = 2.0
    return {
        "sample_id": sample_row["sample_id"],
        "scene": sample_row["scene"],
        "source_episode_id": sample_row["source_episode_id"],
        "goal_role": sample_row["goal_role"],
        "decision_frame": sample_row["decision_frame"],
        "causal_prefix_sha256": sample_row["causal_prefix"][
            "causal_prefix_sha256"
        ],
        "navdp_fifo_sha256": sample_row["navdp_fifo"]["fifo_sha256"],
        "goal_sha256": sample_row["goal"]["content_sha256"],
        "cam_pose_prefix_sha256": "3" * 64 if sample_row["goal_role"] == "B" else "4" * 64,
        "cam_pose_prefix_frame_count": sample_row["decision_frame"],
        "metric_scale_m_per_raw": scale,
        "metric_scale_source": METRIC_SCALE_SOURCE,
        "external_scale_record_sha256": digest(scale_record),
        "external_scale_prefix_end_frame_exclusive": scale_record[
            "prefix_end_frame_exclusive"],
        "external_scale_cam_pose_prefix_sha256": scale_record[
            "cam_pose_prefix_sha256"],
        "external_scale_rgb_prefix_content_sequence_sha256": scale_record[
            "rgb_prefix"]["content_sequence_sha256"],
        "global_no_match_probability": global_no_match,
        "usable_match_probability": usable,
        "dustbin_probability": 1.0 - usable,
        "selected_anchor_frame_index": selected,
        "candidates": [
            candidate(
                rank=0,
                anchor=anchor,
                scale=scale,
                validity=validity,
                usable=usable,
            )
        ],
    }


def inference_fixture(manifest, root):
    configuration = inference_configuration()
    pins, scale_record = physical_inputs(root, manifest)
    records = [
        inference_record(
            manifest["samples"][0],
            anchor=8,
            global_no_match=0.1,
            validity=0.8,
            selected=8,
            scale_record=scale_record,
        ),
        inference_record(
            manifest["samples"][1],
            anchor=20,
            global_no_match=0.5,
            validity=0.2,
            selected=None,
            scale_record=scale_record,
        ),
    ]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "provenance": {
            "input_manifest_sha256": digest(manifest),
            "input_manifest_schema_version": manifest["schema_version"],
            "flow_route_artifact_sha256": FLOW_SHA,
            "causal_scale_artifact_sha256": pins.causal_scale_artifact_sha256,
            "phase_b_checkpoint_sha256": pins.phase_b_checkpoint_sha256,
            "phase_b_checkpoint_model_kind": CHECKPOINT_MODEL_KIND,
            "phase_b_checkpoint_schema_version": (
                PHASE_B_CHECKPOINT_SCHEMA_VERSION),
            "phase_b_checkpoint_deployment_approved": True,
            "phase_b_checkpoint_deployment_input_contract_approved": True,
            "phase_b_external_causal_scale_coverage_approved": True,
            "phase_b_feature_schema_version": PHASE_B_FEATURE_SCHEMA_VERSION,
            "phase_b_feature_names": list(PHASE_B_FEATURE_NAMES),
            "phase_b_feature_names_sha256": digest(list(PHASE_B_FEATURE_NAMES)),
            "deployment_approval_artifact_sha256": (
                pins.deployment_approval_artifact_sha256),
            "inference_producer_source_sha256": INFERENCE_PRODUCER_SHA,
            "inference_configuration_sha256": pins.inference_configuration_sha256,
            "dino_encoder_checkpoint_sha256": DINO_CHECKPOINT_SHA,
            "dino_feature_producer_sha256": DINO_PRODUCER_SHA,
            "lingbot_commit": LINGBOT_COMMIT,
            "lingbot_weights_sha256": LINGBOT_WEIGHTS_SHA,
            "lingbot_stream_source_sha256": LINGBOT_STREAM_SHA,
            "pose_convention": POSE_CONVENTION,
            "privileged_input_policy": PRIVILEGED_INPUT_POLICY,
        },
        "configuration": configuration,
        "records": records,
        "summary": {
            "record_count": 2,
            "goal_b_record_count": 1,
            "goal_c_record_count": 1,
            "activated_match_record_count": 1,
            "rejected_no_match_record_count": 1,
            "candidate_count": 2,
            "future_frames_consumed": 0,
            "privileged_inputs_consumed": 0,
        },
    }
    return artifact, pins


def reverse_fixture(manifest, inference, inference_sha):
    configuration = {
        "spacing_m": 0.75,
        "route_semantics": REVERSE_ROUTE_SEMANTICS,
        "first_subgoal_semantics": FIRST_SUBGOAL_SEMANTICS,
    }
    configuration_sha = digest(configuration)
    source = inference["records"][0]
    # At yaw pi/2, native world delta [dx,dz]=[0.2,1.0] metric maps to
    # NavDP [forward,left]=[1.0,-0.2].
    record = {
        "sample_id": source["sample_id"],
        "causal_prefix_sha256": source["causal_prefix_sha256"],
        "decision_frame": source["decision_frame"],
        "cam_pose_prefix_sha256": source["cam_pose_prefix_sha256"],
        "anchor_frame_index": source["selected_anchor_frame_index"],
        "start_frame_index": source["decision_frame"] - 1,
        "metric_scale_m_per_raw": source["metric_scale_m_per_raw"],
        "metric_scale_source": METRIC_SCALE_SOURCE,
        "current_native_position_xz_raw": [0.0, 0.0],
        "current_native_yaw_rad": math.pi / 2.0,
        "route_nodes": [
            {
                "frame_index": 12,
                "native_position_xz_raw": [0.1, 0.5],
                "segment_path_m": 1.2,
            },
            {
                "frame_index": 8,
                "native_position_xz_raw": [0.3, 1.0],
                "segment_path_m": 1.1,
            },
        ],
        "first_subgoal_forward_left_m": [1.0, -0.2],
        "graph_path_m": 2.3,
    }
    artifact = {
        "schema_version": REVERSE_ROUTE_SCHEMA_VERSION,
        "purpose": REVERSE_ROUTE_PURPOSE,
        "provenance": {
            "input_manifest_sha256": digest(manifest),
            "input_inference_artifact_sha256": inference_sha,
            "flow_route_artifact_sha256": FLOW_SHA,
            "causal_scale_artifact_sha256": inference["provenance"][
                "causal_scale_artifact_sha256"],
            "producer_source_sha256": REVERSE_PRODUCER_SHA,
            "configuration_sha256": configuration_sha,
            "pose_convention": POSE_CONVENTION,
            "privileged_input_policy": PRIVILEGED_INPUT_POLICY,
        },
        "configuration": configuration,
        "records": [record],
        "summary": {
            "record_count": 1,
            "route_node_count": 2,
            "future_frames_consumed": 0,
            "privileged_inputs_consumed": 0,
        },
    }
    return artifact, ReverseRoutePins(REVERSE_PRODUCER_SHA, configuration_sha)


class PhaseBDeploymentInferenceContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = manifest_fixture()
        self.manifest_sha = digest(self.manifest)
        self.inference, self.pins = inference_fixture(self.manifest, self.root)
        self.inference_sha = digest(self.inference)

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, artifact=None):
        artifact = self.inference if artifact is None else artifact
        return validate_phase_b_deployment_inference(
            artifact=artifact,
            artifact_sha256=digest(artifact),
            manifest=self.manifest,
            manifest_sha256=self.manifest_sha,
            pins=self.pins,
        )

    def test_goal_b_and_goal_c_records_bind_exact_causal_prefixes(self):
        validated = self.validate()
        self.assertEqual(
            {row["goal_role"] for row in validated.records_by_sample.values()},
            {"B", "C"},
        )
        for sample_id, record in validated.records_by_sample.items():
            source = validated.manifest_samples[sample_id]
            self.assertEqual(
                record["cam_pose_prefix_frame_count"], source["decision_frame"]
            )
            self.assertLess(
                record["candidates"][0]["anchor_frame_index"],
                source["decision_frame"],
            )

    def test_goal_c_prefix_mismatch_fails_even_under_new_artifact_sha(self):
        value = copy.deepcopy(self.inference)
        value["records"][1]["causal_prefix_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            PhaseBInferenceContractError, "changed manifest field causal_prefix"
        ):
            self.validate(value)

    def test_future_anchor_and_goal_append_fail_closed(self):
        value = copy.deepcopy(self.inference)
        candidate_row = value["records"][0]["candidates"][0]
        candidate_row["anchor_frame_index"] = 16
        candidate_row["goal_append_prefix_end_frame_exclusive"] = 17
        with self.assertRaisesRegex(PhaseBInferenceContractError, "future anchor"):
            self.validate(value)

    def test_neighbor_offsets_must_all_precede_the_decision(self):
        value = copy.deepcopy(self.inference)
        candidate_row = value["records"][0]["candidates"][0]
        candidate_row["anchor_frame_index"] = 0
        candidate_row["goal_append_prefix_end_frame_exclusive"] = 1
        value["records"][0]["selected_anchor_frame_index"] = 0
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "neighbor offsets"):
            self.validate(value)

    def test_temporal_gap_is_enforced_between_selected_candidates(self):
        value = copy.deepcopy(self.inference)
        record = value["records"][0]
        record["candidates"].append(candidate(
            rank=1,
            anchor=10,
            scale=2.0,
            validity=0.8,
            usable=record["usable_match_probability"],
        ))
        value["summary"]["candidate_count"] += 1
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "temporal minimum gap"):
            self.validate(value)

    def test_xz_scale_and_forward_left_mapping_are_numerically_enforced(self):
        value = copy.deepcopy(self.inference)
        candidate_row = value["records"][0]["candidates"][0]
        candidate_row["raw_relative_forward_left_m"] = [0.5, 3.0]
        with self.assertRaisesRegex(PhaseBInferenceContractError, "forward axis"):
            self.validate(value)

    def test_external_scale_quality_features_bind_physical_record(self):
        value = copy.deepcopy(self.inference)
        value["records"][0]["candidates"][0][
            "phase_b_input_features"][17] = 0.5
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "valid-frame ratio"):
            self.validate(value)

    def test_wrong_or_noncausal_scale_source_is_rejected(self):
        value = copy.deepcopy(self.inference)
        value["records"][0]["metric_scale_source"] = "pooled_fallback"
        with self.assertRaisesRegex(PhaseBInferenceContractError, "external causal"):
            self.validate(value)

    def test_each_sample_is_bound_to_the_physical_scale_record(self):
        value = copy.deepcopy(self.inference)
        value["records"][0]["external_scale_record_sha256"] = "f" * 64
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "physical scale record"):
            self.validate(value)

    def test_unapproved_checkpoint_cannot_be_used(self):
        value = copy.deepcopy(self.inference)
        value["provenance"]["phase_b_checkpoint_deployment_approved"] = False
        with self.assertRaisesRegex(PhaseBInferenceContractError, "not deployment-approved"):
            self.validate(value)

    def test_physical_checkpoint_and_approval_bytes_are_authoritative(self):
        checkpoint_path = Path(self.pins.phase_b_checkpoint_path)
        checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "physical Phase-B checkpoint SHA"):
            self.validate()

    def test_checkpoint_scale_convention_must_equal_physical_inference_scale(self):
        checkpoint_path = Path(self.pins.phase_b_checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["external_causal_scale_coverage"]["train"][
            "configuration_sha256"] = "f" * 64
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)

        approval_path = Path(self.pins.deployment_approval_artifact_path)
        approval = json.loads(approval_path.read_text())
        approval["checkpoint_sha256"] = checkpoint_sha
        approval_path.write_bytes(canonical_json_bytes(approval))
        self.pins = replace(
            self.pins,
            phase_b_checkpoint_sha256=checkpoint_sha,
            deployment_approval_artifact_sha256=sha256_file(approval_path),
        )
        with self.assertRaisesRegex(
                PhaseBInferenceContractError,
                "train external-scale configuration_sha256"):
            self.validate()

    def test_held_out_scale_artifact_need_not_equal_training_artifact(self):
        checkpoint_path = Path(self.pins.phase_b_checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
        for role in ("train", "development"):
            coverage = checkpoint["external_causal_scale_coverage"][role]
            coverage["manifest_sha256"] = ["f" * 64]
            coverage["scale_artifact_sha256"] = ["0" * 64]
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)

        approval_path = Path(self.pins.deployment_approval_artifact_path)
        approval = json.loads(approval_path.read_text())
        approval["checkpoint_sha256"] = checkpoint_sha
        approval_path.write_bytes(canonical_json_bytes(approval))
        approval_sha = sha256_file(approval_path)
        self.pins = replace(
            self.pins,
            phase_b_checkpoint_sha256=checkpoint_sha,
            deployment_approval_artifact_sha256=approval_sha,
        )
        self.inference["provenance"][
            "phase_b_checkpoint_sha256"] = checkpoint_sha
        self.inference["provenance"][
            "deployment_approval_artifact_sha256"] = approval_sha
        validated = self.validate()
        self.assertEqual(len(validated.records_by_sample), 2)

    def test_approval_cannot_be_rebound_or_edited_under_old_pin(self):
        approval_path = Path(self.pins.deployment_approval_artifact_path)
        approval = json.loads(approval_path.read_text())
        approval["checkpoint_sha256"] = "f" * 64
        approval_path.write_bytes(canonical_json_bytes(approval))
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "pinned JSON SHA256 changed"):
            self.validate()

    def test_legacy_sixteen_dimensional_checkpoint_fails_closed(self):
        value = copy.deepcopy(self.inference)
        value["provenance"]["phase_b_checkpoint_schema_version"] = 1
        value["provenance"]["phase_b_feature_schema_version"] = "legacy_v1"
        value["provenance"]["phase_b_feature_names"] = list(
            PHASE_B_FEATURE_NAMES[:-1])
        value["provenance"]["phase_b_feature_names_sha256"] = digest(
            value["provenance"]["phase_b_feature_names"])
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "checkpoint_schema_version"):
            self.validate(value)

    def test_checkpoint_without_external_train_dev_coverage_fails_closed(self):
        value = copy.deepcopy(self.inference)
        value["provenance"][
            "phase_b_external_causal_scale_coverage_approved"] = False
        with self.assertRaisesRegex(
                PhaseBInferenceContractError, "external-scale coverage"):
            self.validate(value)

    def test_teacher_leakage_is_rejected_before_any_conversion(self):
        value = copy.deepcopy(self.inference)
        value["records"][0]["candidates"][0]["teacher_covisibility"] = 0.9
        with self.assertRaisesRegex(PhaseBInferenceContractError, "forbidden deployment key"):
            self.validate(value)


class MemoryGraphCandidateAssemblerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = manifest_fixture()
        self.manifest_sha = digest(self.manifest)
        self.inference, self.pins = inference_fixture(self.manifest, self.root)
        self.inference_sha = digest(self.inference)
        self.reverse, self.reverse_pins = reverse_fixture(
            self.manifest, self.inference, self.inference_sha
        )
        self.reverse_sha = digest(self.reverse)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, *, reverse=True, reverse_artifact=None):
        if not reverse:
            return build_memory_graph_candidate_artifact(
                manifest=self.manifest,
                manifest_sha256=self.manifest_sha,
                inference_artifact=self.inference,
                inference_artifact_sha256=self.inference_sha,
                inference_pins=self.pins,
            )
        value = self.reverse if reverse_artifact is None else reverse_artifact
        return build_memory_graph_candidate_artifact(
            manifest=self.manifest,
            manifest_sha256=self.manifest_sha,
            inference_artifact=self.inference,
            inference_artifact_sha256=self.inference_sha,
            inference_pins=self.pins,
            reverse_route_artifact=value,
            reverse_route_artifact_sha256=digest(value),
            reverse_route_pins=self.reverse_pins,
        )

    def test_direct_is_always_first_priority_zero_and_hops_zero(self):
        artifact = self.build()
        matched = next(
            row for row in artifact["records"] if row["activation_status"] == "matched"
        )
        candidates = matched["memory_candidates"]
        self.assertEqual([row["candidate_mode"] for row in candidates], ["direct", "reverse"])
        self.assertEqual(candidates[0]["priority"], 0)
        self.assertEqual(candidates[0]["graph_hops"], 0)
        self.assertEqual(candidates[0]["route_frame_indices"], [])
        self.assertEqual(candidates[0]["graph_path_m"], 0.0)
        self.assertEqual(candidates[1]["priority"], 1)
        self.assertEqual(candidates[1]["graph_hops"], 2)
        self.assertEqual(candidates[1]["route_frame_indices"], [12, 8])

    def test_reverse_does_not_inherit_endpoint_pose_uncertainty(self):
        artifact = self.build()
        matched = next(
            row for row in artifact["records"] if row["activation_status"] == "matched"
        )
        direct, reverse = matched["memory_candidates"]
        self.assertTrue(direct["pose_translation_uncertainty_present"])
        self.assertGreater(direct["pose_translation_p90_m"], 0.0)
        self.assertFalse(reverse["pose_translation_uncertainty_present"])
        self.assertEqual(reverse["pose_translation_p90_m"], 0.0)

    def test_depth_confidence_is_a_score_and_may_exceed_one(self):
        artifact = self.build(reverse=False)
        matched = next(
            row for row in artifact["records"] if row["activation_status"] == "matched"
        )
        self.assertEqual(
            matched["memory_candidates"][0]["depth_confidence_mean"], 4.0
        )

    def test_reverse_is_optional_and_never_required_for_direct(self):
        artifact = self.build(reverse=False)
        matched = next(
            row for row in artifact["records"] if row["activation_status"] == "matched"
        )
        self.assertEqual(
            [row["candidate_mode"] for row in matched["memory_candidates"]],
            ["direct"],
        )
        self.assertEqual(artifact["summary"]["direct_candidate_count"], 1)
        self.assertEqual(artifact["summary"]["reverse_candidate_count"], 0)
        self.assertEqual(artifact["summary"]["states_without_reverse_route"], 1)

    def test_missing_route_in_supplied_graph_never_deletes_or_changes_direct(self):
        direct_only = self.build(reverse=False)
        value = copy.deepcopy(self.reverse)
        value["records"] = []
        value["summary"]["record_count"] = 0
        value["summary"]["route_node_count"] = 0
        with_empty_graph = self.build(reverse_artifact=value)
        direct_record = next(
            row for row in direct_only["records"] if row["activation_status"] == "matched"
        )
        graph_record = next(
            row
            for row in with_empty_graph["records"]
            if row["activation_status"] == "matched"
        )
        self.assertEqual(graph_record["memory_candidates"], direct_record["memory_candidates"])
        self.assertEqual(with_empty_graph["summary"]["states_without_reverse_route"], 1)

    def test_rejected_goal_c_has_no_memory_candidate(self):
        artifact = self.build()
        rejected = next(
            row
            for row in artifact["records"]
            if row["activation_status"] == "rejected_no_match"
        )
        self.assertEqual(rejected["goal_role"], "C")
        self.assertIsNone(rejected["selected_anchor_frame_index"])
        self.assertEqual(rejected["memory_candidates"], [])

    def test_reverse_first_subgoal_axis_is_recomputed_not_trusted(self):
        value = copy.deepcopy(self.reverse)
        value["records"][0]["first_subgoal_forward_left_m"] = [-0.2, 1.0]
        with self.assertRaisesRegex(MemoryGraphCandidateBuildError, "reverse forward"):
            self.build(reverse_artifact=value)

    def test_reverse_route_may_not_use_future_or_nonmonotone_nodes(self):
        value = copy.deepcopy(self.reverse)
        value["records"][0]["route_nodes"][0]["frame_index"] = 16
        with self.assertRaisesRegex(MemoryGraphCandidateBuildError, "strictly descending"):
            self.build(reverse_artifact=value)

        value = copy.deepcopy(self.reverse)
        value["records"][0]["route_nodes"][1]["frame_index"] = 13
        with self.assertRaisesRegex(MemoryGraphCandidateBuildError, "strictly descending"):
            self.build(reverse_artifact=value)

    def test_reverse_route_must_end_at_selected_anchor(self):
        value = copy.deepcopy(self.reverse)
        value["records"][0]["route_nodes"][1]["frame_index"] = 9
        with self.assertRaisesRegex(MemoryGraphCandidateBuildError, "selected anchor"):
            self.build(reverse_artifact=value)

    def test_each_reverse_segment_cannot_be_shorter_than_scaled_xz_chord(self):
        value = copy.deepcopy(self.reverse)
        # Keep the first subgoal valid and corrupt only the second segment, so
        # this exercises every-node geometry rather than the existing first
        # waypoint check.
        value["records"][0]["route_nodes"][1]["segment_path_m"] = 1.0
        value["records"][0]["graph_path_m"] = 2.2
        with self.assertRaisesRegex(
            MemoryGraphCandidateBuildError, "shorter than its scaled x-z chord"
        ):
            self.build(reverse_artifact=value)

    def test_graph_path_must_equal_sum_of_validated_segments(self):
        value = copy.deepcopy(self.reverse)
        value["records"][0]["graph_path_m"] = 2.4
        with self.assertRaisesRegex(MemoryGraphCandidateBuildError, "graph path sum"):
            self.build(reverse_artifact=value)

    def test_negative_depth_confidence_fails_closed(self):
        value = copy.deepcopy(self.inference)
        candidate_row = value["records"][0]["candidates"][0]
        candidate_row["goal_depth_confidence_mean"] = -0.1
        candidate_row["phase_b_input_features"][7] = -0.1
        with self.assertRaisesRegex(
            PhaseBInferenceContractError, "goal depth confidence"
        ):
            build_memory_graph_candidate_artifact(
                manifest=self.manifest,
                manifest_sha256=self.manifest_sha,
                inference_artifact=value,
                inference_artifact_sha256=digest(value),
                inference_pins=self.pins,
            )

    def test_reverse_teacher_field_is_rejected(self):
        value = copy.deepcopy(self.reverse)
        value["records"][0]["teacher_anchor"] = 8
        with self.assertRaisesRegex(MemoryGraphCandidateBuildError, "forbidden key"):
            self.build(reverse_artifact=value)

    def test_atomic_output_resume_is_byte_identical(self):
        artifact = self.build(reverse=False)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "memory_candidates.json"
            status, first_sha = write_artifact(artifact, output)
            self.assertEqual(status, "written")
            status, second_sha = write_artifact(artifact, output, resume=True)
            self.assertEqual(status, "resumed")
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(output.read_bytes(), canonical_json_bytes(artifact))


if __name__ == "__main__":
    unittest.main()
