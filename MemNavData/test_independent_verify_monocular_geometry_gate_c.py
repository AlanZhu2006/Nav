import copy

import pytest

from MemNavData.independent_verify_monocular_geometry_gate_c import (
    THRESHOLDS,
    expected_gate,
    fixed_scene_split,
    verify,
)


def metrics():
    common = {
        "token_smooth_l1": 0.1,
        "epsilon_cosine_error": 0.1,
        "critic_mse": 0.1,
        "critic_top2_overlap": 0.5,
    }
    return {
        "zero_depth_tokens": {
            **common,
            "token_cosine_error": 1.0,
            "epsilon_mse": 1.0,
            "critic_spearman": 0.40,
            "critic_top1_agreement": 0.40,
        },
        "raw_depth_tokens": {
            **common,
            "token_cosine_error": 0.50,
            "epsilon_mse": 0.50,
            "critic_spearman": 0.43,
            "critic_top1_agreement": 0.40,
        },
        "adapter": {
            **common,
            "token_cosine_error": 0.40,
            "epsilon_mse": 0.40,
            "critic_spearman": 0.44,
            "critic_top1_agreement": 0.40,
        },
    }


def population():
    scenes = [f"scene-{index:02d}" for index in range(10)]
    train, validation = fixed_scene_split(scenes, 0.2, "mdtec-gate-c-v1-20260818")
    manifest = {
        "status": "complete",
        "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
        "scale_prefix_frames": 40,
        "scenes": scenes,
        "scene_count": 10,
        "sample_count": 40,
        "rows": [
            {
                "scene": scene,
                "samples": 4,
                "shard": f"{scene}.pt",
                "scale": {
                    "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
                    "whole_episode_ground_cache_consumed": False,
                    "scale_prefix_first_frame": 0,
                    "scale_prefix_last_frame": 39,
                },
            }
            for scene in scenes
        ],
    }
    gate = expected_gate(metrics(), powered=True)
    receipt = {
        "schema": "monocular_geometry_scene_grouped_gate_c_v1_20260818",
        "status": "complete",
        "not_closed_loop_or_sr": True,
        "source_scene_count": 10,
        "source_sample_count": 40,
        "train_scenes": train,
        "validation_scenes": validation,
        "scene_overlap": [],
        "train_samples": 32,
        "validation_samples": 8,
        "metrics": metrics(),
        "gate_c": gate,
        "navdp_gradient_tensors": [],
    }
    return receipt, manifest


def test_independent_verifier_accepts_valid_positive_receipt():
    receipt, manifest = population()
    result = verify(
        receipt,
        manifest,
        validation_fraction=0.2,
        split_salt="mdtec-gate-c-v1-20260818",
        min_gate_samples=8,
        min_gate_scenes=2,
    )
    assert result["verified"] is True
    assert result["authorized"] is True
    assert result["choice"] == "latent_adapter"
    assert result["gate_c"]["thresholds_frozen_before_gate_c"] == THRESHOLDS


def test_independent_verifier_rejects_reported_choice_drift():
    receipt, manifest = population()
    receipt = copy.deepcopy(receipt)
    receipt["gate_c"]["choice"] = "raw_lingbot_depth"
    with pytest.raises(RuntimeError, match="arithmetic drift"):
        verify(
            receipt,
            manifest,
            validation_fraction=0.2,
            split_salt="mdtec-gate-c-v1-20260818",
            min_gate_samples=8,
            min_gate_scenes=2,
        )


def test_underpowered_result_is_verified_without_authorization():
    receipt, manifest = population()
    receipt = copy.deepcopy(receipt)
    receipt["gate_c"] = expected_gate(metrics(), powered=False)
    result = verify(
        receipt,
        manifest,
        validation_fraction=0.2,
        split_salt="mdtec-gate-c-v1-20260818",
        min_gate_samples=32,
        min_gate_scenes=4,
    )
    assert result["verified"] is True
    assert result["authorized"] is False


def test_independent_verifier_accepts_frozen_one_state_input_attrition():
    receipt, manifest = population()
    manifest = copy.deepcopy(manifest)
    receipt = copy.deepcopy(receipt)
    first = manifest["rows"][0]
    first["samples"] = 3
    first["selected_samples"] = 4
    first["skipped_samples"] = [{
        "frame": 240,
        "reason": "frozen_teacher_depth_audit_all_zero",
        "teacher_key": ["mp3d_2leg", first["scene"], "episode_0", 240],
    }]
    for row in manifest["rows"][1:]:
        row["selected_samples"] = row["samples"]
        row["skipped_samples"] = []
    manifest["selected_sample_count"] = 40
    manifest["sample_count"] = 39
    manifest["invalid_teacher_state_count"] = 1
    manifest["teacher_depth_audit"] = {
        "schema": "monocular_geometry_teacher_depth_population_audit_v1_20260818",
        "selected_state_count": 40,
        "valid_state_count": 39,
        "invalid_state_count": 1,
        "invalid_reason_counts": {"all_zero_depth": 1},
    }
    receipt["source_sample_count"] = 39
    validation = set(receipt["validation_scenes"])
    if first["scene"] in validation:
        receipt["validation_samples"] -= 1
    else:
        receipt["train_samples"] -= 1
    result = verify(
        receipt,
        manifest,
        validation_fraction=0.2,
        split_salt="mdtec-gate-c-v1-20260818",
        min_gate_samples=1,
        min_gate_scenes=1,
    )
    assert result["selected_sample_count"] == 40
    assert result["source_sample_count"] == 39
    assert result["invalid_teacher_state_count"] == 1
