#!/usr/bin/env python3
"""Independent arithmetic verifier for the MDTEC scene-grouped Gate C.

This verifier deliberately does not import the trainer.  It checks the frozen
scene partition, source population, threshold arithmetic, and winner selection
from immutable JSON receipts.  It verifies either a positive or a negative
Gate C result; authorization is never required for verification to pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "monocular_geometry_scene_grouped_gate_c_v1_20260818"
THRESHOLDS = {
    "token_cosine_error_vs_zero_max_ratio": 0.80,
    "epsilon_mse_vs_zero_max_ratio": 0.90,
    "critic_noninferiority_margin": 0.05,
    "critic_minimum_improvement": 0.02,
    "adapter_vs_raw_epsilon_winner_ratio": 0.90,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fixed_scene_split(
    scenes: list[str], validation_fraction: float, salt: str
) -> tuple[list[str], list[str]]:
    require(0.0 < validation_fraction < 1.0, "invalid validation fraction")
    ordered = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{salt}:{scene}".encode()).hexdigest(),
    )
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation_count = min(validation_count, len(ordered) - 1)
    validation = sorted(ordered[:validation_count])
    train = sorted(set(ordered) - set(validation))
    return train, validation


def qualifies(candidate: dict, zero: dict) -> bool:
    return bool(
        candidate["token_cosine_error"]
        <= THRESHOLDS["token_cosine_error_vs_zero_max_ratio"]
        * zero["token_cosine_error"]
        and candidate["epsilon_mse"]
        <= THRESHOLDS["epsilon_mse_vs_zero_max_ratio"] * zero["epsilon_mse"]
        and candidate["critic_spearman"]
        >= zero["critic_spearman"] - THRESHOLDS["critic_noninferiority_margin"]
        and candidate["critic_top1_agreement"]
        >= zero["critic_top1_agreement"]
        - THRESHOLDS["critic_noninferiority_margin"]
        and (
            candidate["critic_spearman"]
            >= zero["critic_spearman"]
            + THRESHOLDS["critic_minimum_improvement"]
            or candidate["critic_top1_agreement"]
            >= zero["critic_top1_agreement"]
            + THRESHOLDS["critic_minimum_improvement"]
        )
    )


def expected_gate(metrics: dict, powered: bool) -> dict:
    if not powered:
        return {
            "authorized": False,
            "reason": "underpowered_diagnostic_no_gate_decision",
        }
    zero = metrics["zero_depth_tokens"]
    raw = metrics["raw_depth_tokens"]
    adapter = metrics["adapter"]
    raw_ok = qualifies(raw, zero)
    adapter_ok = qualifies(adapter, zero)
    if not raw_ok and not adapter_ok:
        choice = "stop_no_rgb_only_geometry_path_qualifies"
    elif raw_ok and not adapter_ok:
        choice = "raw_lingbot_depth"
    elif adapter_ok and not raw_ok:
        choice = "latent_adapter"
    else:
        choice = (
            "latent_adapter"
            if adapter["epsilon_mse"]
            <= THRESHOLDS["adapter_vs_raw_epsilon_winner_ratio"]
            * raw["epsilon_mse"]
            and adapter["critic_spearman"] >= raw["critic_spearman"]
            else "raw_lingbot_depth_simpler_tie_break"
        )
    return {
        "authorized": raw_ok or adapter_ok,
        "raw_qualifies": raw_ok,
        "adapter_qualifies": adapter_ok,
        "choice": choice,
        "thresholds_frozen_before_gate_c": THRESHOLDS,
    }


def verify(
    receipt: dict,
    manifest: dict,
    *,
    validation_fraction: float,
    split_salt: str,
    min_gate_samples: int,
    min_gate_scenes: int,
) -> dict:
    require(receipt.get("schema") == SCHEMA, "unexpected Gate C schema")
    require(receipt.get("status") == "complete", "Gate C receipt incomplete")
    require(receipt.get("not_closed_loop_or_sr") is True, "receipt misstates scope")
    require(manifest.get("status") == "complete", "source manifest incomplete")
    require(
        manifest.get("scale_evidence_contract")
        == "causal_first_prefix_rgb_only_v1",
        "source manifest does not use causal scale",
    )
    scale_prefix_frames = int(manifest.get("scale_prefix_frames", -1))
    require(scale_prefix_frames == 40, "scale prefix length drift")
    scenes = sorted({str(scene) for scene in manifest["scenes"]})
    require(len(scenes) == int(manifest["scene_count"]), "manifest scene count drift")
    require(
        sum(int(row["samples"]) for row in manifest["rows"])
        == int(manifest["sample_count"]),
        "manifest sample count drift",
    )
    selected_sample_count = sum(
        int(row.get("selected_samples", row["samples"]))
        for row in manifest["rows"]
    )
    invalid_teacher_state_count = sum(
        len(row.get("skipped_samples", [])) for row in manifest["rows"]
    )
    require(
        selected_sample_count
        == int(manifest["sample_count"]) + invalid_teacher_state_count,
        "selected/valid/invalid sample arithmetic drift",
    )
    if "selected_sample_count" in manifest:
        require(
            int(manifest["selected_sample_count"]) == selected_sample_count,
            "selected sample count drift",
        )
    if "invalid_teacher_state_count" in manifest:
        require(
            int(manifest["invalid_teacher_state_count"])
            == invalid_teacher_state_count,
            "invalid teacher-state count drift",
        )
    teacher_depth_audit = manifest.get("teacher_depth_audit")
    if teacher_depth_audit is not None:
        require(
            teacher_depth_audit.get("schema")
            == "monocular_geometry_teacher_depth_population_audit_v1_20260818",
            "teacher-depth audit schema drift",
        )
        require(
            int(teacher_depth_audit.get("selected_state_count", -1))
            == selected_sample_count,
            "teacher-depth audit population drift",
        )
        require(
            int(teacher_depth_audit.get("valid_state_count", -1))
            == int(manifest["sample_count"]),
            "teacher-depth audit valid-state drift",
        )
        require(
            int(teacher_depth_audit.get("invalid_state_count", -1))
            == invalid_teacher_state_count,
            "teacher-depth audit attrition drift",
        )
        require(
            teacher_depth_audit.get("invalid_reason_counts")
            == ({"all_zero_depth": invalid_teacher_state_count}
                if invalid_teacher_state_count else {}),
            "teacher-depth audit reason drift",
        )
        for row in manifest["rows"]:
            for skipped in row.get("skipped_samples", []):
                require(
                    skipped.get("reason")
                    == "frozen_teacher_depth_audit_all_zero",
                    "unauthorized teacher-depth attrition reason",
                )
    for row in manifest["rows"]:
        scale = row.get("scale", {})
        require(
            scale.get("scale_evidence_contract")
            == "causal_first_prefix_rgb_only_v1",
            "row scale contract drift",
        )
        require(
            scale.get("whole_episode_ground_cache_consumed") is False,
            "row consumed whole-episode scale",
        )
        require(
            int(scale.get("scale_prefix_first_frame", -1)) == 0
            and int(scale.get("scale_prefix_last_frame", -1))
            == scale_prefix_frames - 1,
            "row scale prefix boundary drift",
        )
    require(int(receipt["source_scene_count"]) == len(scenes), "source scene drift")
    require(
        int(receipt["source_sample_count"]) == int(manifest["sample_count"]),
        "source sample drift",
    )
    train, validation = fixed_scene_split(scenes, validation_fraction, split_salt)
    require(receipt["train_scenes"] == train, "train scene split drift")
    require(receipt["validation_scenes"] == validation, "validation split drift")
    require(not set(train) & set(validation), "scene overlap")
    require(receipt.get("scene_overlap") == [], "receipt reports scene overlap")

    metrics = receipt["metrics"]
    required_representations = {"zero_depth_tokens", "raw_depth_tokens", "adapter"}
    require(set(metrics) == required_representations, "representation set drift")
    required_metrics = {
        "token_cosine_error",
        "token_smooth_l1",
        "epsilon_mse",
        "epsilon_cosine_error",
        "critic_mse",
        "critic_spearman",
        "critic_top1_agreement",
        "critic_top2_overlap",
    }
    for name, values in metrics.items():
        require(set(values) == required_metrics, f"metric set drift for {name}")
        require(
            all(math.isfinite(float(value)) for value in values.values()),
            f"non-finite metric for {name}",
        )

    validation_samples = int(receipt["validation_samples"])
    require(
        int(receipt["train_samples"]) + validation_samples
        == int(receipt["source_sample_count"]),
        "train/validation sample count drift",
    )
    powered = (
        validation_samples >= min_gate_samples
        and len(validation) >= min_gate_scenes
    )
    independently_recomputed = expected_gate(metrics, powered)
    require(receipt["gate_c"] == independently_recomputed, "Gate C arithmetic drift")
    require(receipt.get("navdp_gradient_tensors") == [], "NavDP has gradients")
    return {
        "schema": "independent_monocular_geometry_gate_c_verification_v1_20260818",
        "verified": True,
        "authorized": bool(independently_recomputed["authorized"]),
        "choice": independently_recomputed.get("choice"),
        "source_scene_count": len(scenes),
        "source_sample_count": int(manifest["sample_count"]),
        "selected_sample_count": selected_sample_count,
        "invalid_teacher_state_count": invalid_teacher_state_count,
        "validation_scene_count": len(validation),
        "validation_sample_count": validation_samples,
        "scale_evidence_contract": manifest["scale_evidence_contract"],
        "scale_prefix_frames": scale_prefix_frames,
        "gate_c": independently_recomputed,
        "not_closed_loop_or_sr": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-salt", default="mdtec-gate-c-v1-20260818")
    parser.add_argument("--min-gate-samples", type=int, default=32)
    parser.add_argument("--min-gate-scenes", type=int, default=4)
    args = parser.parse_args()
    result = verify(
        json.loads(args.receipt.read_text()),
        json.loads(args.manifest.read_text()),
        validation_fraction=args.validation_fraction,
        split_salt=args.split_salt,
        min_gate_samples=args.min_gate_samples,
        min_gate_scenes=args.min_gate_scenes,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
