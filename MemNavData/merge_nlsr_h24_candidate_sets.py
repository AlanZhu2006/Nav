#!/usr/bin/env python3
"""Strictly join neutral NLSR candidate sets with real paired H24 receipts.

Every rollout artifact and plan-diagnostics sidecar is revalidated by the
same-state protocol before a label is copied.  The join is exact over
``(state_id, candidate_id, diffusion seeds)``: missing, extra, or duplicate
states/candidates fail closed.

The only feature added here is ``native_proposal_relation``.  It is recomputed
from the *raw t0 native proposal tensor* and critic values in the pinned plan
diagnostics, never from the executed rollout or its utility labels.  Match,
co-visibility, and pose labels are preserved as supplied by the pinned
precollection artifact; this module never manufactures them.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from MemNavData.build_nlsr_precollection_candidate_sets import (
        _sha256_file,
        _valid_sha,
        canonical_json_bytes,
        rollout_labeler_code_sha256,
        write_candidate_records,
    )
    from MemNavData.collect_real_h24_rollouts import (
        build_run_signature,
        decision_seeds,
        load_candidate_records,
        load_plan_diagnostics,
        load_server_provenance,
        validate_resume_pair,
    )
    from MemNavData.native_frontier_relation import (
        NATIVE_RELATION_FEATURE_NAMES,
        NATIVE_RELATION_SCHEMA_VERSION,
        native_frontier_relation,
    )
    from MemNavData.novel_candidate_set_schema_v2 import (
        REGRESSION_ADVANTAGE_MARGIN_M,
        USEFUL_ADVANTAGE_MARGIN_M,
        validate_candidate_dataset,
    )
    from MemNavData.novel_rollout_protocol_v2 import (
        PairedRolloutArtifact,
        load_artifact,
    )
    from MemNavData.real_h24_rollout_backend import PurePursuitConfig
except ImportError:  # pragma: no cover - direct script execution
    from build_nlsr_precollection_candidate_sets import (  # type: ignore
        _sha256_file,
        _valid_sha,
        canonical_json_bytes,
        rollout_labeler_code_sha256,
        write_candidate_records,
    )
    from collect_real_h24_rollouts import (  # type: ignore
        build_run_signature,
        decision_seeds,
        load_candidate_records,
        load_plan_diagnostics,
        load_server_provenance,
        validate_resume_pair,
    )
    from native_frontier_relation import (  # type: ignore
        NATIVE_RELATION_FEATURE_NAMES,
        NATIVE_RELATION_SCHEMA_VERSION,
        native_frontier_relation,
    )
    from novel_candidate_set_schema_v2 import (  # type: ignore
        REGRESSION_ADVANTAGE_MARGIN_M,
        USEFUL_ADVANTAGE_MARGIN_M,
        validate_candidate_dataset,
    )
    from novel_rollout_protocol_v2 import (  # type: ignore
        PairedRolloutArtifact,
        load_artifact,
    )
    from real_h24_rollout_backend import PurePursuitConfig  # type: ignore


MERGER_SCHEMA = "nlsr_h24_candidate_merge_v1"
ROLLOUT_LABEL_KEYS = frozenset(
    {
        "geodesic_progress_h8_m",
        "geodesic_progress_h24_m",
        "advantage_h24_m",
        "harm",
        "useful",
        "reachable",
        "collision_h8",
        "regression_h24",
        "rollout_label_valid",
    }
)
ROLLOUT_ZERO_FLOATS = (
    "geodesic_progress_h8_m",
    "geodesic_progress_h24_m",
    "advantage_h24_m",
)
ROLLOUT_FALSE_BOOLS = (
    "harm",
    "useful",
    "reachable",
    "collision_h8",
    "regression_h24",
    "rollout_label_valid",
)


class CandidateMergeError(RuntimeError):
    """The exact H24/precollection join cannot be proven."""


@dataclass(frozen=True)
class H24RunBinding:
    """All external values needed to reproduce the collector run signature."""

    manifest_sha256: str
    geometry_map_sha256: str
    server_provenance_sha256: str
    server_provenance: Mapping[str, str]
    server_url: str
    stop_threshold: float
    legacy_camera_height_m: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateMergeError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_array(value: object, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CandidateMergeError(f"{label} is not numeric") from error
    _require(
        result.size > 0 and np.isfinite(result).all(), f"{label} is non-finite/empty"
    )
    return result


def _strict_neutral_precollection(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        state_id = str(record["provenance"]["state_id"])
        for candidate in record["candidates"]:
            labels = candidate["labels"]
            _require(
                all(float(labels[key]) == 0.0 for key in ROLLOUT_ZERO_FLOATS)
                and all(labels[key] is False for key in ROLLOUT_FALSE_BOOLS),
                f"precollection row already carries rollout outcomes: {state_id}",
            )
            features = candidate["features"]
            _require(
                float(features["feature_presence_mask"][3]) == 0.0
                and all(
                    float(value) == 0.0
                    for value in features["native_proposal_relation"]
                ),
                f"precollection row already carries native proposal relation: {state_id}",
            )
        set_labels = record["set_labels"]
        _require(
            set_labels["candidate_set_has_positive"] is False
            and set_labels["candidate_universe_has_positive"] is False
            and set_labels["candidate_coverage_miss"] is False
            and set_labels["coverage_label_valid"] is False
            and set_labels["oracle_best_candidate_id"] == "dustbin",
            f"precollection set already claims H24 utility: {state_id}",
        )


def _artifact_paths(root: Path) -> list[Path]:
    _require(root.is_dir(), f"H24 rollout root is not a directory: {root}")
    candidates = sorted(
        path for path in root.rglob("*.json") if not path.name.endswith(".plans.json")
    )
    _require(bool(candidates), "H24 rollout root contains no artifacts")
    return candidates


def _diagnostics_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.stem}.plans{artifact_path.suffix}")


def _validate_state_binding(
    record: Mapping[str, Any],
    artifact: PairedRolloutArtifact,
) -> None:
    provenance = record["provenance"]
    state = artifact.state
    expected = {
        "state_id": state.state_id,
        "session_id": state.session_id,
        "goal_epoch": state.goal_epoch,
        "goal_sha256": state.goal_sha256,
        "navdp_fifo_sha256": state.manifest_fifo_sha256,
        "environment_id": state.environment_id,
        "navmesh_sha256": state.navmesh_sha256,
    }
    for field, value in expected.items():
        _require(
            provenance[field] == value,
            f"candidate/H24 state binding mismatch for {field}: {state.state_id}",
        )


def _candidate_ids(record: Mapping[str, Any]) -> list[str]:
    return [
        str(candidate["candidate_id"])
        for candidate in record["candidates"]
        if candidate["candidate_type"] != "dustbin"
    ]


def _verify_run_binding(
    binding: H24RunBinding,
    *,
    candidate_sha256: str,
    expected_run_signature_sha256: str,
    base_seed: int,
) -> tuple[str, str]:
    _require(isinstance(binding, H24RunBinding), "H24 run binding has wrong type")
    manifest_sha = _valid_sha(binding.manifest_sha256, "H24 manifest SHA")
    geometry_sha = _valid_sha(binding.geometry_map_sha256, "geometry map SHA")
    server_sha = _valid_sha(binding.server_provenance_sha256, "server provenance SHA")
    server = binding.server_provenance
    required_server_keys = {
        "navdp_server_sha256",
        "policy_agent_sha256",
        "deterministic_seed_sha256",
        "checkpoint_sha256",
        "wrapper_sha256",
    }
    _require(
        isinstance(server, Mapping) and set(server) == required_server_keys,
        "server provenance fields changed",
    )
    for key in sorted(server):
        _valid_sha(server[key], f"server provenance {key}")
    _require(
        _sha256_bytes(canonical_json_bytes(dict(server))) == server_sha,
        "in-memory server provenance differs from its canonical SHA pin",
    )
    _require(
        isinstance(binding.server_url, str)
        and binding.server_url.startswith(("http://", "https://")),
        "server URL must be HTTP(S)",
    )
    _require(
        isinstance(binding.stop_threshold, (int, float))
        and not isinstance(binding.stop_threshold, bool)
        and math.isfinite(float(binding.stop_threshold)),
        "stop threshold must be finite",
    )
    _require(
        isinstance(binding.legacy_camera_height_m, (int, float))
        and not isinstance(binding.legacy_camera_height_m, bool)
        and math.isfinite(float(binding.legacy_camera_height_m))
        and float(binding.legacy_camera_height_m) > 0.0,
        "legacy camera height must be finite and positive",
    )
    recomputed = build_run_signature(
        candidate_sha256=candidate_sha256,
        manifest_sha256=manifest_sha,
        geometry_map_sha256=geometry_sha,
        server_provenance_sha256=server_sha,
        server_url=binding.server_url,
        base_seed=base_seed,
        stop_threshold=float(binding.stop_threshold),
        legacy_camera_height_m=float(binding.legacy_camera_height_m),
        controller=PurePursuitConfig(),
    )
    expected = _valid_sha(expected_run_signature_sha256, "expected H24 run signature")
    _require(
        recomputed == expected,
        "expected H24 run signature cannot be reproduced from pinned inputs/code",
    )
    binding_sha = _sha256_bytes(
        canonical_json_bytes(
            {
                "manifest_sha256": manifest_sha,
                "geometry_map_sha256": geometry_sha,
                "server_provenance_sha256": server_sha,
                "server_url": binding.server_url.rstrip("/"),
                "stop_threshold": float(binding.stop_threshold),
                "legacy_camera_height_m": float(binding.legacy_camera_height_m),
                "base_seed": base_seed,
                "run_signature_sha256": recomputed,
            }
        )
    )
    return recomputed, binding_sha


def _t0_native_proposals(
    artifact: PairedRolloutArtifact,
    diagnostics: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    native = next(
        (outcome for outcome in artifact.outcomes if outcome.candidate_id == "native"),
        None,
    )
    _require(native is not None, "H24 artifact lacks native outcome")
    t0 = [plan for plan in native.plans if plan.commitment_index == 0]
    _require(len(t0) == 1, "native H24 trace lacks unique t0 plan")
    plan = t0[0]
    by_candidate = diagnostics.get("by_candidate")
    _require(
        isinstance(by_candidate, Mapping), "plan diagnostics candidate map is malformed"
    )
    native_plans = by_candidate.get("native")
    _require(isinstance(native_plans, Mapping), "plan diagnostics lack native arm")
    row = native_plans.get(plan.plan_sha256)
    _require(isinstance(row, Mapping), "plan diagnostics lack native t0 plan")
    trajectories = _finite_array(row.get("all_trajectory"), "raw native trajectories")
    values = _finite_array(row.get("all_values"), "raw native critic values")
    _require(
        trajectories.ndim == 4
        and trajectories.shape[0] == 1
        and trajectories.shape[2:] == (24, 3)
        and values.shape == trajectories.shape[:2],
        "raw native proposal/value tensors have changed shape",
    )
    selected = row.get("server_selected_trajectory_index")
    critic_selected = int(np.argmax(values[0]))
    _require(
        selected is None
        or (
            isinstance(selected, int)
            and not isinstance(selected, bool)
            and 0 <= selected < trajectories.shape[1]
            and selected == critic_selected
        ),
        "native t0 selected proposal index is out of range or disagrees "
        "with critic argmax",
    )
    raw_selected = _finite_array(
        row.get("raw_selected_trajectory"), "raw selected native trajectory"
    )
    _require(
        raw_selected.shape == (24, 3)
        and np.array_equal(raw_selected, trajectories[0, critic_selected]),
        "native t0 selected proposal does not match raw proposal tensor",
    )
    return trajectories[0], values[0], critic_selected


def _validate_residual_t0_subgoal(
    artifact: PairedRolloutArtifact,
    candidate: Mapping[str, Any],
) -> None:
    candidate_id = str(candidate["candidate_id"])
    outcome = next(
        (row for row in artifact.outcomes if row.candidate_id == candidate_id),
        None,
    )
    _require(outcome is not None, f"H24 artifact lacks residual {candidate_id}")
    t0 = [plan for plan in outcome.plans if plan.commitment_index == 0]
    _require(len(t0) == 1, f"residual {candidate_id} lacks unique t0 plan")
    local = t0[0].local_subgoal_forward_left_m
    _require(
        local is not None and len(local) == 2,
        f"residual {candidate_id} lacks t0 local goal",
    )
    features = candidate["features"]
    expected = (
        float(features["subgoal_forward_m"]),
        float(features["subgoal_left_m"]),
    )
    _require(
        all(
            math.isclose(float(actual), reference, rel_tol=0.0, abs_tol=1e-6)
            for actual, reference in zip(local, expected)
        ),
        f"residual {candidate_id} H24 subgoal differs from deployment feature",
    )


def _load_h24_corpus(
    *,
    rollout_root: Path,
    records: Sequence[Mapping[str, Any]],
    expected_run_signature_sha256: str,
    base_seed: int,
) -> tuple[
    dict[str, tuple[PairedRolloutArtifact, Mapping[str, Any]]],
    str,
]:
    run_signature = _valid_sha(
        expected_run_signature_sha256, "expected H24 run signature"
    )
    _require(
        isinstance(base_seed, int)
        and not isinstance(base_seed, bool)
        and 0 <= base_seed < 2**63,
        "H24 base seed is invalid",
    )
    record_by_state = {
        str(record["provenance"]["state_id"]): record for record in records
    }
    _require(
        len(record_by_state) == len(records),
        "precollection state ids are duplicated",
    )
    loaded = {}
    corpus_rows = []
    for artifact_path in _artifact_paths(rollout_root):
        diagnostics_path = _diagnostics_path(artifact_path)
        try:
            artifact = load_artifact(artifact_path)
        except Exception as error:
            raise CandidateMergeError(
                f"invalid H24 artifact {artifact_path}: {error}"
            ) from error
        state_id = artifact.state.state_id
        _require(state_id in record_by_state, f"extra H24 state {state_id}")
        _require(state_id not in loaded, f"duplicate H24 state {state_id}")
        expected_ids = _candidate_ids(record_by_state[state_id])
        seeds = decision_seeds(base_seed, state_id)
        try:
            validated = validate_resume_pair(
                artifact_path,
                diagnostics_path,
                state_id=state_id,
                run_signature_sha256=run_signature,
                diffusion_seeds=seeds,
                candidate_ids=expected_ids,
            )
            diagnostics = load_plan_diagnostics(diagnostics_path)
        except Exception as error:
            raise CandidateMergeError(
                f"H24/plan protocol validation failed for {state_id}: {error}"
            ) from error
        _require(
            validated.artifact_sha256 == artifact.artifact_sha256,
            f"H24 artifact changed during validation for {state_id}",
        )
        _validate_state_binding(record_by_state[state_id], artifact)
        actual_ids = [outcome.candidate_id for outcome in artifact.outcomes]
        expected_order = sorted(
            expected_ids, key=lambda value: (value != "native", value)
        )
        _require(
            actual_ids == expected_order, f"H24 candidate ids changed for {state_id}"
        )
        loaded[state_id] = (artifact, diagnostics)
        corpus_rows.append(
            {
                "state_id": state_id,
                "artifact_file_sha256": _sha256_file(artifact_path),
                "artifact_sha256": artifact.artifact_sha256,
                "diagnostics_file_sha256": _sha256_file(diagnostics_path),
                "diffusion_seeds": list(seeds),
            }
        )
    _require(
        set(loaded) == set(record_by_state),
        "H24 corpus has missing or extra candidate decisions",
    )
    return loaded, _sha256_bytes(
        canonical_json_bytes(sorted(corpus_rows, key=lambda row: row["state_id"]))
    )


def _feature_builder_chain_sha256(
    precollection_builder_sha256: str,
) -> str:
    root = Path(__file__).resolve().parents[1]
    precollection_sha = _valid_sha(
        precollection_builder_sha256, "precollection feature-builder SHA"
    )
    files = (
        ("h24_label_join", "MemNavData/merge_nlsr_h24_candidate_sets.py"),
        ("native_relation", "MemNavData/native_frontier_relation.py"),
    )
    rows = [
        {
            "stage": "precollection",
            "path": "MemNavData/build_nlsr_precollection_candidate_sets.py",
            "sha256": precollection_sha,
        }
    ]
    rows.extend(
        {
            "stage": stage,
            "path": relative,
            "sha256": _sha256_file(root / relative),
        }
        for stage, relative in files
    )
    return _sha256_bytes(canonical_json_bytes(rows))


def _copy_rollout_labels(
    candidate: dict[str, Any],
    raw_labels: Mapping[str, object],
) -> None:
    _require(
        frozenset(raw_labels) == ROLLOUT_LABEL_KEYS,
        f"H24 label fields changed for {candidate['candidate_id']}",
    )
    labels = candidate["labels"]
    for key in ROLLOUT_LABEL_KEYS:
        labels[key] = raw_labels[key]


def merge_candidate_records(
    *,
    records: Sequence[Mapping[str, Any]],
    precollection_sha256: str,
    rollout_root: Path,
    expected_run_signature_sha256: str,
    base_seed: int,
    run_binding: H24RunBinding,
) -> list[dict[str, Any]]:
    """Revalidate and merge one exact real-H24 corpus."""
    precollection_sha = _valid_sha(precollection_sha256, "precollection SHA")
    _strict_neutral_precollection(records)
    labeler_sha = rollout_labeler_code_sha256()
    run_signature, run_binding_sha = _verify_run_binding(
        run_binding,
        candidate_sha256=precollection_sha,
        expected_run_signature_sha256=expected_run_signature_sha256,
        base_seed=base_seed,
    )
    for record in records:
        _require(
            record["provenance"]["rollout_labeler_sha256"] == labeler_sha,
            "precollection rollout-labeler code pin differs from current join code",
        )
        _require(
            record["provenance"]["source_policy_sha256"]
            == run_binding.server_provenance["checkpoint_sha256"],
            "precollection source policy differs from H24 server checkpoint",
        )
    precollection_builder_shas = {
        str(record["provenance"]["feature_builder_sha256"]) for record in records
    }
    _require(
        len(precollection_builder_shas) == 1,
        "precollection records disagree on feature-builder provenance",
    )
    h24, corpus_sha = _load_h24_corpus(
        rollout_root=rollout_root,
        records=records,
        expected_run_signature_sha256=run_signature,
        base_seed=base_seed,
    )
    feature_builder_sha = _feature_builder_chain_sha256(
        next(iter(precollection_builder_shas))
    )
    dataset_digest = _sha256_bytes(
        canonical_json_bytes(
            {
                "schema": MERGER_SCHEMA,
                "precollection_sha256": precollection_sha,
                "h24_corpus_sha256": corpus_sha,
                "run_signature_sha256": run_signature,
                "run_binding_sha256": run_binding_sha,
                "feature_builder_chain_sha256": feature_builder_sha,
                "rollout_labeler_sha256": labeler_sha,
                "native_relation_schema": NATIVE_RELATION_SCHEMA_VERSION,
            }
        )
    )
    merged = copy.deepcopy(list(records))
    for record in merged:
        state_id = str(record["provenance"]["state_id"])
        artifact, diagnostics = h24[state_id]
        trajectories, values, selected = _t0_native_proposals(artifact, diagnostics)
        labels_by_candidate = artifact.labels_by_candidate
        _require(
            set(labels_by_candidate) == set(_candidate_ids(record)),
            f"H24 label candidate ids changed for {state_id}",
        )
        for candidate in record["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            candidate_type = str(candidate["candidate_type"])
            if candidate_type == "dustbin":
                continue
            _copy_rollout_labels(candidate, labels_by_candidate[candidate_id])
            if candidate_type != "native":
                _validate_residual_t0_subgoal(artifact, candidate)
                features = candidate["features"]
                relation = native_frontier_relation(
                    trajectories,
                    (
                        float(features["subgoal_forward_m"]),
                        float(features["subgoal_left_m"]),
                    ),
                    selected_index=selected,
                    native_values=values,
                )
                _require(
                    relation.shape == (len(NATIVE_RELATION_FEATURE_NAMES),),
                    "native proposal relation shape changed",
                )
                features["native_proposal_relation"] = relation.tolist()
                features["feature_presence_mask"][3] = 1.0
        candidates = record["candidates"]
        useful = [
            candidate for candidate in candidates[1:-1] if candidate["labels"]["useful"]
        ]
        set_labels = record["set_labels"]
        set_labels.update(
            {
                "candidate_set_has_positive": bool(useful),
                # The real corpus measures the frozen shortlist only.  It cannot
                # claim utility for uncollected proposal-universe candidates.
                "candidate_universe_has_positive": False,
                "candidate_coverage_miss": False,
                "coverage_label_valid": False,
                "oracle_best_candidate_id": (
                    sorted(
                        useful,
                        key=lambda candidate: (
                            -float(candidate["labels"]["advantage_h24_m"]),
                            str(candidate["candidate_id"]),
                        ),
                    )[0]["candidate_id"]
                    if useful
                    else "dustbin"
                ),
            }
        )
        native_labels = candidates[0]["labels"]
        _require(
            native_labels["rollout_label_valid"] is True
            and native_labels["reachable"] is True
            and float(native_labels["advantage_h24_m"]) == 0.0,
            f"native H24 label is invalid for {state_id}",
        )
        # Re-derive the safety decisions instead of trusting a merely
        # schema-shaped label dictionary.
        native_progress = float(native_labels["geodesic_progress_h24_m"])
        for candidate in candidates[1:-1]:
            labels = candidate["labels"]
            if not labels["rollout_label_valid"]:
                continue
            advantage = float(labels["geodesic_progress_h24_m"]) - native_progress
            _require(
                math.isclose(
                    advantage,
                    float(labels["advantage_h24_m"]),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ),
                f"H24 advantage is inconsistent for {state_id}/{candidate['candidate_id']}",
            )
            regression = advantage <= -REGRESSION_ADVANTAGE_MARGIN_M
            harm = bool(labels["collision_h8"] or regression)
            useful_expected = bool(
                labels["reachable"]
                and not harm
                and advantage >= USEFUL_ADVANTAGE_MARGIN_M
            )
            _require(
                labels["regression_h24"] is regression
                and labels["harm"] is harm
                and labels["useful"] is useful_expected,
                f"H24 safety labels are inconsistent for "
                f"{state_id}/{candidate['candidate_id']}",
            )
        record["provenance"]["dataset_id"] = f"nlsr-h24:{dataset_digest}"
        record["provenance"]["feature_builder_sha256"] = feature_builder_sha
    try:
        validate_candidate_dataset(merged)
    except Exception as error:
        raise CandidateMergeError(
            f"merged candidate dataset is invalid: {error}"
        ) from error
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precollection", type=Path, required=True)
    parser.add_argument("--expected-precollection-sha", required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--expected-run-signature-sha", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--geometry-map-sha", required=True)
    parser.add_argument("--server-provenance", type=Path, required=True)
    parser.add_argument("--expected-server-provenance-sha", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--stop-threshold", type=float, required=True)
    parser.add_argument("--legacy-camera-height-m", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha-out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records, neutral = load_candidate_records(
            args.precollection, args.expected_precollection_sha
        )
    except Exception as error:
        raise CandidateMergeError(
            f"precollection artifact is invalid: {error}"
        ) from error
    _require(neutral, "input must be a strictly neutral precollection artifact")
    try:
        server_provenance = load_server_provenance(
            args.server_provenance, args.expected_server_provenance_sha
        )
    except Exception as error:
        raise CandidateMergeError(
            f"server provenance artifact is invalid: {error}"
        ) from error
    merged = merge_candidate_records(
        records=records,
        precollection_sha256=args.expected_precollection_sha,
        rollout_root=args.rollout_root,
        expected_run_signature_sha256=args.expected_run_signature_sha,
        base_seed=args.base_seed,
        run_binding=H24RunBinding(
            manifest_sha256=args.manifest_sha,
            geometry_map_sha256=args.geometry_map_sha,
            server_provenance_sha256=args.expected_server_provenance_sha,
            server_provenance=server_provenance,
            server_url=args.server_url,
            stop_threshold=args.stop_threshold,
            legacy_camera_height_m=args.legacy_camera_height_m,
        ),
    )
    status, digest = write_candidate_records(
        merged, args.out, args.sha_out, resume=args.resume, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "schema_version": MERGER_SCHEMA,
                "status": status,
                "records": len(merged),
                "output_sha256": digest,
                "feature_builder_chain_sha256": (
                    merged[0]["provenance"]["feature_builder_sha256"]
                ),
                "rollout_labeler_sha256": rollout_labeler_code_sha256(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
