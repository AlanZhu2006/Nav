#!/usr/bin/env python3
"""Pre-submission power audit for the consumed-pool CDEC closed loop.

This audit deliberately uses only two already-consumed, hash-bound reports:

* the 20-scene / 160-episode certified-relocalization closed-loop report; and
* the train40, scene-OOF, same-process dual-proposal certificate report.

It never reads an episode trace, development scene, or blind scene.  Its
purpose is to prevent an expensive replay whose frozen promotion criterion is
mathematically unreachable when the reference geometry decisions reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = "cdec_closed_loop_attainability_audit_v1_20260813"
EXPECTED_REFERENCE_SHA256 = (
    "0e41a6d9b339d143229ba405b04802654d2053b5d641a03ed2d09aefc1a589f4")
EXPECTED_DUAL_SHA256 = (
    "3f00f95c569c0c68175700c82c315aa4660af7050d707e80265560f47f486d39")
EXPECTED_INDEPENDENT_SHA256 = (
    "28c29703d6bb636d3c53cc9ec913327fa364987495ac23833db5f2cf6dab1fe8")
EXPECTED_RUNTIME_ARTIFACT_SHA256 = (
    "eea77098531c6e5865516d49540374edca7bb87a419613ef40d56cc2c85add31")
EXPECTED_PROTOCOL_SHA256 = (
    "1917a8407ad7a04f2fbbb2f421f7a0d302b33ae6c048bc0c05f2e2260c2dd9e5")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    require(path.is_file(), f"missing input: {path}")
    actual = sha256(path)
    require(actual == expected_sha256,
            f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON input {path}: {error}") from error
    require(isinstance(payload, Mapping), f"input is not an object: {path}")
    return payload


def exact_mcnemar(gains: int, losses: int) -> float:
    gains = int(gains)
    losses = int(losses)
    require(gains >= 0 and losses >= 0, "discordant counts must be nonnegative")
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gains, losses) + 1)
    )
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def minimum_zero_loss_gains(alpha: float) -> int:
    alpha = float(alpha)
    require(0.0 < alpha < 1.0, "alpha must lie strictly between zero and one")
    gains = 1
    while exact_mcnemar(gains, 0) >= alpha:
        gains += 1
    return gains


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    require(isinstance(value, Mapping), f"missing object field: {key}")
    return value


def require_nested_subset(
    observed: Any, authoritative: Any, *, path: str
) -> None:
    """Require every independently reconstructed field to match authority."""

    if isinstance(observed, Mapping):
        require(isinstance(authoritative, Mapping),
                f"official field is not an object: {path}")
        for key, value in observed.items():
            require(key in authoritative,
                    f"official report lacks reconstructed field: {path}.{key}")
            require_nested_subset(
                value, authoritative[key], path=f"{path}.{key}")
        return
    require(observed == authoritative,
            f"secondary verifier disagrees with official field: {path}")


def build_audit(
    reference: Mapping[str, Any],
    dual: Mapping[str, Any],
    independent: Mapping[str, Any],
    runtime_artifact: Mapping[str, Any],
    *,
    reference_path: Path,
    dual_path: Path,
    independent_path: Path,
    runtime_artifact_path: Path,
    protocol_path: Path,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return a fail-closed resource decision from immutable summaries."""

    reference_audit = _mapping(reference, "audit")
    require(reference_audit.get("status") == "ok",
            "reference closed-loop audit did not pass")
    require(reference_audit.get("development_read") is False,
            "reference report read development")
    require(reference_audit.get("blind_read") is False,
            "reference report read blind")
    require(reference_audit.get("training_scene_overlap") == [],
            "reference report has training-scene overlap")
    require(int(reference_audit.get("scenes", -1)) == 20,
            "reference report is not the frozen 20-scene run")
    require(int(reference_audit.get("episodes", -1)) == 160,
            "reference report is not the frozen 160-episode run")

    arms = _mapping(reference, "arms")
    certified = _mapping(arms, "certified_relocalization")
    conditional = _mapping(certified, "revisit_given_novel_success")
    runtime = _mapping(reference, "certified_runtime")
    eligible = int(conditional.get("eligible", -1))
    a_success = int(runtime.get("a_success_episodes", -1))
    takeovers = int(runtime.get("takeover_episodes", -1))
    fallbacks = int(runtime.get("fallback_episodes_after_a_success", -1))
    require(eligible == 120 and a_success == eligible,
            "reference conditional-Revisit denominator changed")
    require(0 <= takeovers <= eligible, "invalid reference takeover count")
    require(fallbacks == eligible - takeovers,
            "reference fallback count is inconsistent with takeovers")
    require(fallbacks == 5,
            "reference geometry-reject opportunity count changed")

    dual_scope = _mapping(dual, "scope")
    require(dual_scope.get("development_or_blind_read") is False,
            "dual-proposal report read development or blind")
    require(dual_scope.get("train_scenes_only") is True,
            "dual-proposal report is not train-only")
    require(dual_scope.get("row_scene_held_out_for_cdec") is True,
            "CDEC proposal rows are not scene-held-out")
    require(dual_scope.get("same_gpu_same_lingbot_process") is True,
            "dual proposals were not evaluated in the same process")
    method_gate = _mapping(dual, "method_gate")
    require(method_gate.get("pass") is True,
            "dual-proposal safety gate did not pass")
    require(method_gate.get("deployment_order") ==
            "geometry_first_then_cdec_on_reject",
            "dual-proposal deployment order changed")
    requirements = _mapping(method_gate, "requirements")
    require(all(value is True for value in requirements.values()),
            "one or more dual-proposal safety requirements failed")

    policies = _mapping(dual, "policies")
    geometry = _mapping(policies, "geometry")
    cascade = _mapping(
        policies, "geometry_first_then_cdec_on_reject")
    paired = _mapping(dual, "paired")
    cascade_pair = _mapping(
        paired, "geometry_first_cascade_minus_geometry_certified_actionable")
    require(int(geometry.get("sessions", -1)) == 480,
            "dual geometry policy does not contain 480 sessions")
    require(int(cascade.get("sessions", -1)) == 480,
            "dual cascade policy does not contain 480 sessions")
    require(int(cascade_pair.get("gains", -1)) == 1 and
            int(cascade_pair.get("losses", -1)) == 0,
            "dual cascade paired outcome changed")
    require(int(cascade.get("certificate_false_positive", -1)) ==
            int(geometry.get("certificate_false_positive", -2)),
            "dual cascade added certificate false positives")

    require(runtime_artifact.get("deployment_approved") is False,
            "runtime artifact unexpectedly claims deployment approval")
    runtime_semantics = _mapping(runtime_artifact, "runtime_semantics")
    require(runtime_semantics.get("authority") ==
            "rank_frozen_causal_shortlist_only",
            "runtime artifact learned authority changed")
    require(runtime_semantics.get("activation_authority") ==
            "independent_atomic_pnp_certificate",
            "runtime artifact activation authority changed")
    require(runtime_semantics.get("accepted_geometry_can_be_overridden") is False,
            "runtime artifact can override accepted geometry")

    independent_scope = _mapping(independent, "scope")
    require(independent.get("verified") is True,
            "independent raw-CSV verifier did not pass")
    require(independent_scope.get("independent_of_primary_summarizer") is True,
            "secondary verifier is not independent of the primary summarizer")
    require(independent_scope.get("development_or_blind_read") is False,
            "secondary verifier read development or blind")
    require(independent_scope.get("train40_only") is True,
            "secondary verifier is not train40-only")
    independent_inputs = _mapping(independent, "inputs")
    require(independent_inputs.get("official_report_sha256") ==
            EXPECTED_DUAL_SHA256,
            "secondary verifier was not bound to the official dual report")
    reconstructed = _mapping(independent, "reconstructed")
    for field in ("method_gate", "paired", "policies", "proposal_identity"):
        require_nested_subset(
            reconstructed.get(field), dual.get(field), path=field)

    minimum_gains = minimum_zero_loss_gains(alpha)
    maximum_gains = fallbacks
    best_case_p = exact_mcnemar(maximum_gains, 0)
    attainable = maximum_gains >= minimum_gains
    require(not attainable,
            "reference opportunity count no longer proves an unreachable gate")

    protocol_sha = sha256(protocol_path)
    require(protocol_sha == EXPECTED_PROTOCOL_SHA256,
            "frozen consumed-pool protocol changed")
    reference_joint = int(_mapping(certified, "joint").get("successes", -1))
    conditional_successes = int(conditional.get("successes", -1))
    require(reference_joint == 112 and conditional_successes == 112,
            "reference certified success count changed")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pre_submission_gate_unattainable_under_reference_behavior",
        "producer": {
            "source": str(Path(__file__).resolve()),
            "source_sha256": sha256(Path(__file__).resolve()),
        },
        "question": (
            "Can the frozen consumed-pool CDEC comparison meet its own "
            "promotion gate if the reference geometry decisions reproduce?"),
        "inputs": {
            "reference_report": str(reference_path.resolve()),
            "reference_report_sha256": EXPECTED_REFERENCE_SHA256,
            "dual_proposal_report": str(dual_path.resolve()),
            "dual_proposal_report_sha256": EXPECTED_DUAL_SHA256,
            "independent_verification": str(independent_path.resolve()),
            "independent_verification_sha256": EXPECTED_INDEPENDENT_SHA256,
            "runtime_artifact": str(runtime_artifact_path.resolve()),
            "runtime_artifact_sha256": EXPECTED_RUNTIME_ARTIFACT_SHA256,
            "frozen_protocol": str(protocol_path.resolve()),
            "frozen_protocol_sha256": protocol_sha,
        },
        "scope": {
            "reads_episode_traces": False,
            "reads_development_or_blind": False,
            "reference_pool_already_consumed": True,
            "closed_loop_executed_by_this_audit": False,
            "conditional_on_reference_geometry_decisions_reproducing": True,
        },
        "reference_behavior": {
            "scenes": 20,
            "episodes": 160,
            "a_success_episodes": eligible,
            "geometry_certificate_takeover_episodes": takeovers,
            "geometry_certificate_reject_episodes": fallbacks,
            "reference_joint_successes": reference_joint,
            "maximum_cdec_affected_pairs": maximum_gains,
            "maximum_conditional_risk_difference": maximum_gains / eligible,
            "maximum_joint_risk_difference": maximum_gains / 160.0,
        },
        "frozen_promotion_gate": {
            "two_sided_exact_mcnemar": True,
            "strict_alpha": float(alpha),
            "zero_losses_required": True,
            "minimum_zero_loss_gains_for_p_below_alpha": minimum_gains,
            "best_case_available_gains": maximum_gains,
            "best_case_losses": 0,
            "best_case_exact_mcnemar_p": best_case_p,
            "statistically_attainable": attainable,
        },
        "dual_proposal_diagnostic": {
            "train_only_sessions": int(geometry["sessions"]),
            "geometry_reject_fallback_invocations": int(
                cascade["second_certificate_invocations"]),
            "additional_certified_actionable": int(cascade_pair["gains"]),
            "lost_certified_actionable": int(cascade_pair["losses"]),
            "extra_certificate_false_positives": (
                int(cascade["certificate_false_positive"])
                - int(geometry["certificate_false_positive"])),
            "interpretation": (
                "Safety-compatible but sparse proposal complementarity; this "
                "is not a closed-loop effect estimate."),
        },
        "resource_decision": {
            "submit_full_20_scene_160_episode_replay": False,
            "branch": "do_not_run_statistically_uninformative_consumed_replay",
            "runtime_artifact_deployment_approved": False,
            "reason": (
                "CDEC may alter at most five paired outcomes when the frozen "
                "reference geometry decisions reproduce, but the preregistered "
                "two-sided exact McNemar gate with zero losses needs six gains."),
            "heldout_or_blind_authorized": False,
            "retuning_authorized": False,
        },
        "limitations": [
            "This is a power/attainability audit, not a CDEC closed-loop SR result.",
            "The five-pair bound is conditional on exact reproduction of the "
            "hash-bound reference geometry decisions.",
            "A replay that creates six or more geometry rejects would first be "
            "baseline drift requiring attribution, not automatic evidence for CDEC.",
            "Scene-cluster bootstrap and two-scene gain requirements can only "
            "further restrict promotion; they cannot repair the McNemar bound.",
        ],
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--dual-report", type=Path, required=True)
    parser.add_argument("--independent-verification", type=Path, required=True)
    parser.add_argument("--runtime-artifact", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    reference = load_json(args.reference_report, EXPECTED_REFERENCE_SHA256)
    dual = load_json(args.dual_report, EXPECTED_DUAL_SHA256)
    independent = load_json(
        args.independent_verification,
        EXPECTED_INDEPENDENT_SHA256)
    runtime_artifact = load_json(
        args.runtime_artifact, EXPECTED_RUNTIME_ARTIFACT_SHA256)
    report = build_audit(
        reference, dual, independent, runtime_artifact,
        reference_path=args.reference_report,
        dual_path=args.dual_report,
        independent_path=args.independent_verification,
        runtime_artifact_path=args.runtime_artifact,
        protocol_path=args.protocol,
        alpha=args.alpha,
    )
    atomic_json(args.output, report)
    print(json.dumps(report["resource_decision"], sort_keys=True))


if __name__ == "__main__":
    main()
