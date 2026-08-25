#!/usr/bin/env python3
"""Fail-closed audit for the all-CEC mixed-role integration smoke.

This audit intentionally makes no navigation-performance claim.  It verifies
that every controller received the same per-action CEC proof stream, that the
chosen Novel action fell back to mono NavDP, and that the chosen Revisit action
used only the controller-native projection authorized by that proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "cec_controller_portability_smoke_audit_v1"
HUB_SCHEMA = "cec_controller_portability_hub_v2"
CONTROLLERS = ("navdp", "vint", "gnm", "nomad", "iplanner", "viplanner")
ADAPTERS = {
    "navdp": "bearing_mixedgoal",
    "vint": "verified_anchor_imagegoal",
    "gnm": "verified_anchor_imagegoal",
    "nomad": "verified_anchor_imagegoal",
    "iplanner": "bearing_pointgoal",
    "viplanner": "bearing_pointgoal",
}
# Controllers whose accepted branch takes the hash-bound certified anchor as
# ImageGoal (short RGB-context family) instead of a 2.5 m PointGoal.
ANCHOR_IMAGEGOAL_CONTROLLERS = frozenset({"vint", "gnm", "nomad"})
FORBIDDEN_RUNTIME_FIELDS = {
    "analysis_role", "role", "goal_role", "query_role", "is_revisit",
    "is_novel", "oracle_pose", "gt_pose", "ground_truth_pose",
    "habitat_pose",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    controller, separator, raw_path = value.partition("=")
    require(bool(separator), "--run must be CONTROLLER=PATH")
    require(controller in CONTROLLERS, f"unknown controller {controller!r}")
    path = Path(raw_path).resolve()
    require(path.is_dir(), f"run directory is missing: {path}")
    return controller, path


def pointgoal_norm(payload: dict[str, Any]) -> float:
    goal_x = payload.get("goal_x")
    goal_y = payload.get("goal_y")
    require(
        isinstance(goal_x, list) and len(goal_x) == 1
        and isinstance(goal_y, list) and len(goal_y) == 1,
        "bearing projection lost singleton [forward,left] coordinates",
    )
    return math.hypot(float(goal_x[0]), float(goal_y[0]))


def audit_run(controller: str, root: Path) -> dict[str, Any]:
    result = root / "result"
    summary_path = result / "summary.json"
    metric_path = result / "metric.csv"
    require(summary_path.is_file() and metric_path.is_file(),
            f"{controller}: incomplete result")
    summary = json.loads(summary_path.read_text())
    require(summary.get("server_backend") == "cec_portability",
            f"{controller}: wrong backend")
    require(summary.get("queries") == 2, f"{controller}: expected two queries")
    require(summary.get("role_counts") == {"novel": 1, "revisit": 1},
            f"{controller}: mixed-role population changed")
    require(summary.get("runtime_role_visibility") == "none",
            f"{controller}: role label was runtime-visible")
    require(summary.get("shared_A_all_hashes_ok") is True,
            f"{controller}: frozen online-A identity failed")
    require(summary.get("shared_A_total_diffusion_samples") == 0,
            f"{controller}: replay sampled policy noise")
    require(summary.get("metric_depth_sensor_consumed_episodes") == 0,
            f"{controller}: metric depth entered the policy")
    require(summary.get("runtime_failure_plans") == 0,
            f"{controller}: runtime failure was hidden in the smoke")

    with metric_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2, f"{controller}: metric row count changed")
    require({row["analysis_role"] for row in rows} == {"novel", "revisit"},
            f"{controller}: metric roles changed")
    scene = str(summary["scene"])
    episode = rows[0]["episode"]

    role_receipts: dict[str, dict[str, Any]] = {}
    for role in ("novel", "revisit"):
        matches = sorted(result.glob(f"*_{role}_plans.json"))
        require(len(matches) == 1,
                f"{controller}/{scene}/{role}: expected one plan file")
        plan_path = matches[0]
        payload = json.loads(plan_path.read_text())
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{controller}/{scene}/{role}: role projection seal missing")
        runtime_fields = set(payload.get("query_runtime_fields", []))
        require(not runtime_fields.intersection(FORBIDDEN_RUNTIME_FIELDS),
                f"{controller}/{scene}/{role}: forbidden runtime field")
        plans = payload.get("query_leg")
        require(isinstance(plans, list) and plans,
                f"{controller}/{scene}/{role}: no controller decision")
        proofs = []
        anchors = []
        for plan in plans:
            require(plan.get("cec_portability_schema") == HUB_SCHEMA,
                    f"{controller}/{scene}/{role}: wrong hub schema")
            require(plan.get("cec_decision_scope") == "per_action",
                    f"{controller}/{scene}/{role}: action scope changed")
            require(plan.get("cec_accept_controller") == controller,
                    f"{controller}/{scene}/{role}: controller receipt changed")
            require(plan.get("cec_accept_adapter") == ADAPTERS[controller],
                    f"{controller}/{scene}/{role}: adapter receipt changed")
            require(plan.get("metric_depth_sensor_consumed") is False,
                    f"{controller}/{scene}/{role}: sensor audit is ambiguous")
            proof = plan.get("cec_proof_sha256")
            require(isinstance(proof, str) and SHA256.fullmatch(proof) is not None,
                    f"{controller}/{scene}/{role}: invalid proof digest")
            proofs.append(proof)
            projected = plan.get("cec_projected_goal")
            require(isinstance(projected, dict),
                    f"{controller}/{scene}/{role}: projection is missing")
            if role == "novel":
                require(plan.get("cec_takeover") is False
                        and plan.get("cec_action_state") == "fallback",
                        f"{controller}/{scene}: Novel did not exact-fallback")
                require(projected == {"fallback_this_action": True},
                        f"{controller}/{scene}: Novel projection is not fallback")
                require(plan.get("cec_controller_seed_consumed") is True,
                        f"{controller}/{scene}: fallback seed was not consumed")
                if controller in ANCHOR_IMAGEGOAL_CONTROLLERS:
                    require(plan.get("cec_alternate_context_shadowed") is True,
                            f"{controller}/{scene}: short-context controller "
                            "was not shadowed")
            else:
                require(plan.get("cec_takeover") is True
                        and plan.get("cec_action_state") == "takeover",
                        f"{controller}/{scene}: Revisit was not authorized")
                anchor = plan.get("cec_selected_anchor")
                require(isinstance(anchor, int),
                        f"{controller}/{scene}: Revisit anchor is missing")
                anchors.append(anchor)
                if controller in ANCHOR_IMAGEGOAL_CONTROLLERS:
                    anchor_sha = projected.get("cec_anchor_sha256")
                    require(isinstance(anchor_sha, str)
                            and SHA256.fullmatch(anchor_sha) is not None,
                            f"{controller}/{scene}: anchor JPEG is not hash-bound")
                else:
                    require(math.isclose(pointgoal_norm(projected), 2.5,
                                         rel_tol=0.0, abs_tol=1e-6),
                            f"{controller}/{scene}: residual norm is not 2.5 m")
                if controller != "navdp":
                    require(plan.get("cec_fallback_context_shadowed") is True,
                            f"{controller}/{scene}: fallback context not shadowed")
                    require(plan.get("cec_controller_seed_consumed") is False,
                            f"{controller}/{scene}: deterministic model claimed RNG")
                    controller_receipt = plan.get(
                        "cec_controller_portability_receipt")
                    require(isinstance(controller_receipt, dict)
                            and controller_receipt.get("controller") == controller,
                            f"{controller}/{scene}: controller receipt is missing")
                else:
                    require(plan.get("cec_controller_seed_consumed") is True,
                            f"{controller}/{scene}: NavDP seed was not consumed")
        role_receipts[role] = {
            "plans": len(plans),
            "proof_sha256_sequence": proofs,
            "anchor_sequence": anchors,
            "plan_file": str(plan_path),
            "plan_file_sha256": sha256_file(plan_path),
        }

    return {
        "controller": controller,
        "scene": scene,
        "episode": episode,
        "run_root": str(root),
        "summary_sha256": sha256_file(summary_path),
        "metric_sha256": sha256_file(metric_path),
        "roles": role_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="CONTROLLER=PATH; repeat for every scene/arm")
    parser.add_argument("--minimum-scenes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    parsed = [parse_run(item) for item in cli.run]
    require({controller for controller, _path in parsed} == set(CONTROLLERS),
            "all headline controllers are required")
    receipts = [audit_run(controller, path) for controller, path in parsed]
    scenes = sorted({receipt["scene"] for receipt in receipts})
    require(len(scenes) >= cli.minimum_scenes,
            "insufficient independent scene clusters")
    for scene in scenes:
        scene_rows = [row for row in receipts if row["scene"] == scene]
        require({row["controller"] for row in scene_rows} == set(CONTROLLERS),
                f"{scene}: controller matrix is incomplete")
        for role in ("novel", "revisit"):
            proof_sequences = {
                tuple(row["roles"][role]["proof_sha256_sequence"])
                for row in scene_rows
            }
            require(len(proof_sequences) == 1,
                    f"{scene}/{role}: CEC proof stream differs by controller")
        anchor_sequences = {
            tuple(row["roles"]["revisit"]["anchor_sequence"])
            for row in scene_rows
        }
        require(len(anchor_sequences) == 1,
                f"{scene}: CEC anchor differs by controller")

    output = {
        "schema": SCHEMA,
        "verified": True,
        "interpretation": (
            "Two-branch controller integration only; short horizons provide "
            "no navigation success-rate evidence."
        ),
        "controllers": list(CONTROLLERS),
        "scenes": scenes,
        "scene_clusters": len(scenes),
        "runs": receipts,
    }
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verified": True,
        "controllers": len(CONTROLLERS),
        "scene_clusters": len(scenes),
        "runs": len(receipts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
