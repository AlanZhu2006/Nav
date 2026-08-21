#!/usr/bin/env python3
"""Independent invariant/latency audit for one all-CEC HPC pilot history."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from audit_cec_controller_portability_smoke import (
    ADAPTERS,
    CONTROLLERS,
    FORBIDDEN_RUNTIME_FIELDS,
    HUB_SCHEMA,
    parse_run,
    pointgoal_norm,
    require,
)


SCHEMA = "cec_controller_portability_pilot_audit_v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
LATENCY_FIELDS = (
    "cec_probe_ms", "cec_certificate_ms", "cec_projection_ms",
    "cec_controller_ms", "cec_depth_sidecar_ms",
    "cec_context_shadow_ms", "cec_total_decision_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def audit_run(controller: str, root: Path) -> dict[str, Any]:
    result_root = root / "result"
    summary_path = result_root / "summary.json"
    require(summary_path.is_file(), f"{controller}: summary missing")
    summary = json.loads(summary_path.read_text())
    require(summary.get("server_backend") == "cec_portability",
            f"{controller}: wrong backend")
    require(summary.get("queries") == 2
            and summary.get("role_counts") == {"novel": 1, "revisit": 1},
            f"{controller}: mixed-role population changed")
    require(summary.get("runtime_role_visibility") == "none",
            f"{controller}: role label leaked")
    require(summary.get("shared_A_all_hashes_ok") is True
            and summary.get("shared_A_total_diffusion_samples") == 0,
            f"{controller}: frozen replay contract failed")
    require(summary.get("metric_depth_sensor_consumed_episodes") == 0,
            f"{controller}: metric depth entered policy")
    require(summary.get("runtime_failure_plans") == 0,
            f"{controller}: runtime failure present")

    role_receipts: dict[str, Any] = {}
    all_latency = {field: [] for field in LATENCY_FIELDS}
    for role in ("novel", "revisit"):
        matches = sorted(result_root.glob(f"*_{role}_plans.json"))
        require(len(matches) == 1, f"{controller}/{role}: plan file missing")
        payload = json.loads(matches[0].read_text())
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{controller}/{role}: role projection seal missing")
        require(not set(payload.get("query_runtime_fields", [])).intersection(
                    FORBIDDEN_RUNTIME_FIELDS),
                f"{controller}/{role}: forbidden runtime field")
        plans = payload.get("query_leg")
        require(isinstance(plans, list) and plans,
                f"{controller}/{role}: no decisions")
        states = []
        proofs = []
        anchors = []
        for plan in plans:
            require(plan.get("cec_portability_schema") == HUB_SCHEMA
                    and plan.get("cec_decision_scope") == "per_action",
                    f"{controller}/{role}: hub contract changed")
            require(plan.get("cec_accept_controller") == controller
                    and plan.get("cec_accept_adapter") == ADAPTERS[controller],
                    f"{controller}/{role}: adapter receipt changed")
            require(plan.get("metric_depth_sensor_consumed") is False,
                    f"{controller}/{role}: sensor audit ambiguous")
            proof = plan.get("cec_proof_sha256")
            require(isinstance(proof, str) and SHA256.fullmatch(proof),
                    f"{controller}/{role}: proof digest invalid")
            proofs.append(proof)
            projected = plan.get("cec_projected_goal")
            require(isinstance(projected, dict),
                    f"{controller}/{role}: projected goal missing")
            takeover = plan.get("cec_takeover")
            require(isinstance(takeover, bool),
                    f"{controller}/{role}: action state ambiguous")
            states.append("takeover" if takeover else "fallback")
            if takeover:
                require(plan.get("cec_action_state") == "takeover",
                        f"{controller}/{role}: takeover state mismatch")
                anchor = plan.get("cec_selected_anchor")
                require(isinstance(anchor, int),
                        f"{controller}/{role}: anchor missing")
                anchors.append(anchor)
                if controller == "vint":
                    require(SHA256.fullmatch(str(
                                projected.get("cec_anchor_sha256", ""))),
                            f"{controller}/{role}: anchor is not hash-bound")
                else:
                    require(math.isclose(pointgoal_norm(projected), 2.5,
                                         rel_tol=0.0, abs_tol=1e-6),
                            f"{controller}/{role}: residual is not 2.5 m")
                if controller == "navdp":
                    require(plan.get("cec_controller_seed_consumed") is True,
                            f"{controller}/{role}: diffusion seed not consumed")
                else:
                    require(plan.get("cec_controller_seed_consumed") is False,
                            f"{controller}/{role}: deterministic model claimed RNG")
                    require(plan.get("cec_fallback_context_shadowed") is True,
                            f"{controller}/{role}: fallback context not shadowed")
            else:
                require(plan.get("cec_action_state") == "fallback"
                        and projected == {"fallback_this_action": True},
                        f"{controller}/{role}: exact fallback changed")
                require(plan.get("cec_controller_seed_consumed") is True,
                        f"{controller}/{role}: fallback seed not consumed")
                if controller == "vint":
                    require(plan.get("cec_alternate_context_shadowed") is True,
                            f"{controller}/{role}: ViNT context not shadowed")
            for field in LATENCY_FIELDS:
                value = plan.get(field)
                if field == "cec_depth_sidecar_ms" and value is None:
                    continue
                require(isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        and float(value) >= 0.0,
                        f"{controller}/{role}: invalid {field}")
                all_latency[field].append(float(value))
        role_receipts[role] = {
            "plans": len(plans),
            "takeover_plans": states.count("takeover"),
            "fallback_plans": states.count("fallback"),
            "state_transitions": sum(
                left != right for left, right in zip(states, states[1:])
            ),
            "first_proof_sha256": proofs[0],
            "first_anchor": anchors[0] if anchors else None,
            "plan_sha256": sha256_file(matches[0]),
        }
    latency = {
        field: {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "p90": percentile(values, 0.90),
            "max": max(values) if values else None,
        }
        for field, values in all_latency.items()
    }
    return {
        "controller": controller,
        "scene": summary["scene"],
        "run_root": str(root),
        "roles": role_receipts,
        "latency_ms": latency,
        "summary_sha256": sha256_file(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    parsed = [parse_run(item) for item in cli.run]
    require({controller for controller, _path in parsed} == set(CONTROLLERS),
            "all four controllers are required")
    require(len(parsed) == len(CONTROLLERS),
            "one run per controller is required")
    runs = [audit_run(controller, path) for controller, path in parsed]
    require(len({run["scene"] for run in runs}) == 1,
            "pilot audit accepts one scene at a time")
    for role in ("novel", "revisit"):
        require(len({run["roles"][role]["first_proof_sha256"]
                     for run in runs}) == 1,
                f"{role}: first-action CEC proof differs across arms")
        first_anchors = {
            run["roles"][role]["first_anchor"] for run in runs
        }
        require(len(first_anchors) == 1,
                f"{role}: first-action anchor differs across arms")
    output = {
        "schema": SCHEMA,
        "verified": True,
        "interpretation": (
            "HPC latency/failure pilot only; do not interpret as statistical "
            "navigation performance."
        ),
        "scene": runs[0]["scene"],
        "runs": runs,
    }
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified": True, "scene": output["scene"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
