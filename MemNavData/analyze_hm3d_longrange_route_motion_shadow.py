#!/usr/bin/env python3
"""Aggregate the frozen consumed same-edge route-motion shadow diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


SCHEMA = "hm3d_longrange_route_motion_shadow_result_v1_20260903"
CELL_SCHEMA = "hm3d_longrange_route_motion_shadow_history_v1_20260903"
PROTOCOL_SCHEMA = (
    "hm3d_longrange_route_motion_shadow_protocol_v1_20260903"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()

    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "diagnostic protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "diagnostic protocol schema changed")
    expected = list(protocol["selected_history_indices"])
    rows = []
    for index in expected:
        matches = list((args.run_root / "evaluation").glob(
            f"{index:03d}_*/completion.json"))
        require(len(matches) == 1,
                f"history {index} completion is missing or ambiguous")
        path = matches[0]
        sidecar = path.with_name("completion.json.sha256")
        require(sidecar.is_file(), f"history {index} sidecar missing")
        words = sidecar.read_text().strip().split()
        require(len(words) == 2 and words[0] == sha256_file(path)
                and words[1] == "completion.json",
                f"history {index} completion hash changed")
        row = json.loads(path.read_text())
        require(row.get("schema_version") == CELL_SCHEMA
                and row.get("history_index") == index,
                f"history {index} completion contract changed")
        require(row.get("protocol_sha256") == args.expected_protocol_sha256,
                f"history {index} protocol binding changed")
        require(row.get("shadow_control_authority") is False,
                f"history {index} shadow reached control")
        rows.append(row)

    reproduced = [row for row in rows if row.get("failure_edge") is not None]
    recovered = [
        row for row in reproduced
        if row["failure_edge"]["unfiltered_pnp_shadow"]
        ["local_motion_validity"]["accepted"] is True
    ]
    accurate = []
    translation_errors = []
    yaw_errors = []
    for row in recovered:
        error = row["failure_edge"].get("unfiltered_shadow_pose_error")
        if not isinstance(error, dict):
            continue
        translation = float(error["translation_vector_error_m"])
        yaw = float(error["yaw_error_deg"])
        if math.isfinite(translation) and math.isfinite(yaw):
            translation_errors.append(translation)
            yaw_errors.append(yaw)
            if translation <= 0.10 and yaw <= 5.0:
                accurate.append(row)

    all_reproduced = len(reproduced) == len(expected)
    median_translation = (
        None if not translation_errors else float(median(translation_errors)))
    median_yaw = None if not yaw_errors else float(median(yaw_errors))
    supports = bool(
        all_reproduced
        and len(recovered) >= 10
        and median_translation is not None
        and median_translation <= 0.10
        and median_yaw is not None
        and median_yaw <= 5.0
    )
    result = {
        "schema_version": SCHEMA,
        "claim_scope": "consumed same-edge mechanism diagnostic only",
        "protocol_sha256": args.expected_protocol_sha256,
        "expected_histories": len(expected),
        "completed_histories": len(rows),
        "primary_stop_reproduced": len(reproduced),
        "unfiltered_shadow_valid": len(recovered),
        "unfiltered_shadow_accurate_per_edge": len(accurate),
        "unfiltered_shadow_translation_vector_error_median": (
            median_translation),
        "unfiltered_shadow_yaw_error_deg_median": median_yaw,
        "supports_epipolar_degeneracy": supports,
        "decision": (
            "replace_adjacent_fundamental_prefilter_with_single_direct_pnp_"
            "model_then_test_consumed_closed_loop"
            if supports
            else "simple_epipolar_explanation_not_established"
        ),
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
        "rows": [{
            "history_index": row["history_index"],
            "scene": row["scene"],
            "primary_stop_reproduced": row["failure_edge"] is not None,
            "unfiltered_shadow_valid": (
                None if row["failure_edge"] is None
                else row["failure_edge"]["unfiltered_pnp_shadow"]
                ["local_motion_validity"]["accepted"]
            ),
            "pose_error": (
                None if row["failure_edge"] is None
                else row["failure_edge"].get(
                    "unfiltered_shadow_pose_error"))
        } for row in rows],
    }
    output = args.run_root / "result" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "diagnostic summary already exists")
    encoded = (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode()
    output.write_bytes(encoded)
    output.with_name("summary.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  summary.json\n")
    print(json.dumps({
        "status": "complete",
        "primary_stop_reproduced": len(reproduced),
        "unfiltered_shadow_valid": len(recovered),
        "supports_epipolar_degeneracy": supports,
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
