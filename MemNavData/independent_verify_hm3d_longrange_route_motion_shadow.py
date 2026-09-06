#!/usr/bin/env python3
"""Independently verify the consumed route-motion shadow diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


CELL_SCHEMA = "hm3d_longrange_route_motion_shadow_history_v1_20260903"
PROTOCOL_SCHEMA = "hm3d_longrange_route_motion_shadow_protocol_v1_20260903"
SUMMARY_SCHEMA = "hm3d_longrange_route_motion_shadow_result_v1_20260903"
VERIFY_SCHEMA = (
    "hm3d_longrange_route_motion_shadow_independent_verification_v1_20260903"
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
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require(sha256_file(args.protocol) == args.expected_protocol_sha256,
            "diagnostic protocol changed")
    protocol = json.loads(args.protocol.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "diagnostic protocol schema changed")
    expected = tuple(int(value)
                     for value in protocol["selected_history_indices"])
    require(len(expected) == 14 and len(set(expected)) == len(expected),
            "frozen diagnostic population changed")

    completed: list[int] = []
    valid: list[int] = []
    translation_errors: list[float] = []
    yaw_errors: list[float] = []
    rows = []
    for index in expected:
        matches = sorted((args.run_root / "evaluation").glob(
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
        require(row.get("schema_version") == CELL_SCHEMA,
                f"history {index} completion schema changed")
        require(row.get("history_index") == index,
                f"history {index} identity changed")
        require(row.get("protocol_sha256")
                == args.expected_protocol_sha256,
                f"history {index} protocol binding changed")
        require(row.get("shadow_control_authority") is False,
                f"history {index} shadow reached control")
        failure = row.get("failure_edge")
        require(isinstance(failure, dict)
                and failure.get("primary_stop_reproduced") is True,
                f"history {index} did not reproduce the selected stop")
        completed.append(index)
        shadow = failure["unfiltered_pnp_shadow"]
        is_valid = bool(shadow["local_motion_validity"]["accepted"])
        if is_valid:
            valid.append(index)
            error = failure.get("unfiltered_shadow_pose_error")
            if isinstance(error, dict):
                translation = float(error["translation_vector_error_m"])
                yaw = float(error["yaw_error_deg"])
                if math.isfinite(translation) and math.isfinite(yaw):
                    translation_errors.append(translation)
                    yaw_errors.append(yaw)
        rows.append(row)

    median_translation = (None if not translation_errors
                          else float(median(translation_errors)))
    median_yaw = None if not yaw_errors else float(median(yaw_errors))
    supports = bool(
        len(completed) == len(expected)
        and len(valid) >= 10
        and median_translation is not None
        and median_translation <= 0.10
        and median_yaw is not None
        and median_yaw <= 5.0
    )

    summary_sidecar = args.summary.with_name("summary.json.sha256")
    require(summary_sidecar.is_file(), "summary sidecar missing")
    words = summary_sidecar.read_text().strip().split()
    summary_sha = sha256_file(args.summary)
    require(len(words) == 2 and words[0] == summary_sha
            and words[1] == "summary.json", "summary hash changed")
    summary = json.loads(args.summary.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "summary schema changed")
    independent = {
        "expected_histories": len(expected),
        "completed_histories": len(completed),
        "primary_stop_reproduced": len(completed),
        "unfiltered_shadow_valid": len(valid),
        "unfiltered_shadow_translation_vector_error_median": (
            median_translation),
        "unfiltered_shadow_yaw_error_deg_median": median_yaw,
        "supports_epipolar_degeneracy": supports,
    }
    for key, value in independent.items():
        require(summary.get(key) == value,
                f"summary disagrees with independent {key}")

    receipt = {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "claim_scope": "consumed same-edge mechanism diagnostic only",
        "protocol_sha256": args.expected_protocol_sha256,
        "summary_sha256": summary_sha,
        "shadow_control_authority": False,
        "navigation_claim_allowed": False,
        "fresh_confirmation_required": True,
        "supports_epipolar_degeneracy": supports,
        "independent_counts": independent,
        "completion_sha256": {
            str(row["history_index"]): sha256_file(
                next((args.run_root / "evaluation").glob(
                    f"{row['history_index']:03d}_*/completion.json")))
            for row in rows
        },
    }
    require(not args.out.exists(), "verification output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {args.out.name}\n")
    print(json.dumps({
        "status": "complete",
        "verified": True,
        "supports_epipolar_degeneracy": supports,
        "output": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
