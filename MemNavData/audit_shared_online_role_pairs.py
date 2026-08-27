#!/usr/bin/env python3
"""Independent renderer-free audit for matched online-A Novel/Revisit pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from shared_online_role_pair_contract import runtime_query, validate_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summary(values: list[float]) -> dict:
    return {
        "minimum": min(values) if values else None,
        "median": percentile(values, 0.5),
        "maximum": max(values) if values else None,
    }


def finite_curve(query: dict, expected_length: int, label: str) -> None:
    curve = query.get("covis_curve")
    require(
        isinstance(curve, list) and len(curve) == expected_length,
        f"{label}: co-visibility curve length mismatch",
    )
    values = [float(value) for value in curve]
    require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values),
        f"{label}: invalid co-visibility curve",
    )
    floor = int(query.get("eligible_online_a_frame_floor", 0))
    require(
        0 <= floor < expected_length,
        f"{label}: eligible online-A frame floor is invalid",
    )
    eligible_values = values[floor:]
    maximum = max(eligible_values) if eligible_values else 0.0
    require(
        math.isclose(
            maximum,
            float(query["max_online_a_covis"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{label}: co-visibility maximum mismatch",
    )
    expected_argmax = (
        floor + eligible_values.index(maximum) if eligible_values else None
    )
    require(
        query.get("max_online_a_covis_frame") == expected_argmax,
        f"{label}: co-visibility argmax mismatch",
    )


def audit(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    digest_path = root / "manifest.json.sha256"
    require(manifest_path.is_file(), "manifest is missing")
    require(digest_path.is_file(), "manifest digest sidecar is missing")
    manifest_sha = sha256_file(manifest_path)
    require(
        digest_path.read_text().split()[0] == manifest_sha,
        "manifest digest sidecar mismatch",
    )
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)

    rows = []
    identities = set()
    for episode in manifest["episodes"]:
        identity = (str(episode["scene"]), str(episode["episode"]))
        require(identity not in identities, f"duplicate episode {identity}")
        identities.add(identity)
        episode_root = root / identity[0] / identity[1]
        sidecar_path = episode_root / "role_pairs.json"
        require(sidecar_path.is_file(), f"{identity}: role-pair sidecar missing")
        manifest_copy = dict(episode)
        stored_sidecar_sha = manifest_copy.pop("role_pairs_sha256")
        require(
            sha256_file(sidecar_path) == stored_sidecar_sha,
            f"{identity}: role-pair sidecar hash mismatch",
        )
        require(
            json.loads(sidecar_path.read_text()) == manifest_copy,
            f"{identity}: sidecar/manifest contents differ",
        )
        online_root = Path(episode["online_a_episode"])
        require(online_root.is_dir(), f"{identity}: online-A source missing")
        require(
            sha256_file(online_root / "receipt.json")
            == episode["online_a_receipt_sha256"],
            f"{identity}: online-A receipt hash mismatch",
        )
        require(
            sha256_file(online_root / "online_a_trace.json")
            == episode["online_a_trace_sha256"],
            f"{identity}: online-A trace hash mismatch",
        )
        trace = json.loads((online_root / "online_a_trace.json").read_text())
        require(
            len(trace["poses"]) == int(episode["online_a_steps"]),
            f"{identity}: online-A trace length mismatch",
        )
        require(
            trace["end_position"] == episode["online_a_endpoint"]["floor_position"]
            and float(trace["end_yaw"])
            == float(episode["online_a_endpoint"]["yaw_rad"]),
            f"{identity}: online-A endpoint mismatch",
        )
        for pair in episode["pairs"]:
            assets = {}
            roles = {}
            for query in pair["queries"]:
                role = str(query["analysis_role"])
                roles[role] = query
                label = f"{identity[0]}/{identity[1]}/{pair['pair_id']}/{role}"
                rgb = episode_root / query["goal_rgb"]
                depth = episode_root / query["goal_depth"]
                require(rgb.is_file() and depth.is_file(), f"{label}: goal missing")
                require(
                    sha256_file(rgb) == query["goal_rgb_sha256"],
                    f"{label}: RGB hash mismatch",
                )
                require(
                    sha256_file(depth) == query["goal_depth_sha256"],
                    f"{label}: depth hash mismatch",
                )
                finite_curve(query, int(episode["online_a_steps"]), label)
                projected = runtime_query(query)
                require(
                    "analysis_role" not in projected
                    and "max_online_a_covis" not in projected
                    and "covis_curve" not in projected,
                    f"{label}: runtime projection leaked analysis labels",
                )
                assets[role] = (rgb.read_bytes(), depth.read_bytes())
            require(
                assets["novel"] != assets["revisit"],
                f"{identity}/{pair['pair_id']}: role assets are identical",
            )
            rows.append({
                "scene": identity[0],
                "episode": identity[1],
                "pair_id": pair["pair_id"],
                "role_distance_error_m": float(pair["role_distance_error_m"]),
                "role_initial_path_bearing_error_deg": float(
                    pair["role_initial_path_bearing_error_deg"]
                ),
                "novel_max_online_a_covis": float(
                    roles["novel"]["max_online_a_covis"]
                ),
                "revisit_max_online_a_covis": float(
                    roles["revisit"]["max_online_a_covis"]
                ),
                "revisit_source": roles["revisit"].get(
                    "source_controlled_revisit_role"
                ),
                "novel_sampling_attempts": int(
                    roles["novel"]["sampling_diagnostics"]["attempts"]
                ),
            })
    require(bool(rows), "manifest contains no query pairs")
    return {
        "ok": True,
        "manifest_sha256": manifest_sha,
        "episodes": len(manifest["episodes"]),
        "scenes": len({row["scene"] for row in rows}),
        "pairs": len(rows),
        "queries": 2 * len(rows),
        "runtime_role_visibility": manifest["contract"][
            "runtime_role_visibility"
        ],
        "role_distance_error_m": summary(
            [row["role_distance_error_m"] for row in rows]
        ),
        "role_initial_path_bearing_error_deg": summary(
            [row["role_initial_path_bearing_error_deg"] for row in rows]
        ),
        "novel_max_online_a_covis": summary(
            [row["novel_max_online_a_covis"] for row in rows]
        ),
        "revisit_max_online_a_covis": summary(
            [row["revisit_max_online_a_covis"] for row in rows]
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
