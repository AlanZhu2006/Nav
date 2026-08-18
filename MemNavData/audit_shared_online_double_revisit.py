#!/usr/bin/env python3
"""Independent, renderer-free audit of shared-online double-Revisit assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


V1_SCHEMA = "shared_online_double_revisit_v1_20260812"
V2_SCHEMA = "shared_online_double_revisit_v2_route_negative_20260812"
EXPECTED_SCHEMAS = {V1_SCHEMA, V2_SCHEMA}
V0_NAME = "v0_exact_online_frame"
V1_NAME = "v1_controlled_pose_perturbation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_curve(goal: dict, expected_length: int, label: str) -> None:
    curve = goal.get("covis_curve")
    require(
        isinstance(curve, list) and len(curve) == expected_length,
        f"{label}: co-visibility curve length mismatch",
    )
    require(
        all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
            for value in curve
        ),
        f"{label}: invalid co-visibility value",
    )


def audit_reference_tail(
    episode_dir: Path,
    name: str,
    variant: dict,
    contract: dict,
) -> dict:
    label = f"{episode_dir.parent.name}/{episode_dir.name}/{name}/reference-tail"
    tail = variant.get("reference_path_c_tail")
    require(isinstance(tail, dict), f"{label}: missing reference-path audit")
    curve = tail.get("curve")
    require(isinstance(curve, list) and curve, f"{label}: empty reference curve")
    values = [float(value) for value in curve]
    require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values),
        f"{label}: invalid reference co-visibility value",
    )
    argmax = max(range(len(values)), key=values.__getitem__)
    maximum = values[argmax]
    allowed = float(contract["reference_path_c_tail_max_covis"])
    require(int(tail["frames"]) == len(values), f"{label}: frame count mismatch")
    require(
        int(tail["argmax_reference_frame"]) == argmax,
        f"{label}: argmax mismatch",
    )
    require(
        math.isclose(float(tail["maximum_covisibility"]), maximum, abs_tol=1e-12),
        f"{label}: maximum mismatch",
    )
    require(
        math.isclose(float(tail["maximum_allowed"]), allowed, abs_tol=1e-12),
        f"{label}: stored limit differs from contract",
    )
    require(maximum <= allowed, f"{label}: route-negative contract violated")
    require(
        bool(tail.get("closed_loop_recheck_required")) is True,
        f"{label}: closed-loop recheck is not required",
    )
    return {
        "maximum_covisibility": maximum,
        "argmax_reference_frame": argmax,
        "frames": len(values),
    }
def audit_variant(
    episode_dir: Path,
    source_dir: Path,
    name: str,
    variant: dict,
    contract: dict,
    online_steps: int,
    schema: str,
) -> dict:
    goals = variant["goals"]
    assets = variant["assets"]
    frames = {}
    for role in ("B", "C"):
        label = f"{episode_dir.parent.name}/{episode_dir.name}/{name}/{role}"
        goal = goals[role]
        asset = assets[role]
        finite_curve(goal, online_steps, label)
        rgb = episode_dir / name / asset["rgb"]
        depth = episode_dir / name / asset["depth"]
        require(rgb.is_file() and depth.is_file(), f"{label}: missing goal asset")
        require(
            sha256_file(rgb) == asset["rgb_sha256"],
            f"{label}: RGB hash mismatch",
        )
        require(
            sha256_file(depth) == asset["depth_sha256"],
            f"{label}: depth hash mismatch",
        )
        frame = int(goal["source_online_frame"])
        require(
            int(goal["source_online_step"]) == frame,
            f"{label}: source frame/step mismatch",
        )
        require(
            int(goal["eligible_online_a_frame_floor"])
            == int(contract["minimum_eligible_online_frame"]),
            f"{label}: eligible memory floor mismatch",
        )
        frames[role] = int(goal["max_online_a_covis_frame"])
        if name == V0_NAME:
            source_rgb = source_dir / "rgb" / f"{frame:06d}.jpg"
            source_depth = source_dir / "depth" / f"{frame:06d}.png"
            require(
                rgb.read_bytes() == source_rgb.read_bytes(),
                f"{label}: V0 RGB is not the exact online frame",
            )
            require(
                depth.read_bytes() == source_depth.read_bytes(),
                f"{label}: V0 depth is not the exact online frame",
            )
            require(
                float(goal["source_frame_covis"])
                >= float(contract["v0_min_self_covis"]),
                f"{label}: V0 self co-visibility below contract",
            )
            require(
                float(goal["translation_from_source_m"]) == 0.0
                and float(goal["yaw_delta_from_source_deg"]) == 0.0
                and float(goal["pixel_mae_from_source"]) == 0.0,
                f"{label}: V0 contains a pose/image perturbation",
            )
        else:
            v0_rgb = episode_dir / V0_NAME / f"goal_{role.lower()}.jpg"
            require(
                rgb.read_bytes() != v0_rgb.read_bytes(),
                f"{label}: V1 RGB equals V0",
            )
            require(
                float(contract["v1_min_translation_m"])
                <= float(goal["translation_from_source_m"])
                <= float(contract["v1_max_translation_m"]),
                f"{label}: V1 translation outside contract",
            )
            require(
                float(contract["v1_min_yaw_delta_deg"])
                <= float(goal["yaw_delta_from_source_deg"])
                <= float(contract["v1_max_yaw_delta_deg"]),
                f"{label}: V1 yaw delta outside contract",
            )
            require(
                float(goal["pixel_mae_from_source"])
                >= float(contract["v1_min_pixel_mae"]),
                f"{label}: V1 pixel difference below contract",
            )
            require(
                float(goal["source_frame_covis"])
                >= float(contract["v1_min_source_frame_covis"]),
                f"{label}: V1 source-frame co-visibility below contract",
            )
            require(
                float(contract["v1_min_max_online_a_covis"])
                <= float(goal["max_online_a_covis"])
                <= float(contract["v1_max_max_online_a_covis"]),
                f"{label}: V1 best co-visibility outside contract",
            )
            require(
                abs(frames[role] - frame)
                <= int(contract["v1_max_argmax_gap_frames"]),
                f"{label}: V1 best anchor is not local to source frame",
            )

    measured_gap = abs(frames["B"] - frames["C"])
    require(
        measured_gap == int(variant["anchor_argmax_gap_frames"]),
        f"{episode_dir}: stored anchor gap mismatch",
    )
    require(
        measured_gap >= int(contract["minimum_anchor_gap_frames"]),
        f"{episode_dir}: B/C anchors are insufficiently separated",
    )
    geodesics = variant["leg_geodesics_m"]
    require(
        all(
            math.isfinite(float(value))
            and float(value) >= float(contract["minimum_leg_geodesic_m"])
            for value in geodesics.values()
        ),
        f"{episode_dir}: leg geodesic below contract",
    )
    result = {
        "anchor_gap_frames": measured_gap,
        "minimum_leg_geodesic_m": min(
            float(value) for value in geodesics.values()
        ),
    }
    if schema == V2_SCHEMA:
        result["reference_path_c_tail"] = audit_reference_tail(
            episode_dir, name, variant, contract
        )
    return result


def audit(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    digest_path = root / "manifest.json.sha256"
    require(manifest_path.is_file() and digest_path.is_file(), "manifest missing")
    manifest_digest = sha256_file(manifest_path)
    require(
        digest_path.read_text().split()[0] == manifest_digest,
        "manifest sidecar digest mismatch",
    )
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema_version")
    require(schema in EXPECTED_SCHEMAS, "schema mismatch")
    contract = manifest["contract"]
    rows = []
    seen = set()
    for episode in manifest["episodes"]:
        key = (str(episode["scene"]), str(episode["episode"]))
        require(key not in seen, f"duplicate episode {key}")
        seen.add(key)
        episode_dir = root / key[0] / key[1]
        benchmark_path = episode_dir / "benchmark.json"
        benchmark = json.loads(benchmark_path.read_text())
        manifest_copy = dict(episode)
        stored_benchmark_digest = manifest_copy.pop("benchmark_sha256")
        require(benchmark == manifest_copy, f"{key}: manifest/benchmark mismatch")
        require(
            sha256_file(benchmark_path) == stored_benchmark_digest,
            f"{key}: benchmark hash mismatch",
        )
        source_dir = Path(episode["source_online_episode"])
        require(source_dir.is_dir(), f"{key}: source online episode missing")
        require(
            sha256_file(source_dir / "receipt.json")
            == episode["source_online_receipt_sha256"],
            f"{key}: source receipt hash mismatch",
        )
        require(
            sha256_file(source_dir / "online_a_trace.json")
            == episode["source_online_trace_sha256"],
            f"{key}: source trace hash mismatch",
        )
        assignment = episode["source_anchor_assignment"]
        if schema == V1_SCHEMA:
            source_frames = {
                "B": int(assignment["B_later_frame"]),
                "C": int(assignment["C_earlier_frame"]),
            }
            require(
                source_frames["B"] > source_frames["C"],
                f"{key}: B/C source ordering is inverted",
            )
        else:
            source_frames = {
                "B": int(assignment["B_source_frame"]),
                "C": int(assignment["C_source_frame"]),
            }
        require(
            abs(source_frames["B"] - source_frames["C"])
            == int(assignment["temporal_gap_frames"]),
            f"{key}: stored source temporal gap mismatch",
        )
        require(
            abs(source_frames["B"] - source_frames["C"])
            >= int(contract["minimum_anchor_gap_frames"]),
            f"{key}: source anchors violate temporal gap",
        )
        for name in (V0_NAME, V1_NAME):
            goals = episode["variants"][name]["goals"]
            require(
                all(
                    int(goals[role]["source_online_frame"])
                    == source_frames[role]
                    for role in ("B", "C")
                ),
                f"{key}/{name}: variant source frames differ from assignment",
            )
        audits = {
            name: audit_variant(
                episode_dir,
                source_dir,
                name,
                episode["variants"][name],
                contract,
                int(episode["online_a_steps"]),
                schema,
            )
            for name in (V0_NAME, V1_NAME)
        }
        rows.append({"scene": key[0], "episode": key[1], **audits})
    require(bool(rows), "manifest contains no episodes")
    return {
        "ok": True,
        "schema_version": schema,
        "episodes": len(rows),
        "scenes": len({row["scene"] for row in rows}),
        "manifest_sha256": manifest_digest,
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
