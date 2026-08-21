"""Pure contracts for HM3D actual-online full-monocular mixed roles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from final14_mono_factorial import audit_depth_plans
    from mdtec_raw_depth_gate_d import (
        audit_arm_contract,
        paired_contrast,
        require,
        scene_cluster_interval,
    )
except ImportError:
    from MemNavData.final14_mono_factorial import audit_depth_plans
    from MemNavData.mdtec_raw_depth_gate_d import (
        audit_arm_contract,
        paired_contrast,
        require,
        scene_cluster_interval,
    )


ARMS = ("mono_native", "mono_raw_fixed", "mono_cec")
DEPTH_SOURCE = {arm: "monocular_sidecar" for arm in ARMS}
HYBRID_ROUTE = {
    "mono_native": "native_sidecar",
    "mono_raw_fixed": "phase",
    "mono_cec": "certified_relocalization",
}
REVISIT_ADAPTER = {
    "mono_native": "legacy_metric",
    "mono_raw_fixed": "raw_fixed_bearing_v1",
    "mono_cec": "verified_bearing_v1",
}
EVALUATOR_ARM = {
    "mono_native": "native_sidecar",
    "mono_raw_fixed": "raw_fixed_bearing",
    "mono_cec": "certified",
}
PRIMARY_CONTRASTS = (
    ("mono_cec", "mono_native"),
    ("mono_cec", "mono_raw_fixed"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_parent_manifest(
    protocol: dict[str, Any], protocol_path: Path, parent_path: Path,
) -> tuple[dict[str, Any], str]:
    """Verify either a pre-frozen parent hash or a sealed generated receipt."""

    digest = sha256_file(parent_path)
    expected = protocol["dataset"].get("parent_manifest_sha256")
    if expected is not None:
        require(digest == expected, "parent HM3D manifest changed")
    else:
        require(protocol.get("schema_version") ==
                "hm3d_fresh_fullmono_mixed_role_protocol_v1_20260820",
                "dynamic parent receipt is not authorized by this protocol")
        receipt = parent_path.with_name(parent_path.name + ".sha256")
        require(receipt.is_file(), "sealed parent manifest receipt missing")
        fields = receipt.read_text().split()
        require(fields and fields[0] == digest,
                "sealed parent manifest receipt changed")
    parent = json.loads(parent_path.read_text())
    if expected is None:
        require(parent.get("protocol_sha256") == sha256_file(protocol_path),
                "generated parent references another protocol")
        require(parent.get("query_outcomes_read") is False,
                "generated parent read query outcomes")
        require(parent.get("fresh_scene_generalization") is True,
                "generated parent is not the frozen fresh population")
    return parent, digest


def resolve_parent_scene(
    protocol: dict[str, Any], parent: dict[str, Any], index: int,
) -> tuple[dict[str, Any], str]:
    scenes = protocol["dataset"]["scenes"]
    require(0 <= index < len(scenes), "scene index out of range")
    spec = scenes[index]
    require(int(spec["rank"]) == index, "scene ranks changed")
    scene = str(spec["scene_id"])
    if "parent_index" in spec:
        require(parent["scenes"][int(spec["parent_index"])] == scene,
                f"{scene}: parent scene identity changed")
    else:
        require(parent["scenes"][index] == scene,
                f"{scene}: generated parent scene identity changed")
        require(parent["scene_specs"][index] == spec,
                f"{scene}: generated parent scene specification changed")
    return spec, scene


def expected_parent_source_count(
    protocol: dict[str, Any], parent: dict[str, Any],
) -> int:
    frozen = protocol["dataset"].get("source_episode_count")
    return int(frozen if frozen is not None else parent["episode_count"])


def rotated_arm_order(history_index: int) -> tuple[str, ...]:
    offset = int(history_index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def audit_goal_a_plans(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the already-frozen first-40 raw-depth audit to Goal-A."""

    active = any(
        isinstance(plan.get("monocular_depth_receipt"), dict)
        and int(plan["monocular_depth_receipt"].get("frame_index", -1)) >= 40
        for plan in plans
    )
    return audit_arm_contract("raw_first40", {
        "plans": plans,
        "navdp_depth_source": "monocular_sidecar",
        "metric_depth_sensor_consumed_any": any(
            plan.get("metric_depth_sensor_consumed") is True for plan in plans
        ),
        "monocular_frame40_survived": active,
    })


def audit_query_arm(arm: str, plans: list[dict[str, Any]]) -> dict[str, Any]:
    require(arm in ARMS, f"unknown full-mono arm {arm!r}")
    return audit_depth_plans(arm, plans)


__all__ = [
    "ARMS",
    "DEPTH_SOURCE",
    "EVALUATOR_ARM",
    "HYBRID_ROUTE",
    "PRIMARY_CONTRASTS",
    "REVISIT_ADAPTER",
    "audit_goal_a_plans",
    "audit_query_arm",
    "bind_parent_manifest",
    "expected_parent_source_count",
    "paired_contrast",
    "require",
    "resolve_parent_scene",
    "rotated_arm_order",
    "scene_cluster_interval",
    "sha256_file",
]
