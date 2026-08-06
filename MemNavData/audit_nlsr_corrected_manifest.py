#!/usr/bin/env python3
"""Final cross-artifact audit for the corrected 50-scene NLSR manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from MemNavData.flow_cache_routing import (
        FlowRoutingError,
        load_route_registry,
    )
except ImportError:
    from flow_cache_routing import (  # type: ignore
        FlowRoutingError,
        load_route_registry,
    )


SCHEMA_VERSION = "nlsr_corrected_manifest_audit_v1"
EXPECTED_SPLIT_SHA = "97309c183e25cb3dd65472908748d55a94798a636db6157ab6fe120fca05cf7a"
EXPECTED_SUMMARY = {
    "scene_count": 50,
    "episode_count": 100,
    "sample_count": 400,
    "missing_flow_cache_file_count": 0,
    "all_flow_caches_complete": True,
}
EXPECTED_YMJ = {
    "episode_0000": {"n_frames": 871, "switches": [218, 548]},
    "episode_0001": {"n_frames": 591, "switches": [192, 296]},
}
EXPECTED_MANIFEST_SCHEMA = "nlsr_v2_expert_candidate_manifest_v2"


class ManifestAuditError(RuntimeError):
    """The final manifest does not close the audited raw/flow dependency chain."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checked(path: Path, expected_status: str | None = None) -> tuple[dict, str]:
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ManifestAuditError(f"artifact pair is absent: {path} / {sidecar}")
    digest = sha256_file(path)
    words = sidecar.read_text(encoding="ascii").split()
    if not words or words[0] != digest:
        raise ManifestAuditError(f"SHA sidecar mismatch: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestAuditError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(record, dict):
        raise ManifestAuditError(f"JSON artifact is not an object: {path}")
    if expected_status is not None and record.get("status") != expected_status:
        raise ManifestAuditError(
            f"artifact status mismatch at {path}: {record.get('status')}"
        )
    return record, digest


def _write_new(path: Path, record: dict) -> str:
    payload = (json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{path}.sha256")
    if path.exists() or sidecar.exists():
        raise ManifestAuditError(f"output already exists: {path} / {sidecar}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--flow-provenance", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-episode-root", required=True)
    parser.add_argument("--expected-flow-route-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, manifest_sha = _load_checked(args.manifest)
    raw, raw_sha = _load_checked(
        args.raw_audit, "audited_historical_summary_match",
    )
    flow, flow_sha = _load_checked(
        args.flow_provenance, "flow_routes_audited",
    )

    try:
        route_registry = load_route_registry(args.flow_provenance, flow_sha)
    except FlowRoutingError as exc:
        raise ManifestAuditError(
            f"flow route provenance is not consumable: {exc}") from exc

    if manifest.get("schema_version") != EXPECTED_MANIFEST_SCHEMA:
        raise ManifestAuditError("manifest is not the routed v2 schema")
    if manifest.get("summary") != EXPECTED_SUMMARY:
        raise ManifestAuditError(
            f"manifest summary mismatch: {manifest.get('summary')} != {EXPECTED_SUMMARY}"
        )
    if manifest.get("missing_flow_caches") != []:
        raise ManifestAuditError("manifest still records missing flow caches")
    split = manifest.get("split")
    if not isinstance(split, dict) or split.get("sha256") != EXPECTED_SPLIT_SHA:
        raise ManifestAuditError("manifest split SHA mismatch")
    roots = manifest.get("input_roots")
    if not isinstance(roots, dict):
        raise ManifestAuditError("manifest input_roots is absent")
    if roots.get("episode_root") != args.expected_episode_root:
        raise ManifestAuditError(
            f"manifest episode root mismatch: {roots.get('episode_root')}"
        )
    if "flow_cache_root" in roots:
        raise ManifestAuditError("routed manifest must not claim one flow root")
    if manifest.get("flow_cache_routing") != route_registry.manifest_record():
        raise ManifestAuditError(
            "manifest routing record differs from the pinned route artifact")

    scenes = manifest.get("scenes")
    samples = manifest.get("samples")
    if not isinstance(scenes, list) or not isinstance(samples, list):
        raise ManifestAuditError("manifest scenes/samples are malformed")
    role_counts = {"train": 0, "development": 0}
    ymj = None
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ManifestAuditError("manifest scene is not an object")
        role = scene.get("split_role")
        if role not in role_counts:
            raise ManifestAuditError(f"forbidden scene role: {role}")
        role_counts[role] += 1
        selected = scene.get("selected_episodes")
        if not isinstance(selected, list) or len(selected) != 2:
            raise ManifestAuditError(
                f"scene does not have exactly two selected episodes: {scene.get('scene')}"
            )
        if not all(isinstance(episode, dict) for episode in selected):
            raise ManifestAuditError(
                f"scene selected episode is malformed: {scene.get('scene')}"
            )
        if any(not episode.get("flow_cache", {}).get("complete")
               for episode in selected):
            raise ManifestAuditError(
                f"scene has incomplete selected flow cache: {scene.get('scene')}"
            )
        for episode in selected:
            try:
                route_registry.resolve_manifest_pair(
                    episode,
                    str(scene.get("scene")),
                    str(episode.get("episode")),
                )
            except FlowRoutingError as exc:
                raise ManifestAuditError(
                    "selected flow pair is not authorized by the route "
                    f"artifact: {scene.get('scene')}/{episode.get('episode')}: "
                    f"{exc}") from exc
        if scene.get("scene") == "YmJkqBEsHnH":
            ymj = {
                episode.get("episode"): {
                    "n_frames": episode.get("n_frames"),
                    "switches": episode.get("switches"),
                }
                for episode in selected
                if isinstance(episode, dict)
            }
    if role_counts != {"train": 40, "development": 10}:
        raise ManifestAuditError(f"scene role counts mismatch: {role_counts}")
    if ymj != EXPECTED_YMJ:
        raise ManifestAuditError(f"YmJ manifest summary mismatch: {ymj}")

    sample_ids = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ManifestAuditError("manifest sample is not an object")
        if sample.get("split_role") not in role_counts:
            raise ManifestAuditError("final-reserved sample leaked")
        if sample.get("state_source") != "expert":
            raise ManifestAuditError("non-expert state leaked into v1 manifest")
        sample_ids.append(sample.get("sample_id"))
    if len(set(sample_ids)) != 400 or None in sample_ids:
        raise ManifestAuditError("manifest sample IDs are missing or duplicated")
    per_episode: dict[str, list[dict]] = {}
    for sample in samples:
        key = str(sample.get("source_episode_id"))
        per_episode.setdefault(key, []).append(sample)
    if set(per_episode) != set(route_registry.files_by_episode):
        raise ManifestAuditError(
            "manifest sampled episodes differ from routed cache episodes")
    for key, episode_samples in per_episode.items():
        combinations = {
            (sample.get("state_name"), sample.get("goal_variant"))
            for sample in episode_samples
        }
        expected_combinations = {
            ("goal_b_t0", "factual"),
            ("goal_b_t0", "counterfactual"),
            ("goal_b_midpoint_t1", "factual"),
            ("goal_b_midpoint_t1", "counterfactual"),
        }
        if len(episode_samples) != 4 or combinations != expected_combinations:
            raise ManifestAuditError(
                f"sample state/variant invariant differs for {key}")

    if raw.get("historical_three_leg_reference") != EXPECTED_YMJ:
        raise ManifestAuditError("raw audit historical gate differs")
    if raw.get("byte_identical_historical_recovery_claim") is not False:
        raise ManifestAuditError("raw audit overclaims byte-identical recovery")
    if flow.get("raw_audit_sha256") != raw_sha:
        raise ManifestAuditError("flow provenance does not name this raw audit")
    if flow.get("split_sha256") != EXPECTED_SPLIT_SHA:
        raise ManifestAuditError("flow provenance split SHA mismatch")
    if flow.get("counts") != {
            "scenes": 50, "pairs": 100,
            "official_base": 97, "flow4096_patch": 3}:
        raise ManifestAuditError("flow provenance counts mismatch")
    if Path(str(flow.get("route_root", ""))).resolve() != \
            args.expected_flow_route_root.resolve():
        raise ManifestAuditError("flow provenance route-root mismatch")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "corrected_manifest_audited",
        "wrapper_commit": args.expected_commit,
        "split_sha256": EXPECTED_SPLIT_SHA,
        "manifest_sha256": manifest_sha,
        "raw_audit_sha256": raw_sha,
        "flow_provenance_sha256": flow_sha,
        "summary": EXPECTED_SUMMARY,
        "scene_role_counts": role_counts,
        "ymj_historical_summary_gate": EXPECTED_YMJ,
        "byte_identical_historical_recovery_claim": False,
        "downstream_ready": True,
    }
    digest = _write_new(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.out),
        "sha256": digest,
        "downstream_ready": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
