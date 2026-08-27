#!/usr/bin/env python3
"""Replay one consumed online history and audit proposal/verification coupling.

This is a post-hoc mechanism diagnostic, not a policy evaluation.  It restores
the exact causal online-A RGB prefix, renders the factual A endpoint, and asks
the certified endpoint to apply the unchanged PnP certificate to both:

* the deployed geometry-ranked proposal; and
* the canonical raw-DINO top-1 proposal.

The counterfactual result is read-only and has no action authority.  No query
rollout is executed and no success-rate claim can be derived from this file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from generate_twoleg import K, make_sim, render


SCHEMA_VERSION = "certified_proposal_counterfactual_episode_v1_20260815"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jpg_bytes(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="JPEG", quality=95)
    return output.getvalue()


def _multipart_payload(
    files: dict[str, tuple[str, bytes]],
    data: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    """Encode the small Flask requests without a requests dependency.

    The Habitat runtime intentionally has a minimal Python environment and
    does not install ``requests``.  Keeping transport in the standard library
    also avoids adding a new experimental dependency to this read-only audit.
    """
    boundary = "cec-proposal-audit-20260815"
    chunks: list[bytes] = []

    def append_field(name: str, value: bytes, filename: str | None) -> None:
        require(boundary.encode() not in value, "multipart boundary collision")
        chunks.append(f"--{boundary}\r\n".encode())
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        chunks.append((disposition + "\r\n").encode())
        if filename is not None:
            chunks.append(b"Content-Type: image/jpeg\r\n")
        chunks.append(b"\r\n")
        chunks.append(value)
        chunks.append(b"\r\n")

    for name, value in (data or {}).items():
        append_field(str(name), str(value).encode(), None)
    for name, (filename, value) in files.items():
        append_field(str(name), bytes(value), str(filename))
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def checked_post(url: str, **kwargs: Any) -> dict[str, Any]:
    allowed = {"json", "files", "data"}
    require(not (set(kwargs) - allowed), "unsupported HTTP request argument")
    if "json" in kwargs:
        require("files" not in kwargs and "data" not in kwargs,
                "JSON and multipart payloads cannot be mixed")
        body = json.dumps(kwargs["json"], allow_nan=False).encode()
        content_type = "application/json"
    else:
        require("files" in kwargs, "multipart request omitted files")
        body, content_type = _multipart_payload(
            kwargs["files"], kwargs.get("data"))
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {detail[:4000]}"
        ) from exc
    payload = json.loads(raw.decode("utf-8"))
    require(isinstance(payload, dict), f"non-object response from {url}")
    return payload


def choose_revisit_query(role_pairs: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        query
        for pair in role_pairs["pairs"]
        for query in pair["queries"]
        if query.get("analysis_role") == "revisit"
    ]
    require(len(candidates) == 1, "expected exactly one Revisit query")
    return candidates[0]


def replay_history(
    base_url: str,
    source: Path,
    receipt: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    reset = checked_post(
        f"{base_url}/navigator_reset",
        json={
            "camera_height": float(receipt["camera_height_m"]),
            "camera_intrinsic": np.asarray(K, dtype=float).tolist(),
            "seed": int(trace["episode_seed"]),
            # Match the formal query evaluation's flow-gate tier.
            "episode_len": int(len(trace["poses"]) + 600),
        },
    )
    certificate_status = reset.get("certified_relocalization")
    require(
        isinstance(certificate_status, dict)
        and certificate_status.get("enabled") is True,
        "certified endpoint is disabled",
    )
    require(
        certificate_status.get("counterfactual_dino_top1_audit") is True,
        "server did not enable the read-only DINO top-1 audit",
    )

    replayed = 0
    for expected_step, pose in enumerate(trace["poses"]):
        step = int(pose["step"])
        require(step == expected_step, "online-A trace is not contiguous")
        image = source / "rgb" / f"{step:06d}.jpg"
        require(image.is_file(), f"missing online RGB {image}")
        require(
            sha256_file(image) == pose["jpg_sha256"],
            f"online RGB hash changed at step {step}",
        )
        response = checked_post(
            f"{base_url}/memory_step",
            files={"image": ("image.jpg", image.read_bytes())},
        )
        require(int(response["frame_idx"]) == step, "replay index changed")
        replayed += 1
    return {"reset": reset, "online_frames": replayed}


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = args.benchmark_root.resolve()
    manifest_path = benchmark_root / "manifest.json"
    require(manifest_path.is_file(), "benchmark manifest is missing")
    manifest_sha = sha256_file(manifest_path)
    require(
        manifest_sha == args.expected_manifest_sha256,
        "benchmark manifest SHA changed",
    )
    manifest = json.loads(manifest_path.read_text())
    require(0 <= args.index < len(manifest["episodes"]), "index out of range")
    manifest_row = manifest["episodes"][args.index]
    scene = str(manifest_row["scene"])
    episode = str(manifest_row["episode"])
    episode_dir = benchmark_root / scene / episode
    role_pairs_path = episode_dir / "role_pairs.json"
    require(role_pairs_path.is_file(), "role-pair episode is missing")
    role_pairs = json.loads(role_pairs_path.read_text())
    require(
        role_pairs["scene"] == scene and role_pairs["episode"] == episode,
        "role-pair identity changed",
    )

    source = Path(role_pairs["online_a_episode"])
    receipt_path = source / "receipt.json"
    trace_path = source / "online_a_trace.json"
    require(source.is_dir(), "online-A source is missing")
    require(
        sha256_file(receipt_path) == role_pairs["online_a_receipt_sha256"],
        "online-A receipt hash changed",
    )
    require(
        sha256_file(trace_path) == role_pairs["online_a_trace_sha256"],
        "online-A trace hash changed",
    )
    receipt = json.loads(receipt_path.read_text())
    trace = json.loads(trace_path.read_text())
    require(trace.get("reached") is True, "online A was not successful")

    query = choose_revisit_query(role_pairs)
    goal_path = episode_dir / query["goal_rgb"]
    require(goal_path.is_file(), "Revisit goal RGB is missing")
    require(
        sha256_file(goal_path) == query["goal_rgb_sha256"],
        "Revisit goal RGB hash changed",
    )

    replay = replay_history(args.memnav_url, source, receipt, trace)
    scene_asset = Path(receipt["source_asset"])
    require(scene_asset.is_file(), "source scene asset is missing")
    require(
        sha256_file(scene_asset) == receipt["source_asset_sha256"],
        "source scene asset hash changed",
    )
    simulator = make_sim(str(scene_asset), "", agent_radius=0.30)
    try:
        current_rgb, _ = render(
            simulator,
            np.asarray(trace["end_position"], dtype=np.float64),
            float(trace["end_yaw"]),
        )
    finally:
        simulator.close()
    current_jpg = jpg_bytes(current_rgb)
    goal_jpg = goal_path.read_bytes()
    probe = checked_post(
        f"{args.memnav_url}/retrieval_probe_step",
        files={
            "image": ("image.jpg", current_jpg),
            "goal": ("goal.jpg", goal_jpg),
        },
    )
    shortlist = probe.get("certified_visual_candidates")
    require(isinstance(shortlist, list), "probe omitted certified shortlist")
    localized = checked_post(
        f"{args.memnav_url}/certified_relocalize",
        files={"goal": ("goal.jpg", goal_jpg)},
        data={
            "candidates": json.dumps(shortlist),
            "graph_rescue": "0",
            "learned_rescue": "0",
        },
    )
    audit = localized.get("counterfactual_dino_top1_audit")
    ordered_audit = localized.get("counterfactual_dino_order_audit")
    require(audit is not None, "counterfactual top-1 result is missing")
    require(ordered_audit is not None, "counterfactual ordered result is missing")
    require(audit.get("action_authority") is False, "top-1 audit gained authority")
    require(
        ordered_audit.get("action_authority") is False,
        "ordered audit gained authority",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "consumed_posthoc_mechanism_diagnostic",
        "is_closed_loop_evaluation": False,
        "is_method_selection_evidence": True,
        "is_confirmation_evidence": False,
        "query_role_selected_for_analysis": "revisit",
        "scene": scene,
        "episode": episode,
        "population_index": int(args.index),
        "benchmark_root": str(benchmark_root),
        "benchmark_manifest_sha256": manifest_sha,
        "role_pairs_sha256": sha256_file(role_pairs_path),
        "goal_rgb_sha256": query["goal_rgb_sha256"],
        "current_endpoint_rgb_sha256": hashlib.sha256(current_jpg).hexdigest(),
        "online_frames": replay["online_frames"],
        "candidate_count": len(shortlist),
        "dino_top1_anchor": (
            int(shortlist[0]["anchor"]) if shortlist else None
        ),
        "geometry_selected_anchor": localized.get("selected_anchor"),
        "geometry_accepted": localized.get("accepted"),
        "geometry_reason": localized.get("reason"),
        "geometry_pnp": localized.get("pnp"),
        "geometry_certificate": localized.get("certificate"),
        "counterfactual_dino_top1": audit,
        "counterfactual_dino_first_certified": ordered_audit,
        "ranked_candidates": localized.get("ranked_candidates"),
        "method_action_unchanged": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--memnav-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), "output already exists")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "scene": result["scene"],
        "episode": result["episode"],
        "geometry_anchor": result["geometry_selected_anchor"],
        "geometry_accepted": result["geometry_accepted"],
        "dino_top1_anchor": result["dino_top1_anchor"],
        "dino_top1_accepted": result["counterfactual_dino_top1"]["accepted"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
