#!/usr/bin/env python3
"""Build an immutable consumed counterfactual for fallback-equivalence tests.

The copied episode preserves simulator state, Goal A, metadata, and the target
position.  It replaces only the Goal-B image with a hash-bound cross-scene
negative.  Consequently, the output is a transport/safety test and must never
be pooled into navigation efficacy statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: str, label: str) -> None:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be lowercase 64-hex")


def build(args: argparse.Namespace) -> dict:
    source = args.source_episode.resolve()
    override = args.goal_b_override.resolve()
    output = args.output_root.resolve()
    require_sha(args.expected_source_goal_sha256, "source goal hash")
    require_sha(args.expected_override_sha256, "override hash")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    for path in (
        source / "goal_1.jpg",
        source / "goal_image.jpg",
        source / "meta" / "gen_meta.json",
        source / "data" / "chunk-000" / "episode_000000.parquet",
        override,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_goal_sha = sha256(source / "goal_1.jpg")
    if source_goal_sha != args.expected_source_goal_sha256:
        raise ValueError("source Goal-B identity changed")
    override_sha = sha256(override)
    if override_sha != args.expected_override_sha256:
        raise ValueError("counterfactual Goal-B identity changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.building.", dir=output.parent))
    try:
        episode = temporary / args.episode_id
        shutil.copytree(source, episode)
        shutil.copyfile(override, episode / "goal_1.jpg")
        shutil.copyfile(override, episode / "goal_image.jpg")
        if sha256(episode / "goal_1.jpg") != override_sha:
            raise RuntimeError("copied Goal-B content changed")
        if sha256(episode / "goal_image.jpg") != override_sha:
            raise RuntimeError("legacy Goal-B alias differs")

        receipt = {
            "schema_version": (
                "pi3x_fallback_counterfactual_input_v1_20260817"),
            "scope": (
                "consumed cross-scene Goal-B transport/safety smoke; "
                "never an efficacy or SR sample"),
            "episode_id": args.episode_id,
            "source_episode": str(source),
            "source_goal_b_sha256": source_goal_sha,
            "goal_b_override": str(override),
            "goal_b_override_sha256": override_sha,
            "metadata_sha256": sha256(episode / "meta" / "gen_meta.json"),
            "parquet_sha256": sha256(
                episode / "data" / "chunk-000" / "episode_000000.parquet"),
            "only_goal_b_content_replaced": True,
        }
        receipt_path = temporary / "COUNTERFACTUAL_RECEIPT.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_entries = []
        for path in sorted(p for p in temporary.rglob("*") if p.is_file()):
            relative = path.relative_to(temporary)
            manifest_entries.append(f"{sha256(path)}  ./{relative}")
        (temporary / "INPUTS.sha256").write_text(
            "\n".join(manifest_entries) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    for path in sorted(output.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    output.chmod(0o555)
    receipt["output_root"] = str(output)
    receipt["input_manifest_sha256"] = sha256(output / "INPUTS.sha256")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-episode", type=Path, required=True)
    parser.add_argument("--goal-b-override", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-id", default="episode_0000")
    parser.add_argument("--expected-source-goal-sha256", required=True)
    parser.add_argument("--expected-override-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
