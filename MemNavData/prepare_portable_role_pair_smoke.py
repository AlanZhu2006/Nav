#!/usr/bin/env python3
"""Materialize one consumed role-pair episode for a remote integration smoke.

The query assets and online-A trace are copied byte-for-byte.  Only absolute
paths in the online-A receipt and role-pair manifest are rebound to the frozen
deployment root.  This tool is for consumed implementation tests, never for
creating or selecting a paper population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict:
    source_benchmark = args.source_benchmark.resolve()
    output = args.output_root.resolve()
    deployed = Path(args.deployed_root)
    runtime_asset = Path(args.runtime_scene_asset)
    if not deployed.is_absolute() or not runtime_asset.is_absolute():
        raise ValueError("deployed paths must be absolute")
    if output.exists():
        raise FileExistsError(output)
    source_manifest_path = source_benchmark / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    rows = source_manifest["episodes"]
    if not 0 <= args.selection_index < len(rows):
        raise IndexError("selection index outside source population")
    row = dict(rows[args.selection_index])
    scene = str(row["scene"])
    episode = str(row["episode"])
    source_query_root = source_benchmark / scene / episode
    source_online = Path(row["online_a_episode"]).resolve()
    online_receipt_path = source_online / "receipt.json"
    online_trace_path = source_online / "online_a_trace.json"
    if sha256(online_receipt_path) != row["online_a_receipt_sha256"]:
        raise ValueError("source online-A receipt changed")
    if sha256(online_trace_path) != row["online_a_trace_sha256"]:
        raise ValueError("source online-A trace changed")
    online_receipt = json.loads(online_receipt_path.read_text())
    source_asset = Path(online_receipt["source_asset"])
    source_episode = Path(online_receipt["source_episode"])
    source_parquet = (
        source_episode / "data/chunk-000/episode_000000.parquet"
    )
    if sha256(source_asset) != online_receipt["source_asset_sha256"]:
        raise ValueError("source scene asset changed")
    if sha256(source_parquet) != online_receipt["source_parquet_sha256"]:
        raise ValueError("source parquet changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.building.", dir=output.parent))
    try:
        benchmark_root = temporary / "benchmark"
        query_root = benchmark_root / scene / episode
        query_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_query_root, query_root)

        online_root = temporary / "online_a" / scene / episode
        online_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_online, online_root)

        portable_source = temporary / "source_episode" / scene / episode
        portable_parquet = (
            portable_source / "data/chunk-000/episode_000000.parquet"
        )
        portable_parquet.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_parquet, portable_parquet)

        deployed_online = deployed / "online_a" / scene / episode
        deployed_source = deployed / "source_episode" / scene / episode
        online_receipt["source_asset"] = str(runtime_asset)
        online_receipt["source_episode"] = str(deployed_source)
        write_json(online_root / "receipt.json", online_receipt)

        row["online_a_episode"] = str(deployed_online)
        row["online_a_receipt_sha256"] = sha256(
            online_root / "receipt.json")
        row["online_a_trace_sha256"] = sha256(
            online_root / "online_a_trace.json")
        sidecar = dict(row)
        sidecar.pop("role_pairs_sha256", None)
        write_json(query_root / "role_pairs.json", sidecar)
        row["role_pairs_sha256"] = sha256(query_root / "role_pairs.json")

        portable_manifest = dict(source_manifest)
        portable_manifest["episodes"] = [row]
        portable_manifest["purpose"] = (
            "single consumed role-pair remote integration smoke; no SR claim"
        )
        portable_manifest["source_online_root"] = str(
            deployed / "online_a")
        write_json(benchmark_root / "manifest.json", portable_manifest)
        manifest_sha = sha256(benchmark_root / "manifest.json")
        (benchmark_root / "manifest.json.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="utf-8")

        receipt = {
            "schema_version": "portable_role_pair_smoke_v1_20260817",
            "scope": "consumed integration only; no SR or efficacy claim",
            "source_manifest_sha256": sha256(source_manifest_path),
            "source_selection_index": args.selection_index,
            "scene": scene,
            "episode": episode,
            "deployed_root": str(deployed),
            "runtime_scene_asset": str(runtime_asset),
            "runtime_scene_asset_sha256": online_receipt[
                "source_asset_sha256"],
            "source_parquet_sha256": online_receipt[
                "source_parquet_sha256"],
            "portable_manifest_sha256": manifest_sha,
            "policy_outcomes_read_for_selection": False,
        }
        write_json(temporary / "PORTABLE_RECEIPT.json", receipt)
        entries = []
        for path in sorted(p for p in temporary.rglob("*") if p.is_file()):
            entries.append(f"{sha256(path)}  ./{path.relative_to(temporary)}")
        (temporary / "FILES.sha256").write_text(
            "\n".join(entries) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-benchmark", type=Path, required=True)
    parser.add_argument("--selection-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deployed-root", required=True)
    parser.add_argument("--runtime-scene-asset", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
