#!/usr/bin/env python3
"""Stage A: build the signed exact-LingBot DINO embedding bundle only.

This executable belongs in the MemNav/Torch environment.  It never imports or
constructs Habitat-Sim and has no geometry-label authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from MemNavData.build_manifest_causal_covisibility_teacher import (
        ExactLingBotDINOProvider,
        FORMAL_MANIFEST_SHA256,
        build_dino_embedding_bundle,
        load_pinned_manifest,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution
    from build_manifest_causal_covisibility_teacher import (  # type: ignore
        ExactLingBotDINOProvider,
        FORMAL_MANIFEST_SHA256,
        build_dino_embedding_bundle,
        load_pinned_manifest,
        sha256_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage A: materialize signed exact-DINO embeddings"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=FORMAL_MANIFEST_SHA256)
    parser.add_argument("--expected-sample-count", type=int, default=600)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--expected-lingbot-commit", required=True)
    parser.add_argument("--expected-lingbot-tree", required=True)
    parser.add_argument("--lingbot-weights", type=Path, required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--expected-lingbot-weights-sha256", required=True)
    parser.add_argument("--expected-dino-loader-sha256", required=True)
    parser.add_argument("--expected-dino-vit-sha256", required=True)
    parser.add_argument("--expected-dino-preprocessor-sha256", required=True)
    parser.add_argument("--dino-device", default="cuda")
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--embedding-chunk-size", type=int, default=256)
    parser.add_argument("--expected-producer-sha256", required=True)
    parser.add_argument("--expected-entrypoint-sha256", required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_entrypoint_sha = args.expected_entrypoint_sha256.lower()
    if sha256_file(Path(__file__).resolve()) != expected_entrypoint_sha:
        raise RuntimeError("Stage-A entrypoint differs from external SHA pin")
    manifest = load_pinned_manifest(args.manifest, args.expected_manifest_sha256)
    roots = {
        key: value
        for key, value in {
            "episode_root": args.episode_root,
            "environment_root": args.environment_root,
        }.items()
        if value is not None
    }
    provider = ExactLingBotDINOProvider(
        lingbot_repo=args.lingbot_repo,
        weights=args.lingbot_weights,
        entrypoint_path=Path(__file__),
        expected_entrypoint_sha256=args.expected_entrypoint_sha256,
        expected_python_sha256=args.expected_python_sha256,
        expected_weights_sha256=args.expected_lingbot_weights_sha256,
        expected_source_sha256={
            "exact_loader": args.expected_dino_loader_sha256,
            "vision_transformer": args.expected_dino_vit_sha256,
            "preprocessor": args.expected_dino_preprocessor_sha256,
        },
        expected_lingbot_commit=args.expected_lingbot_commit,
        expected_lingbot_tree=args.expected_lingbot_tree,
        device=args.dino_device,
        batch_size=args.dino_batch_size,
    )
    result = build_dino_embedding_bundle(
        manifest=manifest,
        manifest_sha256=args.expected_manifest_sha256,
        embedding_provider=provider,
        progress_directory=args.progress_dir,
        output_directory=args.out_dir,
        root_overrides=roots,
        expected_sample_count=args.expected_sample_count,
        embedding_chunk_size=args.embedding_chunk_size,
        expected_producer_sha256=args.expected_producer_sha256,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
