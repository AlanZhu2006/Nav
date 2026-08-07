#!/usr/bin/env python3
"""Stage B: assemble geometry labels from a signed DINO embedding bundle.

This executable belongs in the Habitat environment.  It has no live DINO,
LingBot, or Torch fallback: any missing or changed Stage-A input fails closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from MemNavData.build_manifest_causal_covisibility_teacher import (
        FORMAL_MANIFEST_SHA256,
        PinnedDINOEmbeddingBundleProvider,
        PinnedHabitatGoalDepthRenderer,
        TeacherConfig,
        _exclusive_stage_writer,
        _resolved_roots,
        build_teacher_artifact,
        load_pinned_manifest,
        sha256_file,
        write_teacher_bundle,
    )
except ImportError:  # pragma: no cover - direct script execution
    from build_manifest_causal_covisibility_teacher import (  # type: ignore
        FORMAL_MANIFEST_SHA256,
        PinnedDINOEmbeddingBundleProvider,
        PinnedHabitatGoalDepthRenderer,
        TeacherConfig,
        _exclusive_stage_writer,
        _resolved_roots,
        build_teacher_artifact,
        load_pinned_manifest,
        sha256_file,
        write_teacher_bundle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage B: assemble causal geometry labels from signed embeddings"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=FORMAL_MANIFEST_SHA256)
    parser.add_argument("--expected-sample-count", type=int, default=600)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--embedding-bundle", type=Path, required=True)
    parser.add_argument("--expected-embedding-receipt-sha256", required=True)
    parser.add_argument("--expected-habitat-version", required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--habitat-bindings-file", type=Path, required=True)
    parser.add_argument("--expected-habitat-bindings-sha256", required=True)
    parser.add_argument("--expected-producer-sha256", required=True)
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--expected-entrypoint-sha256", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--temporal-nms-radius", type=int, default=4)
    parser.add_argument("--backprojection-stride", type=int, default=6)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.3)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.1)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    if (
        args.progress_dir.exists() or args.progress_dir.is_symlink()
    ) and not args.resume:
        raise RuntimeError(
            f"assembly progress already exists without --resume: {args.progress_dir}"
        )
    manifest = load_pinned_manifest(args.manifest, args.expected_manifest_sha256)
    overrides = {
        key: value
        for key, value in {
            "episode_root": args.episode_root,
            "environment_root": args.environment_root,
        }.items()
        if value is not None
    }
    roots = _resolved_roots(manifest, overrides)
    embeddings = PinnedDINOEmbeddingBundleProvider(
        bundle_directory=args.embedding_bundle,
        expected_receipt_sha256=args.expected_embedding_receipt_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_producer_sha256=args.expected_producer_sha256,
        episode_root=roots["episode_root"],
    )
    renderer = PinnedHabitatGoalDepthRenderer(
        expected_habitat_version=args.expected_habitat_version,
        bindings_file=args.habitat_bindings_file,
        expected_bindings_sha256=args.expected_habitat_bindings_sha256,
        entrypoint_path=Path(__file__),
        expected_entrypoint_sha256=args.expected_entrypoint_sha256,
        expected_python_sha256=args.expected_python_sha256,
    )
    try:
        artifact = build_teacher_artifact(
            manifest=manifest,
            manifest_sha256=args.expected_manifest_sha256,
            embedding_provider=embeddings,
            renderer=renderer,
            config=TeacherConfig(
                top_k=args.top_k,
                temporal_nms_radius=args.temporal_nms_radius,
                backprojection_stride=args.backprojection_stride,
                depth_tolerance_m=args.depth_tolerance_m,
                positive_threshold=args.positive_threshold,
                negative_threshold=args.negative_threshold,
            ),
            root_overrides=overrides,
            expected_sample_count=args.expected_sample_count,
            progress_directory=args.progress_dir,
            expected_producer_sha256=args.expected_producer_sha256,
            expected_geometry_sha256=args.expected_geometry_sha256,
        )
        result = write_teacher_bundle(
            artifact,
            args.out_dir,
            resume=args.resume and args.out_dir.exists(),
        )
    finally:
        renderer.close()
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    expected_entrypoint_sha = args.expected_entrypoint_sha256.lower()
    if sha256_file(Path(__file__).resolve()) != expected_entrypoint_sha:
        raise RuntimeError("Stage-B entrypoint differs from external SHA pin")
    lock_targets = sorted(
        {args.progress_dir.resolve(), args.out_dir.resolve()}, key=str
    )
    if len(lock_targets) != 2:
        raise RuntimeError("assembly progress and output directories must differ")
    with _exclusive_stage_writer(lock_targets[0]):
        with _exclusive_stage_writer(lock_targets[1]):
            _run(args)


if __name__ == "__main__":
    main()
