import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from MemNavData.audit_causal_ground_scale_artifact import (
    AuditContract,
    CausalScaleAuditError,
    PRODUCER_SOURCE_PATHS,
    _load_runtime_camera,
    audit_causal_ground_scale,
)
from MemNavData.build_novel_frontier_candidates import (
    _validate_versioned_cache_pair,
)
from MemNavData.external_causal_scale_contract import (
    CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
    canonical_json_bytes,
    ndarray_sha256,
    sha256_bytes,
    sha256_file,
)
from MemNavData.flow_cache_routing import (
    FLOW_FILE_NAMES,
    ROUTE_SCHEMA_VERSION,
    ROUTE_STATUS,
    load_route_registry,
)


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "content_sha256": sha256_file(path),
    }


def _write_canonical_pair(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


class CausalGroundScaleAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.episode_root = self.root / "episodes"
        self.flow_root = self.root / "flow"
        self.route_root = self.root / "route"
        for directory in (
            self.repository / "MemNavData",
            self.episode_root,
            self.flow_root,
            self.route_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.scene = "scene_a"
        self.episode = "episode_0000"
        self.n_frames = 16
        self.chunk = self.episode_root / self.scene / self.episode / "videos/chunk-000"
        self.rgb = self.chunk / "observation.images.rgb"
        self.rgb.mkdir(parents=True)
        for frame in range(self.n_frames):
            (self.rgb / f"{frame}.jpg").write_bytes(f"rgb-{frame:03d}".encode("ascii"))
        self.goal_b = self.episode_root / self.scene / self.episode / "goal_1.jpg"
        self.goal_c = self.episode_root / self.scene / self.episode / "goal_2.jpg"
        self.goal_b.write_bytes(b"goal-b")
        self.goal_c.write_bytes(b"goal-c")
        self.metadata = self.episode_root / self.scene / self.episode / "gen_meta.json"
        self.metadata.write_text(
            json.dumps(
                {
                    "n_frames": self.n_frames,
                    "camera_height_m": 0.5,
                }
            ),
            encoding="utf-8",
        )

        self.cam_pose = np.zeros((self.n_frames, 9), dtype=np.float32)
        self.cam_pose[:, 6] = 1.0
        self.aggregator, self.camera = self._write_cache_pair(self.cam_pose)
        self.split_sha = "1" * 64
        self.route_record = self._make_route_record()
        self.route_path = self.route_root / "FLOW_ROUTE_PROVENANCE.json"
        self.route_sha = _write_canonical_pair(self.route_path, self.route_record)
        self.registry = load_route_registry(self.route_path, self.route_sha)

        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._make_manifest()
        self.manifest_sha = _write_canonical_pair(self.manifest_path, self.manifest)

        self.config = {
            "prefix_frame_cap": 64,
            "num_scale_frames": 8,
            "bias_correction": 1.15,
            "scale_min": 0.8,
            "scale_max": 6.0,
        }
        self.configuration_sha = sha256_bytes(canonical_json_bytes(self.config))
        self.lingbot_commit = "2" * 40
        self.weights_sha = "3" * 64
        self.stream_sha = "4" * 64
        self.acceptance_commit = "5" * 40
        self.auditor_path = (
            self.repository / "MemNavData/audit_causal_ground_scale_artifact.py"
        )
        source_auditor = Path(__file__).with_name(
            "audit_causal_ground_scale_artifact.py"
        )
        self.auditor_path.write_bytes(source_auditor.read_bytes())
        self.auditor_sha = sha256_file(self.auditor_path)
        self.source_snapshots: dict[str, Path] = {}
        for index, (name, relative) in enumerate(PRODUCER_SOURCE_PATHS.items()):
            path = self.repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"producer-{index}-{name}".encode("utf-8"))
            self.source_snapshots[name] = path
        self.source_files = {
            name: sha256_file(path) for name, path in self.source_snapshots.items()
        }
        self.producer_sha = sha256_bytes(canonical_json_bytes(self.source_files))

        self.artifact_path = self.root / "causal_ground_scale.json"
        self.artifact = self._make_artifact()
        self.artifact_sha = _write_canonical_pair(self.artifact_path, self.artifact)
        self.output = self.root / "acceptance/receipt.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_cache_pair(self, cam_pose: np.ndarray) -> tuple[Path, Path]:
        flow_chunk = self.flow_root / self.scene / self.episode / "videos/chunk-000"
        flow_chunk.mkdir(parents=True, exist_ok=True)
        aggregator = flow_chunk / FLOW_FILE_NAMES[0]
        camera = flow_chunk / FLOW_FILE_NAMES[1]
        num_scale, interval = 8, 4
        anchors = np.arange(num_scale, self.n_frames, interval, dtype=np.int64)
        cam_indices = np.concatenate((np.arange(num_scale, dtype=np.int64), anchors))
        shared = {
            "cache_schema_version": np.asarray([2], dtype=np.int64),
            "keyframe_policy": np.asarray(["post_scale_mod_v1"]),
            "num_frames": np.asarray([self.n_frames], dtype=np.int64),
            "num_scale_frames": np.asarray([num_scale], dtype=np.int64),
            "keyframe_interval": np.asarray([interval], dtype=np.int64),
            "precompute_signature": np.asarray(["acceptance-fixture-v1"]),
        }
        np.savez(
            aggregator,
            dino_cls=np.zeros((self.n_frames, 1), dtype=np.float16),
            anchor_k=np.zeros((len(anchors), 1), dtype=np.float16),
            anchor_v=np.zeros((len(anchors), 1), dtype=np.float16),
            anchor_frame_indices=anchors,
            **shared,
        )
        np.savez(
            camera,
            cam_pose_enc=np.asarray(cam_pose),
            cam_k=np.zeros((len(cam_indices), 1), dtype=np.float16),
            cam_v=np.zeros((len(cam_indices), 1), dtype=np.float16),
            cam_frame_indices=cam_indices,
            **shared,
        )
        return aggregator, camera

    def _make_route_record(self) -> dict[str, object]:
        relative_chunk = f"{self.scene}/{self.episode}/videos/chunk-000"
        files = []
        for name, path in zip(FLOW_FILE_NAMES, (self.aggregator, self.camera)):
            files.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "content_sha256": None,
                }
            )
        return {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "status": ROUTE_STATUS,
            "split_sha256": self.split_sha,
            "raw_audit_sha256": "6" * 64,
            "route_root": str(self.route_root.resolve()),
            "source_roots": {"official_base": str(self.flow_root.resolve())},
            "official_snapshot_semantics": "acceptance fixture",
            "official_snapshot_sha256": "7" * 64,
            "patch_payloads_fully_sha256": True,
            "counts": {"scenes": 1, "pairs": 1, "official_base": 1},
            "pairs": [
                {
                    "episode": f"{self.scene}/{self.episode}",
                    "source_id": "official_base",
                    "source_relative_chunk": relative_chunk,
                    "validation": {"files": files},
                }
            ],
        }

    def _sample(
        self,
        sample_id: str,
        decision: int,
        goal_role: str,
        state: str,
    ) -> dict[str, object]:
        goal = self.goal_b if goal_role == "B" else self.goal_c
        return {
            "sample_id": sample_id,
            "split_role": "train",
            "scene": self.scene,
            "source_episode": self.episode,
            "source_episode_id": f"{self.scene}/{self.episode}",
            "goal_episode": self.episode,
            "goal_source_episode_id": f"{self.scene}/{self.episode}",
            "goal_variant": "factual",
            "goal_role": goal_role,
            "state_name": state,
            "decision_frame": decision,
            "causal_prefix": {
                "frame_count": decision,
                "causal_prefix_sha256": sha256_bytes(
                    f"prefix-{decision}".encode("ascii")
                ),
            },
            "navdp_fifo": {
                "fifo_sha256": sha256_bytes(f"fifo-{decision}".encode("ascii")),
            },
            "goal": _file_record(goal, self.episode_root),
        }

    def _make_manifest(self) -> dict[str, object]:
        samples = [
            self._sample("b_t0", 10, "B", "goal_b_t0"),
            self._sample("b_mid", 12, "B", "goal_b_midpoint_t1"),
            self._sample("c_t2", 14, "C", "goal_c_t2"),
        ]
        flow_files = self.registry.episode_file_records(self.scene, self.episode)
        return {
            "schema_version": ("nlsr_v2_multistage_expert_candidate_manifest_v1"),
            "split": {"sha256": self.split_sha},
            "input_roots": {
                "episode_root": str(self.episode_root.resolve()),
            },
            "flow_cache_routing": self.registry.manifest_record(),
            "scenes": [
                {
                    "scene": self.scene,
                    "split_role": "train",
                    "selected_episodes": [
                        {
                            "episode": self.episode,
                            "n_frames": self.n_frames,
                            "metadata": _file_record(self.metadata, self.episode_root),
                            "flow_cache": {"complete": True, "files": flow_files},
                        }
                    ],
                }
            ],
            "samples": samples,
            "summary": {
                "scene_count": 1,
                "episode_count": 1,
                "sample_count": len(samples),
            },
        }

    def _rgb_prefix_record(self, frame_count: int) -> dict[str, object]:
        rows = [
            _file_record(self.rgb / f"{frame}.jpg", self.episode_root)
            for frame in range(frame_count)
        ]
        return {
            "frame_count": frame_count,
            "path_sequence_sha256": sha256_bytes(
                canonical_json_bytes([row["path"] for row in rows])
            ),
            "content_sequence_sha256": sha256_bytes(canonical_json_bytes(rows)),
        }

    def _make_artifact(self) -> dict[str, object]:
        decisions = [10, 12, 14]
        record = {
            "scene": self.scene,
            "episode": self.episode,
            "split_role": "train",
            "sample_ids": ["b_mid", "b_t0", "c_t2"],
            "decision_frames": decisions,
            "earliest_decision_frame": decisions[0],
            "prefix_end_frame_exclusive": 10,
            "camera_height_m": 0.5,
            "episode_frame_count": self.n_frames,
            "rgb_prefix": self._rgb_prefix_record(10),
            "cam_pose_prefix_sha256": ndarray_sha256(self.cam_pose[:10]),
            "cam_pose_prefix_dtype": self.cam_pose.dtype.str,
            "cache_schema_version": 2,
            "precompute_signature": "acceptance-fixture-v1",
            "valid": True,
            "invalid_reason": None,
            "ground_h_est_raw": 0.25,
            "metric_scale_m_per_raw": 2.3,
            "debug": {
                "n_points": 128,
                "n_frames": 8,
                "n_valid": 8,
                "h_est": 0.25,
                "h_iqr": 0.025,
            },
        }
        return {
            "schema_version": CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
            "purpose": "acceptance fixture",
            "provenance": {
                "input_manifest_path": str(self.manifest_path.resolve()),
                "input_manifest_sha256": self.manifest_sha,
                "input_manifest_schema_version": self.manifest["schema_version"],
                "flow_cache_routing": self.manifest["flow_cache_routing"],
                "producer_source_sha256": self.producer_sha,
                "producer_source_files": self.source_files,
                "configuration_sha256": self.configuration_sha,
                "estimator": {
                    "kind": "frozen_lingbot_compute_metric_scale_prefix",
                    "lingbot_commit": self.lingbot_commit,
                    "weights_sha256": self.weights_sha,
                    "lingbot_stream_source_sha256": self.stream_sha,
                },
            },
            "configuration": self.config,
            "selection": {"selected_scenes": [], "all_manifest_scenes": True},
            "records": [record],
            "summary": {
                "scene_count": 1,
                "episode_count": 1,
                "valid_episode_count": 1,
                "invalid_episode_count": 0,
                "maximum_prefix_frames": 10,
                "future_frames_consumed": 0,
            },
        }

    def _rewrite_manifest_and_artifact(self) -> None:
        self.manifest_sha = _write_canonical_pair(self.manifest_path, self.manifest)
        provenance = self.artifact["provenance"]
        assert isinstance(provenance, dict)
        provenance["input_manifest_sha256"] = self.manifest_sha
        provenance["flow_cache_routing"] = self.manifest["flow_cache_routing"]
        self.artifact_sha = _write_canonical_pair(self.artifact_path, self.artifact)

    def contract(self) -> AuditContract:
        return AuditContract(
            manifest_path=self.manifest_path,
            artifact_path=self.artifact_path,
            output_path=self.output,
            repository_root=self.repository,
            auditor_path=self.auditor_path,
            expected_manifest_sha256=self.manifest_sha,
            expected_artifact_sha256=self.artifact_sha,
            expected_producer_sha256=self.producer_sha,
            expected_configuration_sha256=self.configuration_sha,
            expected_lingbot_commit=self.lingbot_commit,
            expected_weights_sha256=self.weights_sha,
            expected_stream_source_sha256=self.stream_sha,
            expected_auditor_sha256=self.auditor_sha,
            expected_acceptance_commit=self.acceptance_commit,
            expected_scene_count=1,
            expected_episode_count=1,
            expected_sample_count=3,
        )

    def test_default_validator_rebinds_and_receipt_is_idempotent(self) -> None:
        first = audit_causal_ground_scale(self.contract())
        first_payload = self.output.read_bytes()
        second = audit_causal_ground_scale(self.contract())
        self.assertEqual(first, second)
        self.assertEqual(self.output.read_bytes(), first_payload)
        receipt = json.loads(first_payload)
        self.assertEqual(
            receipt["coverage"],
            {
                "scene_count": 1,
                "episode_count": 1,
                "sample_count": 3,
                "future_frames_consumed": 0,
                "all_episode_estimates_valid": True,
            },
        )
        self.assertEqual(
            receipt["physical_rebinding"]["independent_prefix_validation_passes"],
            2,
        )
        digest = sha256_bytes(first_payload)
        self.assertEqual(
            Path(f"{self.output}.sha256").read_text(encoding="ascii"),
            f"{digest}  {self.output.name}\n",
        )

    def test_camera_loader_materializes_only_the_causal_prefix(self) -> None:
        real_load = np.load

        class GuardedNpz:
            def __init__(self, value: object) -> None:
                self.value = value

            def __enter__(self) -> "GuardedNpz":
                self.value.__enter__()
                return self

            def __exit__(self, *args: object) -> object:
                return self.value.__exit__(*args)

            @property
            def files(self) -> list[str]:
                return self.value.files

            def __getitem__(self, name: str) -> object:
                if name == "cam_pose_enc":
                    raise AssertionError("future camera rows were materialized")
                return self.value[name]

        def guarded_load(*args: object, **kwargs: object) -> GuardedNpz:
            return GuardedNpz(real_load(*args, **kwargs))

        with mock.patch(
            "MemNavData.audit_causal_ground_scale_artifact.np.load",
            side_effect=guarded_load,
        ):
            prefix, schema, signature = _load_runtime_camera(
                self.camera,
                expected_num_frames=self.n_frames,
                prefix_frame_count=10,
            )
        np.testing.assert_array_equal(prefix, self.cam_pose[:10])
        self.assertEqual(schema, 2)
        self.assertEqual(signature, "acceptance-fixture-v1")

    def test_camera_prefix_tamper_fails_closed(self) -> None:
        changed = self.cam_pose.copy()
        changed[0, 0] = 1.0
        old_size = self.camera.stat().st_size
        self._write_cache_pair(changed)
        self.assertEqual(self.camera.stat().st_size, old_size)
        with self.assertRaisesRegex(CausalScaleAuditError, "prefix changed"):
            audit_causal_ground_scale(self.contract())

    def test_missing_cache_and_wrong_routing_fail_closed(self) -> None:
        self.camera.unlink()
        with self.assertRaisesRegex(CausalScaleAuditError, "routing|absent|cache"):
            audit_causal_ground_scale(self.contract())

        self._write_cache_pair(self.cam_pose)
        routing = copy.deepcopy(self.manifest["flow_cache_routing"])
        assert isinstance(routing, dict)
        roots = routing["source_roots"]
        assert isinstance(roots, dict)
        roots["official_base"] = str((self.root / "wrong").resolve())
        self.manifest["flow_cache_routing"] = routing
        self._rewrite_manifest_and_artifact()
        with self.assertRaisesRegex(CausalScaleAuditError, "routing|source root"):
            audit_causal_ground_scale(self.contract())

    def test_future_or_inexact_sample_cover_fails_closed(self) -> None:
        summary = self.artifact["summary"]
        assert isinstance(summary, dict)
        summary["future_frames_consumed"] = 1
        self.artifact_sha = _write_canonical_pair(self.artifact_path, self.artifact)
        with self.assertRaisesRegex(CausalScaleAuditError, "summary"):
            audit_causal_ground_scale(self.contract())

        self.artifact = self._make_artifact()
        records = self.artifact["records"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        records[0]["sample_ids"] = ["b_t0", "c_t2"]
        self.artifact_sha = _write_canonical_pair(self.artifact_path, self.artifact)
        with self.assertRaisesRegex(CausalScaleAuditError, "sample/decision"):
            audit_causal_ground_scale(self.contract())

    def test_manifest_sidecar_tamper_fails_closed(self) -> None:
        Path(f"{self.manifest_path}.sha256").write_text(
            f"{'f' * 64}  {self.manifest_path.name}\n", encoding="ascii"
        )
        with self.assertRaisesRegex(CausalScaleAuditError, "sidecar"):
            audit_causal_ground_scale(self.contract())

    def test_producer_configuration_and_lingbot_pins_fail_closed(self) -> None:
        changes = {
            "expected_producer_sha256": "a" * 64,
            "expected_configuration_sha256": "b" * 64,
            "expected_lingbot_commit": "c" * 40,
            "expected_weights_sha256": "d" * 64,
            "expected_stream_source_sha256": "e" * 64,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                with self.assertRaises(CausalScaleAuditError):
                    audit_causal_ground_scale(
                        replace(self.contract(), **{field: value})
                    )

    def test_source_toctou_is_detected_before_publish(self) -> None:
        source = self.source_snapshots["flow_cache_routing.py"]

        def mutate_source(
            aggregator: Path,
            camera: Path,
            *,
            expected_num_frames: int,
        ) -> object:
            result = _validate_versioned_cache_pair(
                aggregator,
                camera,
                expected_num_frames=expected_num_frames,
            )
            source.write_bytes(b"changed-during-audit")
            return result

        with self.assertRaisesRegex(CausalScaleAuditError, "changed during audit"):
            audit_causal_ground_scale(self.contract(), cache_validator=mutate_source)
        self.assertFalse(self.output.exists())

    def test_artifact_toctou_is_detected_before_publish(self) -> None:
        def mutate_artifact(
            aggregator: Path,
            camera: Path,
            *,
            expected_num_frames: int,
        ) -> object:
            result = _validate_versioned_cache_pair(
                aggregator,
                camera,
                expected_num_frames=expected_num_frames,
            )
            self.artifact_path.write_bytes(self.artifact_path.read_bytes() + b" ")
            return result

        with self.assertRaisesRegex(CausalScaleAuditError, "SHA256 changed"):
            audit_causal_ground_scale(self.contract(), cache_validator=mutate_artifact)
        self.assertFalse(self.output.exists())

    def test_cache_replacement_toctou_is_detected_before_publish(self) -> None:
        def replace_camera(
            aggregator: Path,
            camera: Path,
            *,
            expected_num_frames: int,
        ) -> object:
            result = _validate_versioned_cache_pair(
                aggregator,
                camera,
                expected_num_frames=expected_num_frames,
            )
            replacement = camera.with_suffix(".replacement")
            replacement.write_bytes(camera.read_bytes())
            os.replace(replacement, camera)
            return result

        with self.assertRaisesRegex(CausalScaleAuditError, "changed during audit"):
            audit_causal_ground_scale(self.contract(), cache_validator=replace_camera)
        self.assertFalse(self.output.exists())

    def test_existing_conflicting_receipt_is_never_overwritten(self) -> None:
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"conflict")
        Path(f"{self.output}.sha256").write_text(
            "0" * 64 + f"  {self.output.name}\n", encoding="ascii"
        )
        with self.assertRaisesRegex(CausalScaleAuditError, "receipt differs"):
            audit_causal_ground_scale(self.contract())
        self.assertEqual(self.output.read_bytes(), b"conflict")


if __name__ == "__main__":
    unittest.main()
