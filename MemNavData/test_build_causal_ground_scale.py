import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from MemNavData.build_causal_ground_scale import (
    CausalScaleError,
    GroundScaleConfiguration,
    GroundScaleEstimate,
    build_scale_artifact,
    expected_scale_from_ground,
)
from MemNavData.build_novel_frontier_candidates import (
    CandidateBuildError,
    INPUT_MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    write_artifact,
)


class FakeScaleEstimator:
    def __init__(self, ground_h_est_raw=0.25, *, partial_invalid=False):
        self.ground_h_est_raw = ground_h_est_raw
        self.partial_invalid = partial_invalid
        self.calls = []

    def provenance(self):
        return {
            "kind": "fake_causal_scale_unit_test",
            "weights_sha256": "f" * 64,
        }

    def estimate(
        self,
        *,
        rgb_paths,
        cam_pose_prefix,
        camera_height_m,
        configuration,
    ):
        rgb_digest = hashlib.sha256()
        for path in rgb_paths:
            rgb_digest.update(path.read_bytes())
        pose = np.asarray(cam_pose_prefix).copy()
        self.calls.append({
            "rgb_paths": tuple(rgb_paths),
            "pose": pose,
            "camera_height_m": camera_height_m,
        })
        debug = {
            "rgb_content_sha256": rgb_digest.hexdigest(),
            "prefix_frames": len(rgb_paths),
            "pose_sum": float(pose.sum()),
        }
        if self.ground_h_est_raw is None:
            return GroundScaleEstimate(
                None,
                1.0 if self.partial_invalid else None,
                debug,
            )
        return GroundScaleEstimate(
            float(self.ground_h_est_raw),
            expected_scale_from_ground(
                float(self.ground_h_est_raw),
                float(camera_height_m),
                configuration,
            ),
            debug,
        )


class VersionedCacheValidator:
    def __init__(self):
        self.calls = []

    def __call__(self, aggregator_path, camera_path, frame_count):
        self.calls.append((aggregator_path, camera_path, frame_count))
        with np.load(aggregator_path, allow_pickle=False) as cache:
            self._assert_versioned(cache, frame_count)
            if "dino_cls" not in cache.files:
                raise AssertionError("fixture aggregator cache is not dense")
        with np.load(camera_path, allow_pickle=False) as cache:
            self._assert_versioned(cache, frame_count)
            if cache["cam_pose_enc"].shape != (frame_count, 9):
                raise AssertionError("fixture camera cache is not dense")
        return {"validated": True}

    @staticmethod
    def _assert_versioned(cache, frame_count):
        required = {
            "cache_schema_version",
            "num_frames",
            "num_scale_frames",
            "keyframe_interval",
            "keyframe_policy",
            "precompute_signature",
        }
        if not required.issubset(cache.files):
            raise AssertionError("fixture cache is not versioned")
        if int(np.asarray(cache["num_frames"]).reshape(-1)[0]) != frame_count:
            raise AssertionError("fixture cache frame count changed")


class CausalGroundScaleBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episode_root = self.root / "episodes"
        self.flow_root = self.root / "flow"
        self.scene = "scene_train"
        self.episode = "episode_0000"
        self.frame_count = 80
        self.episode_dir = self.episode_root / self.scene / self.episode
        self.rgb_dir = (
            self.episode_dir / "videos/chunk-000/observation.images.rgb"
        )
        self.metadata_dir = self.episode_dir / "meta"
        self.flow_dir = self.flow_root / self.scene / self.episode / "videos/chunk-000"
        self.rgb_dir.mkdir(parents=True)
        self.metadata_dir.mkdir(parents=True)
        self.flow_dir.mkdir(parents=True)
        for frame in range(self.frame_count):
            Image.new(
                "RGB", (8, 6), (frame % 251, (frame * 3) % 251, 17)
            ).save(self.rgb_dir / f"{frame}.jpg")
        (self.metadata_dir / "gen_meta.json").write_text(
            json.dumps({
                "n_frames": self.frame_count,
                "camera_height_m": 0.5,
            }),
            encoding="utf-8",
        )
        self.camera_path = self.flow_dir / "lingbot_cam_cache.npz"
        self.aggregator_path = self.flow_dir / "lingbot_cache.npz"
        pose = np.zeros((self.frame_count, 9), dtype=np.float32)
        pose[:, 2] = np.arange(self.frame_count, dtype=np.float32) / 10.0
        pose[:, 6] = 1.0
        self._write_camera_pose(pose)
        shared = self._version_fields()
        np.savez(
            self.aggregator_path,
            dino_cls=np.zeros((self.frame_count, 4), dtype=np.float16),
            anchor_k=np.zeros((1, 1), dtype=np.float16),
            anchor_v=np.zeros((1, 1), dtype=np.float16),
            anchor_frame_indices=np.asarray([8], dtype=np.int64),
            **shared,
        )
        self.manifest = self._manifest(decisions=(70, 75))
        self.manifest_path = self.root / "manifest.json"
        self.manifest_sha = self._write_manifest(self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def _version_fields(self):
        return {
            "cache_schema_version": np.asarray([2], dtype=np.int64),
            "keyframe_policy": np.asarray(["post_scale_mod_v1"]),
            "num_frames": np.asarray([self.frame_count], dtype=np.int64),
            "num_scale_frames": np.asarray([8], dtype=np.int64),
            "keyframe_interval": np.asarray([4], dtype=np.int64),
            "precompute_signature": np.asarray(["causal_scale_fixture_v1"]),
        }

    def _write_camera_pose(self, pose):
        shared = self._version_fields()
        indices = np.arange(self.frame_count, dtype=np.int64)
        np.savez(
            self.camera_path,
            cam_pose_enc=np.asarray(pose, dtype=np.float32),
            cam_k=np.zeros((self.frame_count, 1), dtype=np.float16),
            cam_v=np.zeros((self.frame_count, 1), dtype=np.float16),
            cam_frame_indices=indices,
            **shared,
        )

    def _manifest(self, *, decisions):
        camera_relative = self.camera_path.relative_to(self.flow_root).as_posix()
        aggregator_relative = self.aggregator_path.relative_to(
            self.flow_root
        ).as_posix()
        metadata_path = self.metadata_dir / "gen_meta.json"
        metadata_relative = metadata_path.relative_to(
            self.episode_root
        ).as_posix()
        return {
            "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
            "input_roots": {
                "episode_root": str(self.episode_root),
                "flow_cache_root": str(self.flow_root),
            },
            "scenes": [{
                "scene": self.scene,
                "split_role": "train",
                "selected_episodes": [{
                    "episode": self.episode,
                    "n_frames": self.frame_count,
                    "metadata": {
                        "path": metadata_relative,
                        "bytes": metadata_path.stat().st_size,
                        "content_sha256": hashlib.sha256(
                            metadata_path.read_bytes()
                        ).hexdigest(),
                    },
                    "flow_cache": {
                        "files": [
                            {
                                "path": aggregator_relative,
                                "bytes": self.aggregator_path.stat().st_size,
                            },
                            {
                                "path": camera_relative,
                                "bytes": self.camera_path.stat().st_size,
                            },
                        ],
                    },
                }],
            }],
            "samples": [
                {
                    "sample_id": f"sample_{index}",
                    "scene": self.scene,
                    "source_episode": self.episode,
                    "decision_frame": decision,
                    "split_role": "train",
                }
                for index, decision in enumerate(decisions)
            ],
        }

    def _write_manifest(self, manifest, path=None):
        path = path or self.manifest_path
        payload = canonical_json_bytes(manifest)
        path.write_bytes(payload)
        return sha256_bytes(payload)

    def _build(self, *, manifest=None, path=None, digest=None, estimator=None):
        validator = VersionedCacheValidator()
        artifact = build_scale_artifact(
            manifest=manifest or self.manifest,
            manifest_path=path or self.manifest_path,
            expected_manifest_sha256=digest or self.manifest_sha,
            estimator=estimator or FakeScaleEstimator(),
            cache_pair_validator=validator,
        )
        self.assertEqual(len(validator.calls), 1)
        return artifact

    def test_uses_only_minimum_of_cap_and_earliest_decision(self):
        estimator = FakeScaleEstimator()
        artifact = self._build(estimator=estimator)
        record = artifact["records"][0]
        self.assertEqual(record["decision_frames"], [70, 75])
        self.assertEqual(record["earliest_decision_frame"], 70)
        self.assertEqual(record["prefix_end_frame_exclusive"], 64)
        self.assertEqual(record["rgb_prefix"]["frame_count"], 64)
        self.assertEqual(len(estimator.calls[0]["rgb_paths"]), 64)
        self.assertEqual(estimator.calls[0]["pose"].shape, (64, 9))
        self.assertEqual(
            estimator.calls[0]["rgb_paths"][-1].name,
            "63.jpg",
        )
        self.assertEqual(artifact["summary"]["future_frames_consumed"], 0)

        short_manifest = self._manifest(decisions=(20, 35))
        short_path = self.root / "manifest_short.json"
        short_sha = self._write_manifest(short_manifest, short_path)
        short_estimator = FakeScaleEstimator()
        short_artifact = self._build(
            manifest=short_manifest,
            path=short_path,
            digest=short_sha,
            estimator=short_estimator,
        )
        self.assertEqual(
            short_artifact["records"][0]["prefix_end_frame_exclusive"], 20
        )
        self.assertEqual(short_estimator.calls[0]["pose"].shape, (20, 9))

    def test_future_rgb_mutation_keeps_canonical_artifact_identical(self):
        before = self._build()
        before_bytes = canonical_json_bytes(before)
        Image.new("RGB", (8, 6), (255, 255, 255)).save(
            self.rgb_dir / "64.jpg"
        )
        after = self._build()
        self.assertEqual(canonical_json_bytes(after), before_bytes)

    def test_future_pose_mutation_keeps_canonical_artifact_identical(self):
        before_bytes = canonical_json_bytes(self._build())
        with np.load(self.camera_path, allow_pickle=False) as source:
            pose = np.asarray(source["cam_pose_enc"]).copy()
        pose[64, 0] = 999.0
        self._write_camera_pose(pose)
        after_bytes = canonical_json_bytes(self._build())
        self.assertEqual(after_bytes, before_bytes)

    def test_prefix_rgb_and_pose_mutations_change_artifact(self):
        baseline = canonical_json_bytes(self._build())
        Image.new("RGB", (8, 6), (255, 255, 255)).save(
            self.rgb_dir / "4.jpg"
        )
        rgb_changed = canonical_json_bytes(self._build())
        self.assertNotEqual(rgb_changed, baseline)

        with np.load(self.camera_path, allow_pickle=False) as source:
            pose = np.asarray(source["cam_pose_enc"]).copy()
        old_size = self.camera_path.stat().st_size
        pose[5, 0] = 123.0
        self._write_camera_pose(pose)
        self.assertEqual(self.camera_path.stat().st_size, old_size)
        pose_changed = canonical_json_bytes(self._build())
        self.assertNotEqual(pose_changed, rgb_changed)

    def test_scale_formula_and_clipping_are_enforced(self):
        configuration = GroundScaleConfiguration(
            bias_correction=1.15,
            scale_min=0.8,
            scale_max=6.0,
        )
        self.assertAlmostEqual(
            expected_scale_from_ground(0.25, 0.5, configuration), 2.3
        )
        self.assertEqual(
            expected_scale_from_ground(100.0, 0.5, configuration), 0.8
        )
        self.assertEqual(
            expected_scale_from_ground(0.001, 0.5, configuration), 6.0
        )
        artifact = self._build(estimator=FakeScaleEstimator(0.25))
        self.assertAlmostEqual(
            artifact["records"][0]["metric_scale_m_per_raw"], 2.3
        )

        class WrongScaleEstimator(FakeScaleEstimator):
            def estimate(self, **kwargs):
                estimate = super().estimate(**kwargs)
                return GroundScaleEstimate(
                    estimate.ground_h_est_raw, 2.31, estimate.debug
                )

        with self.assertRaisesRegex(CausalScaleError, "formula"):
            self._build(estimator=WrongScaleEstimator(0.25))

    def test_invalid_estimate_neutralizes_both_values(self):
        artifact = self._build(estimator=FakeScaleEstimator(None))
        record = artifact["records"][0]
        self.assertFalse(record["valid"])
        self.assertIsNone(record["ground_h_est_raw"])
        self.assertIsNone(record["metric_scale_m_per_raw"])
        with self.assertRaisesRegex(CausalScaleError, "neutralize both"):
            self._build(
                estimator=FakeScaleEstimator(None, partial_invalid=True)
            )

    def test_earliest_decision_before_scale_block_fails(self):
        manifest = self._manifest(decisions=(7, 20))
        path = self.root / "manifest_early.json"
        digest = self._write_manifest(manifest, path)
        with self.assertRaisesRegex(CausalScaleError, "complete scale block"):
            self._build(manifest=manifest, path=path, digest=digest)

    def test_manifest_sha_memory_roles_and_duplicate_episodes_are_rejected(self):
        with self.assertRaisesRegex(CausalScaleError, "SHA changed"):
            self._build(digest="0" * 64)

        changed_memory = copy.deepcopy(self.manifest)
        changed_memory["samples"][0]["sample_id"] = "memory_only_change"
        with self.assertRaisesRegex(CausalScaleError, "in-memory manifest"):
            self._build(manifest=changed_memory)

        crossed_roles = copy.deepcopy(self.manifest)
        crossed_roles["samples"][1]["split_role"] = "development"
        role_path = self.root / "manifest_roles.json"
        role_sha = self._write_manifest(crossed_roles, role_path)
        with self.assertRaisesRegex(CausalScaleError, "roles"):
            self._build(
                manifest=crossed_roles, path=role_path, digest=role_sha
            )

        duplicated_episode = copy.deepcopy(self.manifest)
        duplicated_episode["scenes"][0]["selected_episodes"].append(
            copy.deepcopy(
                duplicated_episode["scenes"][0]["selected_episodes"][0]
            )
        )
        duplicate_path = self.root / "manifest_duplicate_episode.json"
        duplicate_sha = self._write_manifest(duplicated_episode, duplicate_path)
        with self.assertRaisesRegex(CausalScaleError, "duplicated"):
            self._build(
                manifest=duplicated_episode,
                path=duplicate_path,
                digest=duplicate_sha,
            )

    def test_duplicate_sample_identity_is_rejected(self):
        duplicated_sample = copy.deepcopy(self.manifest)
        duplicated_sample["samples"][1]["sample_id"] = duplicated_sample[
            "samples"
        ][0]["sample_id"]
        path = self.root / "manifest_duplicate_sample.json"
        digest = self._write_manifest(duplicated_sample, path)
        with self.assertRaisesRegex(CausalScaleError, "duplicat"):
            self._build(
                manifest=duplicated_sample,
                path=path,
                digest=digest,
            )

    def test_write_artifact_resume_is_byte_exact(self):
        artifact = self._build()
        output = self.root / "scale.json"
        sha_output = self.root / "scale.json.sha256"
        status, digest = write_artifact(artifact, output, sha_output)
        self.assertEqual(status, "written")
        self.assertEqual(output.read_bytes(), canonical_json_bytes(artifact))
        status, resumed_digest = write_artifact(
            artifact, output, sha_output, resume=True
        )
        self.assertEqual(status, "resumed")
        self.assertEqual(resumed_digest, digest)
        output.write_bytes(output.read_bytes() + b"\n")
        with self.assertRaisesRegex(CandidateBuildError, "differs"):
            write_artifact(artifact, output, sha_output, resume=True)


if __name__ == "__main__":
    unittest.main()
