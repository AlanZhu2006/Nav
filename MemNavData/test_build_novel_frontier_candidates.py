import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from MemNavData.build_novel_candidate_manifest import (
    build_manifest,
    canonical_json_bytes as manifest_json_bytes,
)
from MemNavData.build_novel_frontier_candidates import (
    CandidateBuildError,
    DEPLOYMENT_ARM,
    TEACHER_ARM,
    build_artifact,
    sha256_bytes,
    teacher_se2_rows,
    write_artifact,
)
from MemNavData.novel_frontier_candidates_v2 import ProxyMeasurement


M_W = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


class FakeProxyLabeler:
    def provenance(self):
        return {
            "kind": "fake_unit_test_only",
            "real_pathfinder": False,
        }

    def label(self, *, sample_id, arm, candidate):
        del sample_id, arm
        return ProxyMeasurement(
            reachable=True,
            progress_m=float(candidate["subgoal_forward_m"]),
        )


class NovelFrontierBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.episodes = cls.root / "episodes"
        cls.flow = cls.root / "flow"
        cls.environments = cls.root / "environments"
        cls.navmeshes = cls.root / "navmeshes"
        for root in (
                cls.episodes, cls.flow, cls.environments, cls.navmeshes):
            root.mkdir()
        cls.scene = "scene_train"
        cls.split_path = cls.root / "split.json"
        cls.split_path.write_text(json.dumps({
            "version": "fixture_v1",
            "train": [cls.scene],
            "development": [],
            "final_reserved": [],
        }), encoding="utf-8")
        (cls.environments / f"{cls.scene}.glb").write_bytes(b"fake glb")
        (cls.navmeshes / f"{cls.scene}.navmesh").write_bytes(b"fake navmesh")
        for index in range(2):
            cls._make_episode(f"episode_{index:04d}", valid_cache=index == 0)
        preliminary_manifest = build_manifest(
            split_path=cls.split_path,
            episode_root=cls.episodes,
            flow_cache_root=cls.flow,
            environment_root=cls.environments,
            navmesh_root=cls.navmeshes,
            roles=("train",),
        )
        cls._bind_causal_scale_prefix(preliminary_manifest)
        # Rebuild after adding the prefix provenance because the sampling
        # manifest records final flow-cache byte sizes.
        cls.manifest = build_manifest(
            split_path=cls.split_path,
            episode_root=cls.episodes,
            flow_cache_root=cls.flow,
            environment_root=cls.environments,
            navmesh_root=cls.navmeshes,
            roles=("train",),
        )
        cls.manifest_path = cls.root / "causal_manifest.json"
        cls.manifest_bytes = manifest_json_bytes(cls.manifest)
        cls.manifest_path.write_bytes(cls.manifest_bytes)
        cls.manifest_sha = sha256_bytes(cls.manifest_bytes)
        cls.artifact = cls._build()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _bind_causal_scale_prefix(cls, manifest) -> None:
        path = (
            cls.flow / cls.scene / "episode_0000" /
            "videos/chunk-000/lingbot_cam_cache.npz"
        )
        with np.load(path, allow_pickle=False) as source:
            payload = {name: source[name] for name in source.files}
        hashes = np.full(48, "", dtype="<U64")
        for sample in manifest["samples"]:
            if sample["source_episode"] != "episode_0000":
                continue
            hashes[sample["decision_frame"] - 1] = sample["causal_prefix"][
                "causal_prefix_sha256"]
        payload.update({
            "ground_h_est_prefix_causal_prefix_sha256": hashes,
            "ground_h_est_prefix_builder_sha256": np.asarray("a" * 64),
            "ground_h_est_prefix_configuration_sha256": np.asarray("b" * 64),
        })
        np.savez(path, **payload)

    @classmethod
    def _make_episode(cls, name: str, *, valid_cache: bool) -> None:
        episode = cls.episodes / cls.scene / name
        rgb = episode / "videos/chunk-000/observation.images.rgb"
        depth = episode / "videos/chunk-000/observation.images.depth"
        parquet_dir = episode / "data/chunk-000"
        metadata_dir = episode / "meta"
        for path in (rgb, depth, parquet_dir, metadata_dir):
            path.mkdir(parents=True)
        metadata = {
            "scene": f"{cls.scene}.glb",
            "ep_idx": int(name.rsplit("_", 1)[1]),
            "n_frames": 48,
            "n_legs": 3,
            "switches": [8, 40],
            "camera_height_m": 0.5,
            "frame_convention": "positions+parquet in data(Zup,M_W)",
            "goals": [
                {"name": "B", "kind": "novel", "pos": [1.0, 0.0, 0.0]},
                {"name": "C", "kind": "revisit", "pos": [0.0, 0.0, 0.0]},
            ],
        }
        (metadata_dir / "gen_meta.json").write_text(
            json.dumps(metadata), encoding="utf-8")
        Image.new("RGB", (32, 24), (120, 80, 40)).save(episode / "goal_1.jpg")
        Image.new("RGB", (32, 24), (40, 80, 120)).save(episode / "goal_2.jpg")
        intrinsic = np.asarray([
            [25.0, 0.0, 15.5],
            [0.0, 25.0, 11.5],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = M_W.astype(np.float32)
        extrinsic[:3, 3] = M_W @ np.asarray([0.0, 0.5, 0.0])
        rows = []
        for frame in range(48):
            Image.new("RGB", (32, 24), (frame, 20, 30)).save(
                rgb / f"{frame}.jpg")
            # A shallow vertical obstacle band plus max-range background gives
            # deterministic free/unknown boundaries without Habitat.
            depth_array = np.full((24, 32), 60000, dtype=np.uint16)
            depth_array[:, 12:20] = 25000
            Image.fromarray(depth_array).save(depth / f"{frame}.png")
            action = np.eye(4, dtype=np.float32)
            action[:3, :3] = M_W.astype(np.float32)
            action[1, 3] = frame * 0.05
            rows.append({
                "index": frame,
                "observation.camera_intrinsic": intrinsic.tolist(),
                "observation.camera_extrinsic": extrinsic.tolist(),
                "action": action.tolist(),
            })
        pq.write_table(
            pa.Table.from_pylist(rows),
            parquet_dir / "episode_000000.parquet",
        )

        flow_chunk = cls.flow / cls.scene / name / "videos/chunk-000"
        flow_chunk.mkdir(parents=True)
        num_frames, num_scale, interval = 48, 8, 4
        anchor_indices = np.arange(num_scale, num_frames, interval, dtype=np.int64)
        cam_indices = np.concatenate((
            np.arange(num_scale, dtype=np.int64), anchor_indices))
        shared = {
            "cache_schema_version": np.asarray([2], dtype=np.int64),
            "keyframe_policy": np.asarray(["post_scale_mod_v1"]),
            "num_frames": np.asarray([num_frames], dtype=np.int64),
            "num_scale_frames": np.asarray([num_scale], dtype=np.int64),
            "keyframe_interval": np.asarray([interval], dtype=np.int64),
            "precompute_signature": np.asarray(["fixture_versioned"]),
        }
        np.savez(
            flow_chunk / "lingbot_cache.npz",
            dino_cls=np.zeros((num_frames, 1), dtype=np.float16),
            anchor_k=np.zeros((len(anchor_indices), 1), dtype=np.float16),
            anchor_v=np.zeros((len(anchor_indices), 1), dtype=np.float16),
            anchor_frame_indices=anchor_indices,
            **shared,
        )
        raw_pose = np.zeros((48, 9), dtype=np.float32)
        raw_pose[:, 2] = np.arange(48, dtype=np.float32) * 0.025
        raw_pose[:, 6] = 1.0  # xyzw identity, optical +z forward
        if valid_cache:
            np.savez(
                flow_chunk / "lingbot_cam_cache.npz",
                cam_pose_enc=raw_pose,
                cam_k=np.zeros((len(cam_indices), 1), dtype=np.float16),
                cam_v=np.zeros((len(cam_indices), 1), dtype=np.float16),
                cam_frame_indices=cam_indices,
                ground_h_est=np.asarray(0.25, dtype=np.float32),
                ground_h_est_prefix=np.full(
                    num_frames, 0.25, dtype=np.float32),
                ground_h_est_prefix_frame_indices=np.arange(
                    num_frames, dtype=np.int64),
                ground_h_est_prefix_semantics=np.asarray(
                    "causal_prefix_floor_hist_v1"),
                **shared,
            )
        else:
            # A structurally valid versioned pair with only the current
            # whole-episode scale must still fail the causal deployment arm.
            np.savez(
                flow_chunk / "lingbot_cam_cache.npz",
                cam_pose_enc=raw_pose,
                cam_k=np.zeros((len(cam_indices), 1), dtype=np.float16),
                cam_v=np.zeros((len(cam_indices), 1), dtype=np.float16),
                cam_frame_indices=cam_indices,
                ground_h_est=np.asarray(0.25, dtype=np.float32),
                **shared,
            )

    @classmethod
    def _build(cls, **kwargs):
        kwargs.setdefault("expected_ground_prefix_builder_sha256", "a" * 64)
        kwargs.setdefault(
            "expected_ground_prefix_configuration_sha256", "b" * 64)
        return build_artifact(
            manifest=cls.manifest,
            manifest_path=cls.manifest_path,
            manifest_sha256=cls.manifest_sha,
            depth_column_stride=4,
            scan_stride=4,
            **kwargs,
        )

    def test_legacy_identity_and_fixed_mount_decode_to_identical_teacher_se2(self):
        fixed_rows = []
        legacy_rows = []
        for frame, yaw in enumerate((0.0, 0.4, -1.1)):
            c, s = np.cos(yaw), np.sin(yaw)
            base = np.asarray([
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ])
            action = np.eye(4)
            action[:3, :3] = base @ M_W
            action[:2, 3] = [frame * 0.7, -frame * 0.2]
            fixed_mount = np.eye(4)
            fixed_mount[:3, :3] = M_W
            legacy_mount = np.eye(4)
            fixed_rows.append({
                "action": action.tolist(),
                "observation.camera_extrinsic": fixed_mount.tolist(),
            })
            legacy_rows.append({
                "action": action.tolist(),
                "observation.camera_extrinsic": legacy_mount.tolist(),
            })
        convention = (
            "positions+parquet in data(Zup,M_W); "
            "yaw_habitat in render frame")
        fixed = teacher_se2_rows(
            fixed_rows, decision_frame=3, frame_convention=convention)
        legacy = teacher_se2_rows(
            legacy_rows, decision_frame=3, frame_convention=convention)
        self.assertEqual(fixed, legacy)
        for frame, yaw in enumerate((0.0, 0.4, -1.1)):
            expected_heading = np.arctan2(np.cos(yaw), -np.sin(yaw))
            self.assertAlmostEqual(fixed[frame].yaw_rad, expected_heading)
            self.assertAlmostEqual(fixed[frame].x_m, frame * 0.7)
            self.assertAlmostEqual(fixed[frame].y_m, -frame * 0.2)

    def test_causal_scale_producer_hashes_require_explicit_exact_pins(self):
        artifact = build_artifact(
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha,
            depth_column_stride=4,
            scan_stride=4,
        )
        valid_episode_rows = [
            row for row in artifact["records"]
            if row["source_episode"] == "episode_0000"
        ]
        self.assertTrue(valid_episode_rows)
        self.assertTrue(all(
            not row["arms"][DEPLOYMENT_ARM]["proposal"]["valid"]
            and "is not pinned" in row["arms"][DEPLOYMENT_ARM][
                "proposal"]["invalid_reason"]
            for row in valid_episode_rows
        ))
        wrong = self._build(
            expected_ground_prefix_builder_sha256="c" * 64)
        self.assertTrue(all(
            not row["arms"][DEPLOYMENT_ARM]["proposal"]["valid"]
            for row in wrong["records"]
            if row["source_episode"] == "episode_0000"
        ))

    def test_two_pose_arms_are_separate_and_deployment_fails_closed(self):
        records = self.artifact["records"]
        self.assertEqual(len(records), 8)
        self.assertEqual(
            self.artifact["summary"]["arms"][TEACHER_ARM]["valid_sample_count"],
            8,
        )
        self.assertEqual(
            self.artifact["summary"]["arms"][DEPLOYMENT_ARM]["valid_sample_count"],
            4,
        )
        for record in records:
            teacher = record["arms"][TEACHER_ARM]
            deployment = record["arms"][DEPLOYMENT_ARM]
            self.assertTrue(teacher["proposal"]["valid"])
            self.assertFalse(teacher["deployment_eligible_pose_source"])
            self.assertFalse(teacher["pose_provenance"]["gt_sim2_used"])
            self.assertLessEqual(teacher["proposal"]["shortlist_count"], 6)
            if record["source_episode"] == "episode_0000":
                self.assertTrue(deployment["proposal"]["valid"])
                self.assertFalse(deployment["pose_provenance"]["gt_sim2_used"])
                self.assertEqual(
                    deployment["pose_provenance"]["metric_scale_source"],
                    "clamp(1.15 * camera_height_m / causal_ground_h_est_prefix, 0.8, 6.0)",
                )
                self.assertAlmostEqual(
                    deployment["pose_provenance"]["metric_scale_m_per_raw"],
                    2.3,
                )
                self.assertIn(
                    "dense_cam_pose_enc_row_equals_raw_frame",
                    deployment["pose_provenance"]["pose_frame_mapping"],
                )
                self.assertEqual(
                    deployment["pose_provenance"][
                        "ground_h_est_prefix_causal_prefix_sha256"],
                    record["causal_prefix_sha256"],
                )
            else:
                self.assertFalse(deployment["proposal"]["valid"])
                self.assertIn(
                    "whole-episode ground_h_est",
                    deployment["proposal"]["invalid_reason"],
                )
                self.assertEqual(deployment["proposal"]["shortlist"], [])

    def test_missing_patch_uses_masked_degraded_shortlist(self):
        self.assertEqual(self.artifact["provenance"]["patch_scores"]["status"],
                         "absent")
        for record in self.artifact["records"]:
            for arm in (TEACHER_ARM, DEPLOYMENT_ARM):
                proposal = record["arms"][arm]["proposal"]
                self.assertEqual(proposal["goal_patch_relation_mask"], 0)
                for candidate in proposal["shortlist"]:
                    self.assertNotIn(
                        "goal_patch_top2", candidate["selection_sources"])

    def test_patch_artifact_is_bound_to_goal_prefix_and_builder_hashes(self):
        records = []
        for sample in self.manifest["samples"]:
            records.append({
                "sample_id": sample["sample_id"],
                "causal_prefix_sha256": sample["causal_prefix"][
                    "causal_prefix_sha256"],
                "goal_sha256": sample["goal"]["content_sha256"],
                "frame_scores": [
                    {"frame_index": frame, "score": 0.5 - frame / 1000.0}
                    for frame in range(sample["decision_frame"])
                ],
            })
        patch = {
            "schema_version": "nlsr_v2_goal_patch_frame_scores_v1",
            "input_manifest_sha256": self.manifest_sha,
            "encoder_checkpoint_sha256": "1" * 64,
            "feature_builder_sha256": "2" * 64,
            "configuration_sha256": "3" * 64,
            "records": records,
        }
        path = self.root / "patch_scores.json"
        path.write_text(json.dumps(patch), encoding="utf-8")
        with self.assertRaisesRegex(CandidateBuildError, "explicit expected"):
            self._build(patch_score_path=path)
        artifact = self._build(
            patch_score_path=path,
            expected_patch_encoder_checkpoint_sha256="1" * 64,
            expected_patch_feature_builder_sha256="2" * 64,
            expected_patch_configuration_sha256="3" * 64,
        )
        self.assertEqual(
            artifact["provenance"]["patch_scores"][
                "feature_builder_sha256"],
            "2" * 64,
        )
        self.assertTrue(any(
            "goal_patch_top2" in candidate["selection_sources"]
            for record in artifact["records"]
            for candidate in record["arms"][TEACHER_ARM]["proposal"]["shortlist"]
        ))

        patch["records"][0]["goal_sha256"] = "4" * 64
        path.write_text(json.dumps(patch), encoding="utf-8")
        with self.assertRaisesRegex(CandidateBuildError, "goal content mismatch"):
            self._build(
                patch_score_path=path,
                expected_patch_encoder_checkpoint_sha256="1" * 64,
                expected_patch_feature_builder_sha256="2" * 64,
                expected_patch_configuration_sha256="3" * 64,
            )

    def test_no_proxy_claim_without_labeler_and_fake_labels_remain_isolated(self):
        self.assertFalse(self.artifact["summary"]["real_pathfinder_claim"])
        for record in self.artifact["records"]:
            self.assertEqual(
                record["arms"][TEACHER_ARM]["proposal_proxy"]["status"],
                "not_requested",
            )
        labeled = self._build(labeler=FakeProxyLabeler())
        self.assertFalse(labeled["summary"]["real_pathfinder_claim"])
        self.assertTrue(labeled["summary"]["proposal_proxy_labeler_supplied"])
        for record in labeled["records"]:
            proposal = record["arms"][TEACHER_ARM]["proposal"]
            proxy = record["arms"][TEACHER_ARM]["proposal_proxy"]
            self.assertEqual(proxy["status"], "labeled")
            self.assertEqual(
                proxy["proposal_sha256"],
                record["arms"][TEACHER_ARM]["proposal_proxy"]["proposal_sha256"],
            )
            self.assertNotIn("progress_m", str(proposal))

    def test_atomic_resume_accepts_only_identical_artifact(self):
        output = self.root / "output/proposals.json"
        sidecar = self.root / "output/proposals.json.sha256"
        status, digest = write_artifact(self.artifact, output, sidecar)
        self.assertEqual(status, "written")
        resumed, resumed_digest = write_artifact(
            self.artifact, output, sidecar, resume=True)
        self.assertEqual((resumed, resumed_digest), ("resumed", digest))
        changed = dict(self.artifact)
        changed["purpose"] = "changed"
        with self.assertRaisesRegex(CandidateBuildError, "differs"):
            write_artifact(changed, output, sidecar, resume=True)

    def test_manifest_pin_is_required(self):
        with self.assertRaisesRegex(CandidateBuildError, "SHA mismatch"):
            build_artifact(
                manifest=self.manifest,
                manifest_path=self.manifest_path,
                manifest_sha256="0" * 64,
            )

        changed = dict(self.manifest)
        changed["purpose"] = "in-memory substitution"
        with self.assertRaisesRegex(CandidateBuildError, "in-memory manifest"):
            build_artifact(
                manifest=changed,
                manifest_path=self.manifest_path,
                manifest_sha256=self.manifest_sha,
            )

    def test_changed_prefix_depth_is_rejected(self):
        depth = (
            self.episodes / self.scene / "episode_0000" /
            "videos/chunk-000/observation.images.depth/3.png"
        )
        original = depth.read_bytes()
        try:
            Image.fromarray(np.full((24, 32), 12345, dtype=np.uint16)).save(depth)
            with self.assertRaisesRegex(CandidateBuildError, "causal prefix content"):
                self._build()
        finally:
            depth.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
