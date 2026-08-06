import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from MemNavData.build_novel_candidate_manifest import (
    ManifestError,
    aligned_midpoint,
    build_manifest,
    canonical_json_bytes,
    sha256_bytes,
    write_artifact,
)


class NovelCandidateManifestBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episodes = self.root / "episodes"
        self.flow = self.root / "flow"
        self.environments = self.root / "environments"
        self.navmeshes = self.root / "navmeshes"
        for root in (
                self.episodes, self.flow, self.environments, self.navmeshes):
            root.mkdir()
        self.split_path = self.root / "router_multiscene_split_20260805.json"
        self.split = {
            "version": "fixture_split_v1",
            "train": ["scene_train"],
            "development": ["scene_development"],
            "final_reserved": ["scene_final"],
        }
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")
        for scene in ("scene_train", "scene_development"):
            (self.environments / f"{scene}.glb").write_bytes(
                f"environment:{scene}".encode())
            nested_navmesh = self.navmeshes / scene
            nested_navmesh.mkdir()
            (nested_navmesh / f"{scene}.navmesh").write_bytes(
                f"navmesh:{scene}".encode())
            for number in range(2):
                self._episode(scene, f"episode_{number:04d}")
                if number == 0:
                    self._complete_flow(scene, f"episode_{number:04d}")

    def tearDown(self):
        self.temporary.cleanup()

    def _episode(self, scene: str, name: str) -> Path:
        episode = self.episodes / scene / name
        (episode / "meta").mkdir(parents=True)
        (episode / "data/chunk-000").mkdir(parents=True)
        rgb = episode / "videos/chunk-000/observation.images.rgb"
        depth = episode / "videos/chunk-000/observation.images.depth"
        rgb.mkdir(parents=True)
        depth.mkdir(parents=True)
        metadata = {
            "scene": f"{scene}.glb",
            "ep_idx": int(name.rsplit("_", 1)[1]),
            "n_frames": 48,
            "n_legs": 3,
            "switches": [8, 40],
            "goals": [
                {"name": "B", "kind": "novel", "pos": [1, 2, 3]},
                {"name": "C", "kind": "revisit", "pos": [3, 2, 1]},
            ],
        }
        (episode / "meta/gen_meta.json").write_text(
            json.dumps(metadata), encoding="utf-8")
        self._write_parquet(episode)
        (episode / "goal_1.jpg").write_bytes(
            f"goal-b:{scene}:{name}".encode())
        (episode / "goal_2.jpg").write_bytes(
            f"goal-c:{scene}:{name}".encode())
        for frame in range(48):
            (rgb / f"{frame}.jpg").write_bytes(
                f"rgb:{scene}:{name}:{frame}".encode())
            (depth / f"{frame}.png").write_bytes(
                f"depth:{scene}:{name}:{frame}".encode())
        return episode

    @staticmethod
    def _write_parquet(episode: Path, *, changed_row: int | None = None,
                       delta: float = 0.0) -> None:
        intrinsic = [
            [355.0, 0.0, 240.0],
            [0.0, 351.0, 135.0],
            [0.0, 0.0, 1.0],
        ]
        extrinsic = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
        rows = []
        for frame in range(48):
            action = [
                [1.0, 0.0, 0.0, float(frame)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            if frame == changed_row:
                action[0][3] += delta
            rows.append({
                "index": frame,
                "observation.camera_intrinsic": intrinsic,
                "observation.camera_extrinsic": extrinsic,
                "action": action,
            })
        table = pa.Table.from_pylist(rows)
        pq.write_table(
            table,
            episode / "data/chunk-000/episode_000000.parquet",
        )

    def _complete_flow(self, scene: str, name: str) -> None:
        chunk = self.flow / scene / name / "videos/chunk-000"
        chunk.mkdir(parents=True)
        (chunk / "lingbot_cache.npz").write_bytes(b"aggregate")
        (chunk / "lingbot_cam_cache.npz").write_bytes(b"camera")

    def _build(self, **kwargs):
        return build_manifest(
            split_path=self.split_path,
            episode_root=self.episodes,
            flow_cache_root=self.flow,
            environment_root=self.environments,
            navmesh_root=self.navmeshes,
            **kwargs,
        )

    def test_build_is_deterministic_and_has_canonical_hash(self):
        first = self._build()
        second = self._build()
        self.assertEqual(first, second)
        first_bytes = canonical_json_bytes(first)
        second_bytes = canonical_json_bytes(second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(sha256_bytes(first_bytes), sha256_bytes(second_bytes))
        self.assertEqual(first["summary"]["scene_count"], 2)
        self.assertEqual(first["summary"]["episode_count"], 4)
        self.assertEqual(first["summary"]["sample_count"], 16)

    def test_scene_roles_and_goals_never_cross_scene(self):
        manifest = self._build()
        expected_roles = {
            "scene_train": "train",
            "scene_development": "development",
        }
        self.assertEqual(
            {row["scene"]: row["split_role"] for row in manifest["scenes"]},
            expected_roles,
        )
        self.assertNotIn("scene_final", {
            sample["scene"] for sample in manifest["samples"]})
        for sample in manifest["samples"]:
            self.assertEqual(sample["split_role"], expected_roles[sample["scene"]])
            self.assertTrue(sample["goal"]["path"].startswith(
                f"{sample['scene']}/"))
            self.assertTrue(sample["state_frame"]["path"].startswith(
                f"{sample['scene']}/"))
            self.assertEqual(sample["state_source"], "expert")
            self.assertEqual(
                sample["source_episode_id"],
                f"{sample['scene']}/{sample['source_episode']}",
            )
            self.assertEqual(
                sample["goal_source_episode_id"],
                f"{sample['scene']}/{sample['goal_episode']}",
            )
        # Both scenes intentionally use identical short episode names.  The
        # globally qualified identifiers must remain distinct.
        self.assertGreater(
            len({sample["source_episode_id"] for sample in manifest["samples"]}),
            len({sample["source_episode"] for sample in manifest["samples"]}),
        )

    def test_counterfactual_is_fixed_other_episode_and_states_are_aligned(self):
        manifest = self._build(roles=("train",))
        samples = manifest["samples"]
        self.assertEqual(len(samples), 8)
        by_state = {}
        for sample in samples:
            key = (sample["source_episode"], sample["state_name"])
            by_state.setdefault(key, {})[sample["goal_variant"]] = sample
        self.assertEqual(len(by_state), 4)
        for (source_episode, state_name), variants in by_state.items():
            self.assertEqual(set(variants), {"factual", "counterfactual"})
            self.assertEqual(variants["factual"]["goal_episode"], source_episode)
            self.assertNotEqual(
                variants["counterfactual"]["goal_episode"], source_episode)
            self.assertEqual(
                variants["factual"]["causal_prefix"],
                variants["counterfactual"]["causal_prefix"],
            )
            decision = variants["factual"]["decision_frame"]
            if state_name == "goal_b_t0":
                self.assertEqual(decision, 8)
            else:
                self.assertEqual(decision, 24)
                self.assertEqual((decision - 8) % 8, 0)
            self.assertEqual(
                variants["factual"]["causal_prefix"]["frame_count"],
                decision,
            )
            prefix = variants["factual"]["causal_prefix"]
            self.assertEqual(set(prefix["modalities"]), {"rgb", "depth"})
            self.assertEqual(prefix["parquet_row_count"], decision)
            self.assertTrue(prefix["parquet_rows_sha256"])
            self.assertTrue(prefix["causal_prefix_sha256"])
            self.assertTrue(variants["factual"]["goal"]["path_sha256"])
            self.assertTrue(variants["factual"]["goal"]["content_sha256"])

    def test_midpoint_uses_earlier_frame_on_equal_distance(self):
        self.assertEqual(aligned_midpoint(7, 31), 15)
        with self.assertRaises(ManifestError):
            aligned_midpoint(7, 15)

    def test_missing_flow_caches_are_explicit_not_silently_dropped(self):
        manifest = self._build(roles=("train",))
        missing = manifest["missing_flow_caches"]
        self.assertEqual(len(missing), 2)
        self.assertEqual(
            {row["cache_file"] for row in missing},
            {"lingbot_cache.npz", "lingbot_cam_cache.npz"},
        )
        self.assertEqual({row["episode"] for row in missing}, {"episode_0001"})
        self.assertFalse(manifest["summary"]["all_flow_caches_complete"])
        self.assertEqual(
            manifest["scenes"][0]["selected_episodes"][1]["flow_cache"]
            ["complete"],
            False,
        )

    def test_incomplete_depth_sequence_cannot_be_selected(self):
        missing = (
            self.episodes /
            "scene_train/episode_0000/videos/chunk-000/"
            "observation.images.depth/7.png"
        )
        missing.unlink()
        with self.assertRaisesRegex(
                ManifestError, "1 valid three-leg episodes"):
            self._build(roles=("train",))

    def test_final_reserved_role_is_rejected_before_data_access(self):
        with self.assertRaisesRegex(ManifestError, "final_reserved"):
            self._build(roles=("final_reserved",))
        with self.assertRaisesRegex(ManifestError, "rejected roles"):
            self._build(roles=("train", "final_reserved"))

    def test_input_content_or_split_change_changes_manifest_hash(self):
        original = sha256_bytes(canonical_json_bytes(
            self._build(roles=("train",))))
        goal = self.episodes / "scene_train/episode_0000/goal_1.jpg"
        goal.write_bytes(b"changed-goal-content")
        changed_goal = sha256_bytes(canonical_json_bytes(
            self._build(roles=("train",))))
        self.assertNotEqual(original, changed_goal)

        self.split["version"] = "fixture_split_v2"
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")
        changed_split = sha256_bytes(canonical_json_bytes(
            self._build(roles=("train",))))
        self.assertNotEqual(changed_goal, changed_split)

    def test_causal_prefix_hash_ignores_future_but_tracks_prefix_pose_and_depth(self):
        def hashes():
            samples = self._build(roles=("train",))["samples"]
            factual = {
                sample["state_name"]: sample["causal_prefix"][
                    "causal_prefix_sha256"]
                for sample in samples
                if sample["source_episode"] == "episode_0000"
                and sample["goal_variant"] == "factual"
            }
            return factual

        original = hashes()
        episode = self.episodes / "scene_train/episode_0000"
        # Row 12 is outside t0=[0,8), but inside midpoint t1=[0,24).
        self._write_parquet(episode, changed_row=12, delta=0.75)
        future_changed = hashes()
        self.assertEqual(
            original["goal_b_t0"], future_changed["goal_b_t0"])
        self.assertNotEqual(
            original["goal_b_midpoint_t1"],
            future_changed["goal_b_midpoint_t1"],
        )

        self._write_parquet(episode, changed_row=3, delta=0.5)
        prefix_pose_changed = hashes()
        self.assertNotEqual(
            future_changed["goal_b_t0"], prefix_pose_changed["goal_b_t0"])

        depth = (episode /
                 "videos/chunk-000/observation.images.depth/3.png")
        depth.write_bytes(b"changed-prefix-depth")
        prefix_depth_changed = hashes()
        self.assertNotEqual(
            prefix_pose_changed["goal_b_t0"],
            prefix_depth_changed["goal_b_t0"],
        )

    def test_resume_and_overwrite_are_fail_closed(self):
        manifest = self._build(roles=("train",))
        output = self.root / "artifact/manifest.json"
        sidecar = self.root / "artifact/manifest.json.sha256"
        status, digest = write_artifact(manifest, output, sidecar)
        self.assertEqual(status, "written")
        self.assertIn(digest, sidecar.read_text(encoding="ascii"))
        with self.assertRaisesRegex(ManifestError, "already exists"):
            write_artifact(manifest, output, sidecar)
        status, resumed_digest = write_artifact(
            manifest, output, sidecar, resume=True)
        self.assertEqual((status, resumed_digest), ("resumed", digest))

        changed = dict(manifest)
        changed["purpose"] = "different"
        with self.assertRaisesRegex(ManifestError, "differs"):
            write_artifact(changed, output, sidecar, resume=True)
        status, changed_digest = write_artifact(
            changed, output, sidecar, overwrite=True)
        self.assertEqual(status, "written")
        self.assertNotEqual(changed_digest, digest)


if __name__ == "__main__":
    unittest.main()
