import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.build_revisit_fresh_manifest import (
    build_manifest,
    file_record,
)
from MemNavData.summarize_revisit_fresh_confirmation import (
    cluster_interval,
    conditional_paired,
    decision_branch,
    validate_summary,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class RevisitFreshConfirmationTest(unittest.TestCase):
    def make_fixture(self, root: Path):
        scenes = [f"scene_{index:02d}" for index in range(20)]
        protocol = {
            "schema_version": 1,
            "protocol_id": "test",
            "scope": "fresh episodes only",
            "scenes": scenes,
            "episodes_per_scene": 8,
            "generation": {
                "base_seed": 1000,
                "covis_lo": 0.2,
                "covis_hi": 1.0,
                "head_max_deg": 45.0,
                "expected_gen_protocol": "multileg_v2_symmetric_20260807",
            },
            "evaluation": {},
            "analysis": {},
            "data_role_guards": {"blind_allowed": False},
        }
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol))
        generator = root / "generator.py"
        generator.write_text("# frozen\n")
        generated = root / "generated"
        assets_root = root / "assets"
        historical = {
            "selection": {"selected_scenes": scenes},
            "training_scenes": [],
            "assets": {},
            "episodes": {},
            "dependencies": {},
        }
        for scene_index, scene in enumerate(scenes):
            asset = assets_root / scene / f"{scene}.glb"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(f"asset-{scene}".encode())
            historical["assets"][scene] = file_record(asset)
            historical["episodes"][scene] = [
                {
                    "episode": f"episode_{old:04d}",
                    "files": {
                        kind: {"sha256": digest(f"old-{scene}-{old}-{kind}".encode())}
                        for kind in ("metadata", "parquet", "goal")
                    },
                }
                for old in range(2)
            ]
            for episode_index in range(8):
                episode = (generated / "mp3d_2leg" / scene /
                           f"episode_{episode_index:04d}")
                rgb = episode / "videos/chunk-000/observation.images.rgb"
                depth = episode / "videos/chunk-000/observation.images.depth"
                data = episode / "data/chunk-000"
                meta = episode / "meta"
                for directory in (rgb, depth, data, meta):
                    directory.mkdir(parents=True)
                (rgb / "0.jpg").write_bytes(b"rgb")
                (depth / "0.png").write_bytes(b"depth")
                (data / "episode_000000.parquet").write_bytes(
                    f"parquet-{scene}-{episode_index}".encode())
                goal = f"goal-{scene}-{episode_index}".encode()
                (episode / "goal_image.jpg").write_bytes(goal)
                (episode / "goal_1.jpg").write_bytes(goal)
                metadata = {
                    "scene": f"{scene}.glb",
                    "ep_idx": episode_index,
                    "generation_seed": 1000 + scene_index,
                    "n_frames": 1,
                    "n_legs": 2,
                    "gen_protocol": "multileg_v2_symmetric_20260807",
                    "goals": [{
                        "name": "B", "kind": "revisit", "covis": 0.5,
                        "head_off_deg": 10.0, "recall_gap": 32,
                    }],
                }
                (meta / "gen_meta.json").write_text(json.dumps(metadata))
        historical_path = root / "historical.json"
        historical_path.write_text(json.dumps(historical))
        return protocol_path, historical_path, generated, assets_root, generator

    def test_manifest_accepts_exact_fresh_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory))
            manifest = build_manifest(*paths)
            self.assertEqual(manifest["audit"]["episodes"], 160)
            self.assertFalse(manifest["audit"]["historical_episode_hash_overlap"])
            self.assertEqual(manifest["episodes"]["scene_00"][0]["recall_gap"], 32)
            self.assertEqual(manifest["generation"], json.loads(paths[0].read_text())["generation"])

    def test_manifest_rejects_historical_goal_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, historical_path, generated, assets, generator = self.make_fixture(root)
            historical = json.loads(historical_path.read_text())
            goal = generated / "mp3d_2leg/scene_00/episode_0000/goal_image.jpg"
            historical["episodes"]["scene_00"][0]["files"]["goal"]["sha256"] = (
                digest(goal.read_bytes()))
            historical_path.write_text(json.dumps(historical))
            with self.assertRaisesRegex(ValueError, "duplicates historical goal"):
                build_manifest(protocol, historical_path, generated, assets, generator)

    def test_clustered_pairing_and_frozen_decision(self):
        scenes = [f"scene_{index:02d}" for index in range(20)]
        episodes = {scene: [f"episode_{i:04d}" for i in range(8)] for scene in scenes}
        left = {}
        right = {}
        expected = set()
        for scene in scenes:
            for index, episode in enumerate(episodes[scene]):
                key = (scene, episode)
                expected.add(key)
                left[key] = {"reached_a": True, "reached_b": False, "joint": False}
                right[key] = {
                    "reached_a": True,
                    "reached_b": index == 0,
                    "joint": index == 0,
                }
        paired = conditional_paired("left", "right", left, right, expected)
        self.assertEqual(paired["outcomes"]["right_only_revisit_success"], 20)
        interval = cluster_interval(
            scenes, episodes, left, right, conditional=False,
            seed=7, resamples=1000)
        self.assertAlmostEqual(interval[0], 0.125)
        self.assertAlmostEqual(interval[1], 0.125)
        self.assertEqual(
            decision_branch(0.125, 0.001, interval, 0.7, 0.6),
            "replace_geometry_hard_gate_then_seek_fresh_scene_confirmation",
        )
        self.assertEqual(
            decision_branch(0.02, 0.4, [-0.02, 0.08], 0.7, 0.6),
            "inconclusive_keep_geometry_and_do_not_retune_on_these_episodes",
        )

    def test_native_summary_matches_recorded_inert_adapter_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({
                "episodes": 8,
                "max_steps": 500,
                "exec_horizon": 8,
                "leg1_mode": "shared_trace",
                "write_leg1_trace": False,
                "deterministic_plan_seeds": True,
                "trajectory_selector": "server",
                "trajectory_selector_scope": "all",
                "graph_subgoal_spacing_m": None,
                "graph_subgoal_arrival_m": None,
                "server_backend": "navdp",
                "hybrid_route": "phase",
                "revisit_adapter": "legacy_metric",
            }))
            validate_summary(path, "native", 8)


if __name__ == "__main__":
    unittest.main()
