import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.build_graph_blind_manifest import build_manifest
from MemNavData.validate_expanded_navdp_router_eval import validate_selection


class BlindManifestBuilderTest(unittest.TestCase):
    @staticmethod
    def source():
        return {
            "schema_version": 1,
            "selection": {
                "selected_scenes": ["used"],
                "eligible_unseen_scenes": ["used", "blind_a", "blind_b"],
            },
            "training_scenes": ["train"],
            "paths": {"asset_root": "/old", "expanded_episode_root": "/old"},
            "dependencies": {},
            "evaluation": {"episodes_per_scene": 2},
            "assets": {},
            "episodes": {},
        }

    @staticmethod
    def materialize(root: Path, scene: str):
        asset_root = root / "assets"
        episode_root = root / "episodes"
        asset = asset_root / scene / f"{scene}.glb"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(scene.encode())
        for number in range(2):
            episode = episode_root / scene / f"episode_{number:04d}"
            (episode / "meta").mkdir(parents=True)
            (episode / "data/chunk-000").mkdir(parents=True)
            rgb = episode / "videos/chunk-000/observation.images.rgb"
            rgb.mkdir(parents=True)
            metadata = {
                "scene": f"{scene}.glb",
                "n_legs": 2,
                "n_frames": 2,
                "goals": [{"recall_gap": 32}],
            }
            (episode / "meta/gen_meta.json").write_text(json.dumps(metadata))
            (episode / "data/chunk-000/episode_000000.parquet").write_bytes(b"pq")
            (episode / "goal_1.jpg").write_bytes(b"goal")
            (rgb / "0.jpg").write_bytes(b"0")
            (rgb / "1.jpg").write_bytes(b"1")
        return asset_root, episode_root

    def test_builder_uses_only_untouched_scenes_and_validates_hash_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in ("blind_a", "blind_b"):
                asset_root, episode_root = self.materialize(root, scene)
            source = self.source()
            manifest = build_manifest(
                source,
                source_sha256=hashlib.sha256(b"source").hexdigest(),
                asset_root=asset_root,
                episode_root=episode_root,
            )
            selected = validate_selection(manifest)
            self.assertEqual(set(selected), {"blind_a", "blind_b"})
            self.assertNotIn("used", manifest["episodes"])
            self.assertEqual(sum(map(len, manifest["episodes"].values())), 4)


if __name__ == "__main__":
    unittest.main()
