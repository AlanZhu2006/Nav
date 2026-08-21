#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from freeze_final14_source_manifest import freeze, sha256_file
from validate_paper_online_a_scene import validate


def write_image(path: Path, color: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (color, color, color)).save(path, format="JPEG")


class FreezeFinal14SourceManifestTest(unittest.TestCase):
    def materialize_episode(
        self, root: Path, scene: str, number: int, *, valid: bool = True
    ) -> None:
        episode = root / scene / f"episode_{number:04d}"
        (episode / "meta").mkdir(parents=True, exist_ok=True)
        (episode / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        metadata = {
            "scene": f"{scene}.glb",
            "n_legs": 2,
            "n_frames": 2,
            "goals": [{"recall_gap": 32}],
        }
        (episode / "meta/gen_meta.json").write_text(json.dumps(metadata))
        (episode / "data/chunk-000/episode_000000.parquet").write_bytes(b"pq")
        write_image(episode / "goal_1.jpg", 60)
        rgb = episode / "videos/chunk-000/observation.images.rgb"
        write_image(rgb / "0.jpg", 10)
        if valid:
            write_image(rgb / "1.jpg", 20)

    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        assets = root / "assets"
        episodes = root / "episodes"
        checkpoint = root / "navdp.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        final = [f"final_{index:02d}" for index in range(14)]
        for index, scene in enumerate(final):
            asset = assets / scene / f"{scene}.glb"
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(scene.encode())
            # One scene has an invalid lexically earlier candidate and only
            # one available episode; all other scenes have two.
            if index == 0:
                self.materialize_episode(episodes, scene, 0, valid=False)
                self.materialize_episode(episodes, scene, 1)
            else:
                self.materialize_episode(episodes, scene, 0)
                self.materialize_episode(episodes, scene, 1)
        train = [f"train_{index:02d}" for index in range(40)]
        development = [f"dev_{index:02d}" for index in range(20)]
        consumed = [f"blind_{index:02d}" for index in range(16)]
        budget_path = root / "scene_budget.json"
        budget_path.write_text(json.dumps({
            "schema_version": "mp3d_scene_budget_v1_20260816",
            "freeze_precedes_new_control_outcomes": True,
            "partitions": {
                "train40": train,
                "consumed_development20": development,
                "consumed_blind16": consumed,
                "untouched_final14": final,
            },
        }, sort_keys=True))
        base_path = root / "base.json"
        base_path.write_text(json.dumps({
            "schema_version": 2,
            "selection": {},
            "training_scenes": train,
            "paths": {
                "asset_root": str(assets),
                "expanded_episode_root": str(episodes),
            },
            "dependencies": {
                "navdp_checkpoint": {
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
            },
            "assets": {},
            "episodes": {},
            "evaluation": {"episodes_per_scene": 2, "base_seed": 17},
        }, sort_keys=True))
        return base_path, budget_path, checkpoint

    def test_freezes_ledger_order_shortage_and_validates_without_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, budget, checkpoint = self.fixture(root)
            out = root / "final14.json"
            result = freeze(
                base_manifest_path=base,
                scene_budget_path=budget,
                out=out,
                episode_root_override=root / "episodes",
                expected_base_manifest_sha256=sha256_file(base),
                expected_scene_budget_sha256=sha256_file(budget),
            )
            payload = json.loads(out.read_text())
            self.assertEqual(result["scenes"], 14)
            self.assertEqual(result["source_episodes_available"], 27)
            self.assertEqual(result["scenes_below_target"], 14)
            self.assertFalse(result["policy_outcomes_read"])
            self.assertEqual(
                payload["paths"]["expanded_episode_root"],
                str(root / "episodes"),
            )
            self.assertEqual(
                [row["episode"] for row in payload["episodes"]["final_00"]],
                ["episode_0001"],
            )
            audit = validate(
                out, sha256_file(out), 0, checkpoint,
                expected_scene_count=14,
                expected_scene_budget_sha256=sha256_file(budget),
            )
            self.assertEqual(audit["source_episode_count"], 1)
            self.assertEqual(audit["source_episode_target"], 8)
            self.assertEqual(audit["source_episode_shortage"], 7)


if __name__ == "__main__":
    unittest.main()
