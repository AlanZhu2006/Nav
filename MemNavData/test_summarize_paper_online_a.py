#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from summarize_paper_online_a import summarize


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SummarizePaperOnlineATest(unittest.TestCase):
    def make_fixture(self, episodes_per_scene: int = 4):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "run"
        scene = "scene_a"
        episode_ids = [
            f"episode_{index:04d}" for index in range(episodes_per_scene)
        ]
        manifest_path = Path(temporary.name) / "manifest.json"
        manifest = {
            "selection": {"selected_scenes": [scene]},
            "evaluation": {"episodes_per_scene": episodes_per_scene},
            "episodes": {
                scene: [{"episode": episode} for episode in episode_ids]
            },
        }
        manifest_sha = write_json(manifest_path, manifest)
        scene_root = root / "traces" / f"00_{scene}"
        traces = []
        materialized = []
        attrition = []
        for index, episode in enumerate(episode_ids):
            payload = {
                "episode": episode,
                "reached": index % 2 == 0,
                "steps": 10 + index,
            }
            trace_path = scene_root / "native_a" / f"{episode}_leg1_trace.json"
            trace_sha = write_json(trace_path, payload)
            traces.append({**payload, "sha256": trace_sha})
            if payload["reached"]:
                materialized.append({"scene": scene, "episode": episode})
            else:
                attrition.append({"scene": scene, "episode": episode})
        write_json(
            scene_root / "receipt.json",
            {
                "schema_version": "paper_online_a_scene_receipt_v1_20260814",
                "manifest_sha256": manifest_sha,
                "query_outcomes_read": False,
                "scene": scene,
                "traces": traces,
            },
        )
        write_json(
            scene_root / "online_a" / "manifest.json",
            {
                "schema_version": "shared_online_a_materialized_v1",
                "source_trace_count": episodes_per_scene,
                "selection": {"all_eligible_traces_attempted": True},
                "episodes": materialized,
                "attrition": attrition,
            },
        )
        return temporary, root, manifest_path, manifest_sha

    def test_manifest_drives_four_episode_population(self):
        temporary, root, manifest_path, manifest_sha = self.make_fixture(4)
        self.addCleanup(temporary.cleanup)

        result = summarize(root, manifest_path, manifest_sha)

        self.assertEqual(result["source_scenes"], 1)
        self.assertEqual(result["episodes_per_scene"], 4)
        self.assertEqual(result["source_episodes"], 4)
        self.assertEqual(result["goal_a_successes"], 2)
        self.assertEqual(result["goal_a_failures"], 2)
        self.assertEqual(result["materialized_histories"], 2)
        self.assertFalse(result["query_outcomes_read"])

    def test_rejects_trace_population_not_in_manifest(self):
        temporary, root, manifest_path, manifest_sha = self.make_fixture(4)
        self.addCleanup(temporary.cleanup)
        receipt_path = root / "traces/00_scene_a/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["traces"] = receipt["traces"][:-1]
        write_json(receipt_path, receipt)

        with self.assertRaisesRegex(RuntimeError, "trace population"):
            summarize(root, manifest_path, manifest_sha)

    def test_rejects_manifest_hash_mismatch(self):
        temporary, root, manifest_path, manifest_sha = self.make_fixture(4)
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(RuntimeError, "manifest hash"):
            summarize(root, manifest_path, "0" * len(manifest_sha))

    def test_final14_variable_source_counts_preserve_shortage(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "run"
        scenes = ["scene_a", "scene_b"]
        counts = {"scene_a": 2, "scene_b": 1}
        episodes = {
            scene: [
                {"episode": f"episode_{index:04d}"}
                for index in range(counts[scene])
            ]
            for scene in scenes
        }
        manifest_path = Path(temporary.name) / "manifest.json"
        manifest_sha = write_json(manifest_path, {
            "schema_version": "final14_paper_source_manifest_v1_20260817",
            "selection": {"selected_scenes": scenes},
            "evaluation": {
                "episode_target_per_scene": 8,
                "episode_counts_by_scene": counts,
            },
            "episodes": episodes,
            "source_attrition": [{
                "scene": "scene_b", "stage": "source_episode_shortage"
            }],
        })
        for scene_index, scene in enumerate(scenes):
            scene_root = root / "traces" / f"{scene_index:02d}_{scene}"
            traces = []
            for row in episodes[scene]:
                episode = row["episode"]
                payload = {"episode": episode, "reached": True, "steps": 5}
                path = scene_root / "native_a" / f"{episode}_leg1_trace.json"
                traces.append({**payload, "sha256": write_json(path, payload)})
            write_json(scene_root / "receipt.json", {
                "schema_version": "paper_online_a_scene_receipt_v1_20260814",
                "manifest_sha256": manifest_sha,
                "query_outcomes_read": False,
                "scene": scene,
                "traces": traces,
            })
            write_json(scene_root / "online_a/manifest.json", {
                "schema_version": "shared_online_a_materialized_v1",
                "source_trace_count": counts[scene],
                "selection": {"all_eligible_traces_attempted": True},
                "episodes": [], "attrition": [],
            })

        result = summarize(root, manifest_path, manifest_sha)
        self.assertIsNone(result["episodes_per_scene"])
        self.assertEqual(result["episode_target_per_scene"], 8)
        self.assertEqual(result["source_episodes"], 3)
        self.assertEqual(result["source_episode_target"], 16)
        self.assertEqual(result["source_asset_attrition_count"], 1)


if __name__ == "__main__":
    unittest.main()
