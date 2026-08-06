import csv
import json
from pathlib import Path
import tempfile
import unittest

from MemNavData.validate_strict_graph_smoke import validate


class StrictGraphSmokeValidationTest(unittest.TestCase):
    def make_arm(self, root: Path, name: str, *, spacing: float,
                 active: bool, aux_pose: list[float]) -> Path:
        arm = root / name
        output = arm / "scenes/07_scene/geometry_router"
        output.mkdir(parents=True)
        metric = {
            "episode": "episode_0000",
            "leg1_trace_sha256": "a" * 64,
            "deterministic_plan_seeds": "True",
            "graph_subgoal_spacing_m": str(spacing),
            "reached_A": "1.0", "spl_A": "0.9", "geo_A": "2.0",
            "len_A": "2.1", "final_dist_A": "0.8", "steps_A": "80",
        }
        with (output / "metric.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric))
            writer.writeheader()
            writer.writerow(metric)
        row_a = {
            "requested_diffusion_seed": 10, "diffusion_seed": 10,
            "router_active": False, "aux_pose": None,
        }
        row_b = {
            "requested_diffusion_seed": 20, "diffusion_seed": 20,
            "router_active": active, "aux_pose": aux_pose,
        }
        (output / "episode_0000_plans.json").write_text(
            json.dumps({"legA": [row_a], "legB": [row_b]}),
            encoding="utf-8")
        return arm

    def make_valid(self, root: Path):
        source = self.make_arm(
            root, "source", spacing=0.0, active=False, aux_pose=[0.0, 0.0])
        direct = self.make_arm(
            root, "direct", spacing=0.0, active=True, aux_pose=[3.0, 0.0])
        graph = self.make_arm(
            root, "graph", spacing=1.25, active=True, aux_pose=[1.0, 0.0])
        return source, direct, graph

    def test_accepts_active_graph_with_matched_prefix_and_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self.make_valid(Path(temporary))
            report = validate(*roots)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["paired_active_plans"], 1)
            self.assertEqual(report["max_direct_graph_aux_pose_delta_m"], 2.0)

    def test_rejects_smoke_that_never_exercises_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, direct, graph = self.make_valid(Path(temporary))
            plans = next(graph.glob(
                "scenes/*/geometry_router/episode_*_plans.json"))
            value = json.loads(plans.read_text(encoding="utf-8"))
            value["legB"][0]["router_active"] = False
            plans.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "never activated memory"):
                validate(source, direct, graph)

    def test_rejects_graph_that_does_not_change_point_goal(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, direct, graph = self.make_valid(Path(temporary))
            plans = next(graph.glob(
                "scenes/*/geometry_router/episode_*_plans.json"))
            value = json.loads(plans.read_text(encoding="utf-8"))
            value["legB"][0]["aux_pose"] = [3.0, 0.0]
            plans.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not change"):
                validate(source, direct, graph)


if __name__ == "__main__":
    unittest.main()
