import tempfile
import unittest
from pathlib import Path

from MemNavData.shared_online_double_revisit_runtime import (
    replay_online_a,
    should_activate_certified_graph_rescue,
    should_activate_certified_stagnation_intervention,
    summarize_c_tail,
)


class ReplayOnlineATest(unittest.TestCase):
    def test_replays_all_long_memory_and_only_decision_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            poses = []
            for step, data in enumerate((b"a", b"b", b"c", b"d")):
                image = source / "rgb" / f"{step:06d}.jpg"
                image.parent.mkdir(exist_ok=True)
                image.write_bytes(data)
                import hashlib

                poses.append(
                    {
                        "step": step,
                        "x": float(step),
                        "y": 0.0,
                        "z": 0.0,
                        "yaw": 0.0,
                        "jpg_sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
            frozen = {
                "source": source,
                "trace": {
                    "poses": poses,
                    "plans": [{"step": 0}, {"step": 2}],
                },
            }
            memory_calls = []
            navdp_calls = []

            def memory_step(image):
                memory_calls.append(image)
                return {"frame_idx": len(memory_calls) - 1}

            def navdp_step(image):
                navdp_calls.append(image)
                return {
                    "diffusion_sampled": False,
                    "queue_lengths": [len(navdp_calls)],
                    "memory_size": 8,
                }

            result = replay_online_a(
                frozen,
                memory_step=memory_step,
                navdp_replay_step=navdp_step,
            )
            self.assertEqual(memory_calls, [b"a", b"b", b"c", b"d"])
            self.assertEqual(navdp_calls, [b"a", b"c"])
            self.assertEqual(result["online_frames"], 4)
            self.assertEqual(result["decision_steps"], [0, 2])
            self.assertEqual(result["diffusion_samples_during_replay"], 0)

    def test_fails_on_server_index_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            image = source / "rgb" / "000000.jpg"
            image.parent.mkdir()
            image.write_bytes(b"frame")
            import hashlib

            frozen = {
                "source": source,
                "trace": {
                    "poses": [{
                        "step": 0,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "yaw": 0.0,
                        "jpg_sha256": hashlib.sha256(b"frame").hexdigest(),
                    }],
                    "plans": [{"step": 0}],
                },
            }
            with self.assertRaisesRegex(RuntimeError, "index differs"):
                replay_online_a(
                    frozen,
                    memory_step=lambda _image: {"frame_idx": 7},
                    navdp_replay_step=lambda _image: {
                        "diffusion_sampled": False,
                        "queue_lengths": [1],
                        "memory_size": 8,
                    },
                )


class CTailSummaryTest(unittest.TestCase):
    def test_hard_negative_passes_at_boundary(self):
        result = summarize_c_tail([0.02, 0.10, 0.03], maximum_allowed=0.10)
        self.assertTrue(result["ok"])
        self.assertEqual(result["argmax_b_frame"], 1)
        self.assertEqual(result["endpoint_covisibility"], 0.03)

    def test_contamination_fails(self):
        result = summarize_c_tail([0.02, 0.11], maximum_allowed=0.10)
        self.assertFalse(result["ok"])


class CertifiedGraphRescueDecisionTest(unittest.TestCase):
    def decide(self, **overrides):
        values = {
            "mode": "rescue",
            "server_backend": "hybrid_pose",
            "hybrid_route": "certified_relocalization",
            "policy_backend": "navdp_auto",
            "attempted": False,
            "plans": [{"certified_relocalization_accepted": True}],
        }
        values.update(overrides)
        return should_activate_certified_graph_rescue(**values)

    def test_only_accepted_certified_route_can_activate_once(self):
        self.assertTrue(self.decide())
        self.assertTrue(should_activate_certified_stagnation_intervention(
            mode="budget_control",
            server_backend="hybrid_pose",
            hybrid_route="certified_relocalization",
            policy_backend="navdp_auto",
            attempted=False,
            plans=[{"certified_relocalization_accepted": True}],
        ))
        self.assertFalse(self.decide(mode="off"))
        self.assertFalse(self.decide(attempted=True))
        self.assertFalse(self.decide(plans=[]))
        self.assertFalse(self.decide(
            plans=[{"certified_relocalization_accepted": False}]))
        self.assertFalse(self.decide(hybrid_route="phase"))
        self.assertFalse(self.decide(policy_backend="navdp"))

    def test_unknown_mode_fails_closed(self):
        with self.assertRaisesRegex(
                ValueError, "unknown certified stagnation mode"):
            self.decide(mode="mystery")


if __name__ == "__main__":
    unittest.main()
