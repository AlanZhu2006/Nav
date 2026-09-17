"""CPU contract tests; the physical primitive test is separate."""
import ast
import math
from pathlib import Path
import types
import tempfile
import unittest

import numpy as np
import torch

from MemNavData.habitat_bullet_alignment import PhysicalAlignment, install_loop_hook, wrap
from MemNavData.navdp_interface_diagnostic_server import instrument_mask
from MemNavData.lingbot_pose_diagnostic_server import pose_receipt

ROOT = Path(__file__).resolve().parents[1]


def source_method(filename, name):
    tree = ast.parse(filename.read_text())
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(torch=torch, np=np)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(filename), "exec"), scope)
    return scope[name]


def accepted(point):
    return {"certified_relocalization_accepted": True, "router_active": True,
            "memory_controller_pointgoal": point,
            "certified_relocalization_guidance_mode": "endpoint_bearing"}


class Tests(unittest.TestCase):
    def test_pose_observer_accepts_normal_uninitialized_scale_prefix(self):
        agent = types.SimpleNamespace(cam_pose=[], executor_motion_receipts=[None],
                                      monocular_depth_status=lambda: {"scale_ready": False})
        row = pose_receipt(agent, b"rgb", 0)
        self.assertIsNone(row["camera_pose9"])
        agent.cam_pose = [torch.zeros(9)]
        self.assertEqual(pose_receipt(agent, b"rgb", 7)["camera_pose9"], [0]*9)

    def test_process_local_loop_patch_compiles_without_editing_evaluator(self):
        filename = ROOT / "MemNavData/eval_2leg_habitat.py"
        before = filename.read_bytes()
        original = source_method(filename, "run_policy_leg")
        with tempfile.TemporaryDirectory(prefix="bullet_alignment_hook_") as temporary:
            modified = install_loop_hook(types.SimpleNamespace(), original,
                                         PhysicalAlignment(True), Path(temporary) / "leg.py")
            self.assertEqual(modified.__name__, original.__name__)
        self.assertEqual(before, filename.read_bytes())

    def test_rearward_clipping_is_the_actual_upstream_code(self):
        process = source_method(ROOT / "NavDP/baselines/navdp/policy_agent.py", "process_pointgoal")
        points = np.array([[-2.4207269510200806, -.6245646712751393, 0],
                           [-2.4916430034291723, .20424285412824902, 0]])
        result = process(None, points)
        self.assertTrue(np.array_equal(result[:, 0], np.zeros(2)))
        self.assertTrue(np.array_equal(result[:, 1:], points[:, 1:]))
        self.assertTrue(np.all(points[:, 0] < 0))

    def test_real_source_instrumentation_preserves_outputs_and_rng(self):
        original = source_method(ROOT / "NavDP/baselines/navdp/policy_network.py", "predict_ip_action")
        saved = {}
        instrumented = instrument_mask(original, lambda tag, trajectories, values:
                                       saved.update({tag: trajectories.detach().clone()}))

        class Scheduler:
            config = types.SimpleNamespace(num_train_timesteps=1)
            timesteps = torch.tensor([0])

            def set_timesteps(self, _):
                pass

            def step(self, *, model_output, timestep, sample):
                scale = torch.ones((sample.shape[0], 1, 1))
                scale[:8] = .01
                return types.SimpleNamespace(prev_sample=sample * scale)

        class Dummy:
            device = "cpu"
            predict_size = 24
            noise_scheduler = Scheduler()

            def rgbd_encoder(self, images, depths):
                return torch.zeros((images.shape[0], 2, 4))

            def image_encoder(self, image):
                return torch.zeros((image.shape[0], 4))

            def point_encoder(self, point):
                return torch.zeros((point.shape[0], 4))

            def predict_mix_noise(self, action, time, goals, rgbd):
                return torch.zeros_like(action)

            def predict_critic(self, action, rgbd):
                return action.square().mean(dim=(1, 2))

        arguments = (np.zeros((1, 3)), np.zeros((1, 2, 2, 3)),
                     np.zeros((1, 8, 2, 2, 3)), np.zeros((1, 2, 2, 1)))
        torch.manual_seed(710)
        expected = original(Dummy(), *arguments)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(710)
        observed = instrumented(Dummy(), *arguments)
        for before, after in zip(expected, observed):
            self.assertTrue(np.array_equal(before, after))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        mask = saved["pre_mask"][:, :, -1, :2].norm(dim=-1) < .5
        self.assertGreaterEqual(int(mask.sum()), 8)
        self.assertGreater(int((~mask).sum()), 0)
        self.assertTrue(torch.equal(saved["post_mask"][~mask], saved["pre_mask"][~mask]))
        self.assertEqual(int(torch.count_nonzero(saved["post_mask"][mask][:, :, :2])), 0)

    def test_no_authorization_no_turn(self):
        controller = PhysicalAlignment(True)
        controller.consider({"memory_controller_pointgoal": [-2.5, 0]}, 0, 0)
        self.assertFalse(controller.active)

    def test_first_forward_is_a_noop_for_regression_controls(self):
        controller = PhysicalAlignment(True)
        controller.consider(accepted([2.5, 0]), 0, 0)
        controller.consider(accepted([-2.5, 0]), 0, 8)
        self.assertTrue(controller.considered)
        self.assertFalse(controller.active)

    def test_two_turn_signs_hold_target_without_translation(self):
        for angle in (-165, 175):
            radians = math.radians(angle)
            controller = PhysicalAlignment(True)
            controller.consider(accepted([2.5*math.cos(radians), 2.5*math.sin(radians)]), 2.9, 0)
            yaw = 2.9
            while controller.active:
                speed, rate = controller.command(yaw)
                self.assertEqual(speed, 0)
                self.assertLessEqual(abs(rate), math.pi/4)
                yaw = wrap(yaw + .1*rate)
                controller.observe_after_action(yaw, controller.action_count)
            self.assertLessEqual(abs(wrap(yaw - 2.9 - radians)), controller.tolerance)
            self.assertLessEqual(controller.action_count, 40)


if __name__ == "__main__":
    unittest.main()
