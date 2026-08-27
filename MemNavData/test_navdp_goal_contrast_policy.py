from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

NAVDP_DIR = Path(__file__).resolve().parents[1] / "NavDP" / "baselines" / "navdp"
sys.path.insert(0, str(NAVDP_DIR))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402
from policy_network import NavDP_Policy  # noqa: E402


class _FakePolicy:
    score_imagegoal_trajectories = (
        NavDP_Policy.score_imagegoal_trajectories)

    def __init__(self):
        self.device = "cpu"
        self.predict_size = 2
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=4,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

    @staticmethod
    def rgbd_encoder(images, depths):
        return torch.zeros((images.shape[0], 2, 1), dtype=torch.float32)

    @staticmethod
    def image_encoder(images):
        tensor = torch.as_tensor(images, dtype=torch.float32)
        return tensor.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)

    @staticmethod
    def predict_noise(noisy_actions, timestep, goal_embed, rgbd_embed):
        del timestep, rgbd_embed
        goal_value = goal_embed[:, 0, 0].reshape(-1, 1, 1)
        return noisy_actions * 0.25 + goal_value


def test_policy_score_is_paired_deterministic_and_rng_read_only():
    policy = _FakePolicy()
    goal = np.ones((1, 2, 2, 3), dtype=np.float32)
    control = np.zeros_like(goal)
    images = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
    depths = np.zeros((1, 2, 2, 1), dtype=np.float32)
    candidate = np.asarray([
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ], dtype=np.float32)
    trajectories = np.asarray([[candidate, candidate, candidate * 2.0]])

    rng_before = torch.random.get_rng_state().clone()
    first = policy.score_imagegoal_trajectories(
        goal,
        images,
        depths,
        trajectories,
        control_goal_image=control,
        timesteps=(0, 2, 3),
        noise_samples=2,
        seed=17,
    )
    rng_after = torch.random.get_rng_state().clone()
    second = policy.score_imagegoal_trajectories(
        goal,
        images,
        depths,
        trajectories,
        control_goal_image=control,
        timesteps=(0, 2, 3),
        noise_samples=2,
        seed=17,
    )

    assert torch.equal(rng_before, rng_after)
    assert first == second
    assert first["is_calibrated_likelihood"] is False
    assert first["shared_noise_across_candidates"] is True
    assert first["timesteps"] == [0, 2, 3]
    assert np.asarray(first["goal_advantage"]).shape == (1, 3)
    # Duplicate trajectories see exactly the same noise, so no Monte-Carlo
    # accident can create a preference between them.
    assert first["goal_advantage"][0][0] == pytest.approx(
        first["goal_advantage"][0][1], abs=0.0)
    assert first["control_goal_advantage"][0][0] == pytest.approx(
        first["control_goal_advantage"][0][1], abs=0.0)


def test_policy_score_rejects_invalid_timestep_contract():
    policy = _FakePolicy()
    goal = np.zeros((1, 2, 2, 3), dtype=np.float32)
    images = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
    depths = np.zeros((1, 2, 2, 1), dtype=np.float32)
    trajectories = np.zeros((1, 1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="unique"):
        policy.score_imagegoal_trajectories(
            goal, images, depths, trajectories, timesteps=(1, 1))
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        policy.score_imagegoal_trajectories(
            goal, images, depths, trajectories, timesteps=(4,))
