import numpy as np
import pytest

from MemNavData.vint_rgb_input_bridge import install, rgb_from_server_bgr


def test_exact_rgb_recovered_without_changing_the_source():
    rgb = np.array([[[[11, 50, 239], [210, 30, 7]]]], dtype=np.uint8)
    bgr = rgb[..., ::-1].copy()
    before = bgr.copy()
    np.testing.assert_array_equal(rgb_from_server_bgr(bgr), rgb)
    np.testing.assert_array_equal(bgr, before)


@pytest.mark.parametrize("shape", [(3, 3), (4, 5, 3), (1, 4, 5, 1)])
def test_shape_errors_are_visible(shape):
    with pytest.raises(ValueError):
        rgb_from_server_bgr(np.zeros(shape))


def test_context_and_goal_use_same_rgb_contract_once():
    class Agent:
        def __init__(self):
            self.observations = []

        def callback_obs(self, image):
            self.observations.append(image)

        def observe(self, image):
            self.callback_obs(image)

        def step_imagegoal(self, goal, image):
            self.callback_obs(image)
            return goal, image

        def step_nogoal(self, image):
            self.callback_obs(image)
            return image

    records = []
    install(Agent, records.append)
    a = Agent()
    rgb = np.array([[[[12, 55, 199]]]], dtype=np.uint8)
    goal = np.array([[[[94, 102, 6]]]], dtype=np.uint8)
    a.observe(rgb[..., ::-1])
    got_goal, got_current = a.step_imagegoal(goal[..., ::-1], rgb[..., ::-1])
    a.step_nogoal(rgb[..., ::-1])
    np.testing.assert_array_equal(got_goal, goal)
    np.testing.assert_array_equal(got_current, rgb)
    assert len(a.observations) == 3
    for obs in a.observations:
        np.testing.assert_array_equal(obs, rgb)
    assert [r["kind"] for r in records] == [
        "replay_observation", "goal", "observation", "nogoal_observation"]
