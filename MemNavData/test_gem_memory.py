"""Lifecycle and evidence contracts at the integrated GEM boundary."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from NavDP.baselines.memnav.gem import GeometricEpisodicMemory
from NavDP.baselines.memnav.policy_agent import MemNavAgent
from MemNavData.monocular_depth_runtime import (
    compute_first40_scale_receipt,
    decode_monocular_depth_payload,
)


@pytest.fixture
def agent(tmp_path):
    value = object.__new__(MemNavAgent)
    value.buffer_root = str(tmp_path)
    value._episode_counter = -1
    value.S, value.W = 8, 32
    value.flow_gate = "auto"
    value.device = torch.device("cpu")
    value.certified_reference_depth_source = "online_history"
    value.certified_relocalization_matcher = object()
    value.lb = SimpleNamespace(
        model=SimpleNamespace(
            clean_kv_cache=Mock(),
            camera_head=SimpleNamespace(clean_kv_cache=Mock())),
        load_images=lambda paths: torch.zeros((len(paths), 3, 4, 4)),
    )
    value.reset(seed=0, episode_len=64)
    return value


def test_reset_owns_one_episode_and_releases_all_memory_views(agent):
    gem = agent.memory
    assert isinstance(gem, GeometricEpisodicMemory)
    old_depths = gem.online_depths
    old_depths[8] = (np.ones((2, 2)), np.ones((2, 2)))
    agent.cam_pose.append(torch.ones(9))
    agent._certified_relocalization_cache["goal"] = {"accepted": True}
    assert agent.cam_pose is gem.poses
    assert "cam_pose" not in vars(agent)
    assert agent._certified_route_reference_depth_cache is gem.online_depths
    agent.reset(episode_len=2048)
    assert agent.memory is gem
    assert gem.online_depths == {} and gem.online_depths is not old_depths
    assert gem.poses == [] and gem.descriptors == [] and gem.certificates == {}
    assert gem.frame_count == 0 and gem.current_relative_depth is None
    assert gem.metric_scale_receipt is None
    assert agent.flow_threshold == 50.0
    assert agent.lb.model.clean_kv_cache.call_count == 2
    assert agent.lb.model.camera_head.clean_kv_cache.call_count == 2


def test_write_and_dense_read_bind_same_rgb_without_appending_on_read(agent):
    from io import BytesIO
    image = BytesIO()
    Image.new("RGB", (6, 4), color="red").save(image, format="JPEG")
    jpeg = image.getvalue()
    assert agent.memory.write(jpeg) == 0
    assert agent.n == 1 and len(agent.memory.pending_images) == 1
    payload = agent.memory.read_dense()
    depth, receipt = decode_monocular_depth_payload(
        payload, expected_image_sha256=hashlib.sha256(jpeg).hexdigest())
    assert receipt["frame_index"] == 0
    np.testing.assert_array_equal(depth, np.zeros((4, 6)))
    assert agent.n == 1
    assert (Path(agent.rgb_dir) / "0.jpg").read_bytes() == jpeg


def test_online_depth_missing_is_explicit_and_reader_cannot_mutate_history(agent):
    agent._certified_reference_depth_impl = Mock(side_effect=AssertionError("replay"))
    stored = (np.ones((2, 3), np.float32), np.ones((2, 3), np.float32))
    agent.memory.online_depths[8] = stored
    first = agent._certified_route_reference_depth(8)
    first[0][:] = 7
    np.testing.assert_array_equal(agent.memory.read_online_depth(8)[0], stored[0])
    with pytest.raises(RuntimeError, match="unavailable"):
        agent.memory.read_online_depth(9)
    agent._certified_reference_depth_impl.assert_not_called()


def test_goal_switch_preserves_history_and_reopens_causal_boundary(agent):
    agent.n = 80
    agent.memory.online_depths[42] = (np.ones((2, 2)), np.ones((2, 2)))
    history = agent.memory.online_depths
    agent.memory.replay_goal_session(b"a", 80)
    key = hashlib.md5(b"a").hexdigest()
    agent.memory.certificates[key] = {"accepted": False}
    agent.n = 100
    agent.memory.replay_goal_session(b"b", 100)
    agent.n = 120
    agent.memory.replay_goal_session(b"a", 120)
    assert agent.memory.goal_start_frames[key] == 120
    assert key not in agent.memory.certificates
    assert agent.memory.online_depths is history and 42 in history
    assert agent.memory.goal_session_index == 3


@pytest.mark.parametrize("frame,ceiling", [(12, 10), (80, 20)])
def test_shortlist_is_causal_frozen_and_caller_cannot_change_it(agent, frame, ceiling):
    scores = torch.arange(frame + 1, dtype=torch.float32)
    selected = agent.memory.shortlist_from_scores("goal", ceiling, frame, scores)
    assert selected and all(8 <= item["anchor"] <= ceiling for item in selected)
    assert len(selected) <= 8
    original = [dict(item) for item in selected]
    selected[0]["anchor"] = frame
    scores[:] = -scores
    assert agent.memory.shortlist_from_scores("goal", ceiling, frame, scores) == original


def test_dense_reuses_current_prediction_and_frozen_first40_scale(agent):
    for frame in range(41):
        Image.new("RGB", (4, 4), color=(frame, 0, 0)).save(
            Path(agent.rgb_dir) / f"{frame}.jpg")
    agent.lb.compute_metric_scale = Mock(return_value=(2.0, {
        "h_est": 0.25, "n_points": 100, "n_frames": 40,
        "n_valid": 30, "h_iqr": 0.025}))
    agent.memory.metric_scale_receipt = compute_first40_scale_receipt(
        agent.lb, agent.rgb_dir, np.zeros((40, 9)), 0.5)
    agent.n = 41
    agent._last_agg, agent._psi = [torch.ones(1)], 6
    jpeg = (Path(agent.rgb_dir) / "40.jpg").read_bytes()
    agent._last_frame_jpg_sha256 = hashlib.sha256(jpeg).hexdigest()
    agent._last_stream_depth_frame_index = 40
    agent._last_stream_relative_depth = torch.full((4, 4), 0.75)
    agent.lb.model._predict_depth = Mock(side_effect=AssertionError("duplicate depth"))
    before = dict(agent.memory.metric_scale_receipt)
    first = agent.monocular_depth_observation()
    second = agent.memory.read_dense()
    digest = hashlib.sha256(jpeg).hexdigest()
    d1, _ = decode_monocular_depth_payload(first, expected_image_sha256=digest)
    d2, _ = decode_monocular_depth_payload(second, expected_image_sha256=digest)
    np.testing.assert_array_equal(d1, d2)
    assert np.count_nonzero(d1) == 16
    assert first["depth_prediction_cache_hit"] and second["depth_prediction_cache_hit"]
    assert agent.memory.metric_scale_receipt == before and agent.n == 41
    agent.lb.model._predict_depth.assert_not_called()


def test_episode_checkpoint_roundtrip_preserves_new_backend_and_rgb_directory(agent):
    from io import BytesIO
    resident = {"lb", "device", "certified_relocalization_matcher",
                "buffer_root", "rgb_dir", "_episode_counter"}
    agent.n = 10
    agent.cam_pose = [torch.arange(9, dtype=torch.float32) for _ in range(10)]
    agent.dino_cls = [torch.arange(8, dtype=torch.float32) for _ in range(10)]
    agent.memory.online_depths[8] = (np.ones((2, 3)), np.full((2, 3), 2.0))
    exported = agent.export_episode_state(resident_fields=resident)
    assert "memory" not in exported and "rgb_dir" not in exported
    assert exported["n"] == 10 and "cam_pose" in exported
    stream = BytesIO()
    torch.save(exported, stream)  # A live weakref/model owner is not serializable.
    agent.reset(episode_len=64)
    gem, rgb_dir, model = agent.memory, agent.rgb_dir, agent.lb
    stream.seek(0)
    agent.restore_episode_state(
        torch.load(stream, weights_only=False), resident_fields=resident)
    assert agent.memory is gem and gem.backend is agent and agent.lb is model
    assert agent.rgb_dir == rgb_dir and agent.n == 10
    assert len(gem.poses) == len(gem.descriptors) == 10
    np.testing.assert_array_equal(gem.read_online_depth(8)[1], np.full((2, 3), 2.0))
    assert "n" not in vars(agent) and "cam_pose" not in vars(agent)


def test_episode_restore_rejects_missing_or_resident_state_before_mutation(agent):
    resident = {"lb", "device", "certified_relocalization_matcher",
                "buffer_root", "rgb_dir", "_episode_counter"}
    state = agent.export_episode_state(resident_fields=resident)
    missing = dict(state)
    missing.pop("n")
    with pytest.raises(ValueError, match="missing GEM"):
        agent.restore_episode_state(missing, resident_fields=resident)
    with pytest.raises(ValueError, match="resident"):
        agent.restore_episode_state(dict(state, memory=object()), resident_fields=resident)
    assert agent.n == 0
