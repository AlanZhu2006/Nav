"""Evidence preservation and failure boundaries for the complete memory API."""
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from NavDP.baselines.memnav.gem.archive import FrameDepthArchive
from NavDP.baselines.memnav.gem.connected import ConnectedEpisodeMemory
from NavDP.baselines.memnav.gem.episodic import EpisodicGEM
from NavDP.baselines.memnav.policy_agent import MemNavAgent


def test_archive_lossless_first_write_and_coordinate_scale(tmp_path):
    archive = FrameDepthArchive(tmp_path / 'depth')
    depth = np.random.default_rng(1).uniform(.1, 9, (11, 13)).astype(np.float32)
    confidence = depth / 3
    archive.append(0, depth, confidence, 2.5)
    got, conf = archive[0]
    np.testing.assert_array_equal(got, depth * np.float32(2.5))
    np.testing.assert_array_equal(conf, confidence)
    got[:] = 0
    np.testing.assert_array_equal(archive[0][0], depth * np.float32(2.5))
    assert archive.array_bytes == depth.nbytes * 2
    with pytest.raises(ValueError):
        archive.append(0, depth, confidence, 1.)
    with pytest.raises(KeyError):
        archive[1]
    path = archive.directory / '000000.npz'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        archive[0]


def test_stream_partial_failure_cannot_be_resumed(monkeypatch):
    model = SimpleNamespace(aggregator=SimpleNamespace(use_sdpa=True, total_frames_processed=0),
        clean_kv_cache=Mock())
    memory = ConnectedEpisodeMemory(model, np.array([[8, 8]]), np.array([True]))
    monkeypatch.setattr(memory, '_forward', Mock(side_effect=RuntimeError('kernel failed')))
    image = torch.zeros(3, 518, 518)
    for i in range(7):
        assert memory.write(i, image) is None
    with pytest.raises(RuntimeError, match='kernel failed'):
        memory.write(7, image)
    with pytest.raises(RuntimeError, match='reset'):
        memory.write(8, image)


def test_rgb_archive_failure_requires_reset_and_clears_stale_depth(tmp_path, monkeypatch):
    agent = object.__new__(MemNavAgent)
    agent.memory_mechanism = 'connected_reciprocal'
    agent.buffer_root, agent._episode_counter = str(tmp_path), -1
    agent.S, agent.W, agent.flow_gate = 8, 64, 'off'
    agent.device = torch.device('cpu')
    agent.certified_reference_depth_source = 'online_history'
    agent.certified_eager_depth_cache = agent.depth_observation_only = False
    model = SimpleNamespace(clean_kv_cache=Mock(), camera_head=SimpleNamespace(
        clean_kv_cache=Mock(), float=Mock()), aggregator=SimpleNamespace(
        to=Mock(), use_sdpa=True, total_frames_processed=0))
    agent.lb = SimpleNamespace(model=model, agg=model.aggregator,
        load_images=lambda p: torch.zeros(len(p), 3, 518, 518))
    agent._dino_out = [None]
    agent.reset()
    assert isinstance(agent.memory, EpisodicGEM)
    rgb = BytesIO()
    Image.new('RGB', (32, 24)).save(rgb, format='JPEG')
    jpeg = rgb.getvalue()
    for i in range(7):
        assert agent.memory.write(jpeg) == i
    agent.memory.current_relative_depth = torch.ones(2, 2)
    agent.memory.current_depth_frame = 6
    monkeypatch.setattr(agent.memory.stream, '_forward', Mock(side_effect=RuntimeError('write failed')))
    with pytest.raises(RuntimeError, match='write failed'):
        agent.memory.write(jpeg)
    assert agent.n == 7 and agent.memory.current_depth_frame is None
    with pytest.raises(RuntimeError, match='reset'):
        agent.memory.read_dense()
    with pytest.raises(RuntimeError, match='reset'):
        agent.memory.read_sparse(b'goal', [])
    old_dir = agent.rgb_dir
    agent.reset()
    assert agent.n == 0 and agent.memory.failure is None and agent.rgb_dir != old_dir
    assert agent.memory.write(jpeg) == 0


def test_cached_dense_prediction_needs_no_transformer_token_copy(tmp_path):
    from MemNavData.monocular_depth_runtime import compute_first40_scale_receipt, decode_monocular_depth_payload

    backend = object.__new__(MemNavAgent)
    backend.lb = SimpleNamespace(compute_metric_scale=Mock(return_value=(2., dict(
        h_est=.25, n_points=100, n_frames=40, n_valid=30, h_iqr=.025))))
    memory = EpisodicGEM(backend)
    memory.rgb_dir = str(tmp_path)
    for frame in range(41):
        Image.new('RGB', (16, 12)).save(tmp_path / f'{frame}.jpg')
    memory.frame_count = 41
    memory.current_rgb_sha256 = 'a' * 64
    memory.metric_scale_receipt = compute_first40_scale_receipt(backend.lb, tmp_path, np.zeros((40, 9)), .5)
    memory.current_depth_frame = 40
    memory.current_relative_depth = torch.ones(4, 4)
    assert memory.current_aggregation is None and memory.patch_start_index is None
    result = memory.read_dense()
    assert result['frame_index'] == 40
    assert result['depth_prediction_runtime_ms'] == 0
    assert result['depth_prediction_cache_hit'] is True
    depth, _ = decode_monocular_depth_payload(result, expected_image_sha256='a'*64)
    np.testing.assert_array_equal(depth, np.full((4, 4), 2.))
