from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NavDP.baselines.memnav.policy_agent import (
    MemNavAgent,
    flow_threshold_for_length,
)


def test_flow_threshold_length_tiers_match_training_policy():
    expected = {
        1: 20.0,
        702: 20.0,
        703: 25.0,
        877: 25.0,
        878: 30.0,
        1075: 30.0,
        1076: 40.0,
        1506: 40.0,
        1507: 50.0,
        2048: 50.0,
        2049: 60.0,
    }
    assert {n: flow_threshold_for_length(n) for n in expected} == expected


def _minimal_agent(flow_threshold):
    agent = object.__new__(MemNavAgent)
    agent.scale_k = torch.zeros(2, 3, 8, 4, 5)
    agent.scale_v = torch.ones_like(agent.scale_k)
    agent.psi = 4
    agent.anchor_k = [torch.zeros(2, 3, 4, 5), torch.ones(2, 3, 4, 5)]
    agent.anchor_v = [torch.ones(2, 3, 4, 5), torch.zeros(2, 3, 4, 5)]
    agent.cam_k = [torch.zeros(2, 2, 3, 5) for _ in range(3)]
    agent.cam_v = [torch.ones(2, 2, 3, 5) for _ in range(3)]
    agent.cam_pose = [torch.zeros(9) for _ in range(12)]
    agent.device = torch.device("cpu")
    agent.flow_threshold = flow_threshold
    agent.anchor_frame_indices = [8, 11]
    agent.cam_frame_indices = [0, 1, 11]
    return agent


def test_live_sparse_cache_carries_raw_frame_row_mapping():
    cache = _minimal_agent(20.0)._live_cache()
    assert cache["anchor_frame_indices"].tolist() == [8, 11]
    assert cache["cam_frame_indices"].tolist() == [0, 1, 11]
    assert cache["anchor_k"].shape == (2, 3, 2, 4, 5)
    assert cache["cam_pose_enc"].shape == (12, 9)


def test_live_dense_cache_omits_sparse_row_mapping():
    cache = _minimal_agent(0.0)._live_cache()
    assert "anchor_frame_indices" not in cache
    assert "cam_frame_indices" not in cache
