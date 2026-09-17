import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from MemNavData.certified_relocalization_runtime import (
    COVERAGE_ABLATION_AUTHORITY_POLICY, STRICT_AUTHORITY_POLICY,
    UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
)
from MemNavData.extract_low_covisibility_witnesses import first_decision
from MemNavData.score_low_covisibility_witnesses import angle, endpoint_bearing, load_evidence
from NavDP.baselines.memnav.policy_agent import MemNavAgent


def test_endpoint_bearing_uses_controller_axes_not_goal_orientation():
    state = {"x": 0., "z": 0., "yaw": 0.}
    np.testing.assert_allclose(endpoint_bearing(state, [0, 8, -2]), [1, 0])
    np.testing.assert_allclose(endpoint_bearing(state, [-2, 8, 0]), [0, 1])
    state["yaw"] = np.pi / 2
    np.testing.assert_allclose(endpoint_bearing(state, [-2, 0, 0]), [1, 0], atol=1e-12)
    assert angle([1, 0], [-1, 0]) == 180.
    assert angle([0, 0], [1, 0]) is None


def test_first_decision_requires_same_rgb_for_pose_and_physical_state():
    plans = {"query_leg": [{"step": 0, "frame_idx": 40}],
             "rollout_traces": {"query": [{"step": 0, "jpg_sha256": "a"}]}}
    poses = [{"frame_idx": 40, "image_sha256": "b", "camera_pose9": [], "pose_count": 41}]
    with pytest.raises(ValueError, match="frame-bound"):
        first_decision(plans, poses)


def test_incomplete_or_duplicated_extract_is_not_reported_as_full(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text('EVIDENCE_HEADER {"expected_rows": 1}\nEVIDENCE_ROW {"task": 1}\n')
    with pytest.raises(ValueError, match="incomplete"):
        load_evidence(path)
    path.write_text('EVIDENCE_HEADER {"expected_rows": 2}\n'
                    'EVIDENCE_ROW {"task": 1}\nEVIDENCE_ROW {"task": 1}\n'
                    'EVIDENCE_FINAL {"completed": true, "rows": 2}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        load_evidence(path)


def agent_fixture(tmp_path, monkeypatch, *, area=0.003, inliers=32):
    from MemNavData import certified_relocalization_runtime as runtime
    from MemNavData import lingbot_pnp_localization as pnp_module
    goal = b"goal-jpeg"
    agent = MemNavAgent.__new__(MemNavAgent)
    agent.n, agent.S = 80, 8
    agent.rgb_dir = str(tmp_path)
    agent.lb = SimpleNamespace(patch_size=14)
    agent.cam_pose = [torch.tensor([0., 0., 0., 0., 0., 0., 1., 1., 1.])] * agent.n
    agent._goal_start_frame = {hashlib.md5(goal).hexdigest(): 79}
    agent._certified_relocalization_cache = {}
    for i in (40, 56):
        (tmp_path / f"{i}.jpg").write_bytes(b"history")
    points = np.column_stack([np.arange(40), np.arange(40)])
    matched = dict(reference_raw_points=points, query_raw_points=points,
                   reference_points=points, query_points=points, scores=np.ones(40),
                   reference_raw_hw=(270, 480), query_raw_hw=(270, 480))
    agent.certified_relocalization_matcher = SimpleNamespace(match_paths=lambda *a, **kw: matched)
    support = dict(lightglue_matches=40, lightglue_score_median=.8,
                   fundamental_inliers=inliers, fundamental_query_grid_coverage=.25,
                   fundamental_query_hull_coverage=area,
                   fundamental_reference_hull_coverage=area)
    monkeypatch.setattr(runtime, "fundamental_support", lambda *a, **kw: dict(support))
    calls = []
    pnp = dict(status="ok", inliers=32, query_inlier_coverage=area,
               reference_inlier_coverage=area, reprojection_rmse_px=1.,
               pose9=[-1., 0., 2., 0., 0., 0., 1., 1., 1.])

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(pnp)

    monkeypatch.setattr(pnp_module, "correspondence_pnp_localize", solve)
    monkeypatch.setattr(pnp_module, "jsonable_pnp", lambda value: value)
    agent._certified_reference_depth = lambda anchor: (np.ones((4, 4)), np.ones((4, 4)))
    agent._certified_view_alignment = lambda pose: {}
    agent._certified_graph_direction = lambda **kw: (kw["direct_bearing"], {})
    candidates = [{"anchor": 40, "score": .9}, {"anchor": 56, "score": .8}]
    return agent, goal, candidates, calls


def test_agent_reaches_same_pnp_after_area_precheck_is_removed(tmp_path, monkeypatch):
    agent, goal, candidates, calls = agent_fixture(tmp_path, monkeypatch)
    strict = agent.certified_relocalize(goal, candidates)
    assert not strict["accepted"]
    assert strict["reason"] == "precheck_fundamental_query_hull_coverage"
    assert not calls
    # Different policies must not silently reuse the strict cached rejection.
    cached = agent.certified_relocalize(
        goal, candidates, authority_policy=COVERAGE_ABLATION_AUTHORITY_POLICY)
    assert cached["reason"] == "candidate_contract_changed"
    agent._certified_relocalization_cache.clear()
    result = agent.certified_relocalize(
        goal, candidates, authority_policy=COVERAGE_ABLATION_AUTHORITY_POLICY)
    assert result["accepted"] and not result["certificate"]["accepted"]
    assert result["selected_anchor"] == strict["selected_anchor"] == 40
    np.testing.assert_allclose(result["aux_pose"], [2., 1.])
    assert len(calls) == 1
    again = agent.certified_relocalize(
        goal, candidates, authority_policy=COVERAGE_ABLATION_AUTHORITY_POLICY)
    assert again["cached"] and again["accepted"] and len(calls) == 1
    np.testing.assert_array_equal(result["aux_pose"], again["aux_pose"])


def test_agent_strong_support_unchanged_and_low_inlier_precheck_kept(tmp_path, monkeypatch):
    agent, goal, candidates, calls = agent_fixture(tmp_path, monkeypatch, area=.2)
    strict = agent.certified_relocalize(goal, candidates)
    agent._certified_relocalization_cache.clear()
    no_cov = agent.certified_relocalize(
        goal, candidates, authority_policy=COVERAGE_ABLATION_AUTHORITY_POLICY)
    assert strict["accepted"] and no_cov["accepted"]
    for key in ("selected_anchor", "ranked_candidates", "pnp", "aux_pose"):
        assert strict[key] == no_cov[key]
    assert len(calls) == 2
    agent, goal, candidates, calls = agent_fixture(tmp_path, monkeypatch, inliers=15)
    no_cov = agent.certified_relocalize(
        goal, candidates, authority_policy=COVERAGE_ABLATION_AUTHORITY_POLICY)
    assert not no_cov["accepted"] and not calls
    assert no_cov["reason"] == "precheck_fundamental_inliers"
