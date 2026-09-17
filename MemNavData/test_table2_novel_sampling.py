import math
import numpy as np
import pytest

from MemNavData.table2_novel_sampling import (
    balanced_c_sources, goal_yaw, initial_a_yaw, select_novel_view,
)
from MemNavData import test_policy_agent_graph as graph_tests


def candidate(index, stratum="front"):
    return dict(proposal_index=index, direction_stratum=stratum,
                floor_position=[index, 0., 0.], yaw_rad=goal_yaw("query", index))


def test_same_independent_goal_view_rule_covers_all_stages():
    for key in ("source/A", "source/B_N", "source/C_N_after_novel", "source/C_N_after_revisit"):
        values = [goal_yaw(key, i) for i in range(128)]
        assert len(set(values)) == 8
        assert values == [goal_yaw(key, i) for i in range(128)]
        assert all(-math.pi <= value < math.pi for value in values)
        assert all(abs(value / (math.pi/4) - round(value / (math.pi/4))) < 1e-12 for value in values)


def test_initial_a_yaw_is_independent_of_route_and_leg_count():
    assert initial_a_yaw("source") == initial_a_yaw("source")
    assert initial_a_yaw("source") != initial_a_yaw("another")
    assert -math.pi <= initial_a_yaw("source") < math.pi


def test_novel_support_checks_all_frames_including_early_nonretrievable_frames():
    candidates = [candidate(0), candidate(1)]
    def rendered(row):
        curve = np.zeros(80)
        curve[2] = .5 if row["proposal_index"] == 0 else .08
        return b"rendered-goal", curve
    result, checked = select_novel_view(candidates, "front", rendered, history_frames=80)
    assert result["proposal_index"] == 1 and len(checked) == 2
    assert result["max_covis"] == .08 and result["visual_support_measured"]


@pytest.mark.parametrize("bad_curve", [np.zeros(79), np.zeros(81), [float("nan")]*80, [1.1]*80])
def test_partial_or_invalid_history_cannot_define_novel(bad_curve):
    with pytest.raises(ValueError, match="every causal history frame"):
        select_novel_view([candidate(0)], "front", lambda _: (b"rgb", bad_curve), history_frames=80)


def test_direction_shortage_does_not_silently_substitute_rear_goal():
    def must_not_render(_):
        raise AssertionError("Another direction was substituted")
    result, checked = select_novel_view([candidate(0, "rear")], "front", must_not_render, history_frames=80)
    assert result is None and checked == []


def test_support_boundary_is_exclusive_and_empty_history_is_only_explicit():
    result, _ = select_novel_view([candidate(0)], "front", lambda _: (b"rgb", [.1]), history_frames=1)
    assert result is None
    result, _ = select_novel_view([candidate(0)], "front", lambda _: (b"rgb", []), history_frames=0)
    assert result["max_covis"] == 0.


def test_c_mix_is_balanced_before_c_outcomes_and_preserves_declared_order():
    rows = [dict(id=i, b_role=role, b_reached=reached, both_c_constructed=constructible,
                 collector="native")
            for i, role, reached, constructible in (
                (0, "novel", True, True), (1, "novel", True, True),
                (2, "revisit", False, True), (3, "revisit", True, True),
                (4, "revisit", True, False))]
    assert [row["id"] for row in balanced_c_sources(rows)] == [0, 3]
    assert len(rows) == 5
    assert balanced_c_sources(rows[:2]) == []


def test_gem_b_cannot_replace_a_failed_native_b_in_shared_c():
    row = dict(b_role="revisit", b_reached=True, both_c_constructed=True, collector="gem")
    with pytest.raises(ValueError, match="native-B"):
        balanced_c_sources([row])


def test_revisit_to_novel_clears_query_caches_not_rgb_geometry_or_scale():
    import torch
    agent = graph_tests.LifelongGoalSessionTest.make_agent()
    history = [b"A-rgb", b"B-rgb"]
    geometry, scale = object(), object()
    agent.dino_cls = history
    agent.lb = geometry
    agent._first40_scale_receipt = scale
    agent._begin_goal_session("revisit-b")
    agent._goal_start_frame["revisit-b"] = 40
    agent._goal_cache[("cls", "revisit-b")] = object()
    agent._anchor_state["revisit-b"] = {"anchor": 23, "pointgoal": [-2.5, 0.]}
    agent._certified_candidate_cache[("revisit-b", 39)] = [23]
    agent._certified_relocalization_cache["revisit-b"] = {"accepted": True, "anchor": 23}
    before_rng = np.random.get_state()
    before_torch_rng = torch.get_rng_state().clone()
    assert agent._begin_goal_session("novel-c")
    for name in ("_goal_start_frame", "_goal_cache", "_anchor_state",
                 "_certified_candidate_cache", "_certified_relocalization_cache"):
        assert getattr(agent, name) == {}
    assert agent.dino_cls is history and agent.lb is geometry
    assert agent._first40_scale_receipt is scale
    assert agent._certified_reference_depth_cache == {42: "history-only"}
    after_rng = np.random.get_state()
    np.testing.assert_array_equal(before_rng[1], after_rng[1])
    assert before_rng[2:] == after_rng[2:]
    assert torch.equal(before_torch_rng, torch.get_rng_state())


def test_empty_novel_cannot_reuse_a_previous_accepted_shortlist(tmp_path):
    # Exercise the current runtime method on a short CPU-only stream. This is
    # a state test, not a neural relocalization or closed-loop SR experiment.
    from MemNavData.test_policy_agent_graph import EarlyCertifiedShortlistTest
    import torch
    agent = graph_tests.LifelongGoalSessionTest.make_agent()
    agent.n, agent.S, agent.W = 12, 8, 32
    agent.device, agent.rgb_dir = torch.device("cpu"), str(tmp_path)
    agent.certified_relocalization_matcher = object()
    agent.lb = EarlyCertifiedShortlistTest.FakeLingBot()
    agent.dino_cls = [torch.tensor([1., 0.]) for _ in range(agent.n)]
    agent._begin_goal_session("revisit-b")
    agent._certified_candidate_cache[("revisit-b", 10)] = [{"anchor": 10}]
    agent._begin_goal_session("novel-c")
    # History ceiling before frame 8 legitimately has no eligible anchor.
    candidates, _ = agent._certified_shortlist_before_decoder_warmup(
        b"new-goal", "novel-c", frame_index=11, candidate_ceiling=7)
    assert candidates == []
    assert ("revisit-b", 10) not in agent._certified_candidate_cache
