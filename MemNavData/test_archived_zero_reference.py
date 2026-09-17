from MemNavData.audit_archived_zero_reference import inspect


def test_observed_motion_after_zero_endpoints_is_not_a_success_label():
    row = dict(step=0, candidate_endpoint_length_mean=0., candidate_endpoint_length_std=0.,
               trajectory_candidate_count=16, navdp_critic_max=.1,
               memory_controller_pointgoal=[-2.5, 0.], navdp_stop_evidence=False)
    trace = [dict(step=0, x=1., z=2.), dict(step=1, x=1.03, z=2.)]
    result = inspect([row], trace)[0]
    assert result["pointgoal_behind"] and result["next_action_moved_gt_1um"]
    assert result["critic_at_least_minus_half"]
    assert "reached" not in result
    assert row["candidate_endpoint_length_mean"] == 0.


def test_missing_next_position_not_filled_with_zero_displacement():
    row = dict(step=0, candidate_endpoint_length_mean=0., candidate_endpoint_length_std=0.,
               trajectory_candidate_count=16, navdp_critic_max=-.7)
    result = inspect([row], [dict(step=0, x=1., z=2.)])[0]
    assert result["next_action_observed_displacement_m"] is None
    assert result["next_action_moved_gt_1um"] is None
    assert not result["critic_at_least_minus_half"]


def test_nonzero_or_missing_endpoint_statistics_not_labeled_zero():
    assert inspect([dict(step=0, candidate_endpoint_length_mean=.1), dict(step=8)], []) == []
