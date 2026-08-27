from MemNavData.summarize_certified_2leg_stagnation_graph_gate import (
    exact_mcnemar,
    graph_execution,
)


def test_exact_mcnemar_small_sample_values():
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(1, 0) == 1.0
    assert exact_mcnemar(3, 0) == 0.25
    assert exact_mcnemar(5, 0) == 0.0625


def test_empty_route_request_is_not_historical_subgoal_execution():
    report = graph_execution([
        {
            "certified_graph_rescue_requested": True,
            "certified_graph_rescue_active": True,
            "certified_graph_reason": "route_complete_direct_bearing",
            "certified_graph_node": None,
            "certified_graph_count": 0,
        }
    ])
    assert report["reported_active_plan_count_pre_diagnostic_fix"] == 1
    assert report["historical_subgoal_plan_count"] == 0
    assert report["empty_route_direct_plan_count"] == 1
    assert report["executed_historical_subgoal"] is False


def test_historical_subgoal_is_execution_grounded():
    report = graph_execution([
        {
            "certified_graph_rescue_requested": True,
            "certified_graph_rescue_active": True,
            "certified_graph_reason": "historical_subgoal",
            "certified_graph_node": 42,
            "certified_graph_count": 3,
        }
    ])
    assert report["historical_subgoal_plan_count"] == 1
    assert report["executed_historical_subgoal"] is True
