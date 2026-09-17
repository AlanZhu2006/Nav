"""Small tests for experiment-only source selection and endpoint measurement."""
import numpy as np
import pytest
from flask import Flask

from MemNavData.run_cec_stream_depth_closed_loop import install_server_hooks
from MemNavData.final14_spl_replay import measurement


def agent_type():
    class FakeAgent:
        def reset(self, **kwargs):
            self._certified_route_reference_depth_cache = {}
            return kwargs

        def certified_relocalize(self, *a, **kw):
            return {"accepted": True, "reference_depth_source": kw["reference_depth_source"]}

    install_server_hooks(FakeAgent)
    return FakeAgent


@pytest.mark.parametrize("source,stride", [("canonical", 0), ("route_sparse", 1)])
def test_source_bound_by_reset_and_numpy_storage(source, stride):
    app = Flask(__name__)
    agent = agent_type()()
    with app.test_request_context(json={"cec_depth_experiment_source": source}):
        agent.reset(seed=7)
    assert agent.certified_route_depth_cache_stride == stride
    agent._cec_experiment_first = False  # The real initial state needs a model.
    agent._certified_route_reference_depth_cache[8] = (
        np.zeros((2, 3), np.float32), np.ones((2, 3), np.float32))
    result = agent.certified_relocalize(b"goal", [])
    assert result["reference_depth_source"] == source
    assert result["depth_experiment"]["online_cache_bytes"] == 48


def test_default_reset_unchanged_and_invalid_source_rejected():
    app = Flask(__name__)
    agent = agent_type()()
    agent.reset()
    assert agent.certified_route_depth_cache_stride == 0
    assert agent._cec_experiment_source == "canonical"
    with app.test_request_context(json={"cec_depth_experiment_source": "bad"}):
        with pytest.raises(ValueError):
            agent.reset()


def test_actual_endpoint_included_in_spl():
    leg = {"end_pos": [2, 0, 0], "end_psi": 0,
           "rollout_trace": [{"step": 0, "x": 0, "z": 0},
                             {"step": 1, "x": 1, "z": 0}],
           "steps": 2, "reached": True, "path_len": 7,
           "final_goal_dist_m": 0}
    result = measurement(leg, [2, 0], 1)
    assert result["actual_path_len_m"] == 2
    assert result["commanded_path_len_m"] == 7
    assert result["spl"] == 0.5
