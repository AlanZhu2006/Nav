from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from independent_verify_hm3d_table3_causal_survey_result import audit_raw_outcome


def _row() -> dict[str, str]:
    return {
        "reached": "0",
        "final_goal_dist_m": "2.0",
        "path_len_m": "1.5",
        "steps": "3",
        "termination_reason": "max_steps",
        "certificate_accept_plans": "0",
        "runtime_failure_plans": "0",
    }


def _payload(role_visibility):
    return {
        "query_result": {
            "reached": False,
            "final_goal_dist_m": 2.0,
            "path_len_m": 1.5,
            "steps": 3,
            "termination_reason": "max_steps",
        },
        "query_leg": [{
            "role_label_visible": role_visibility,
            "learned_pi3x_relocalization_ok": None,
        }],
    }


def test_null_endpoint_visibility_diagnostic_is_hidden():
    assert audit_raw_outcome(_row(), _payload(None), "fixture") == (0, 0)


def test_explicit_visible_role_is_rejected():
    try:
        audit_raw_outcome(_row(), _payload(True), "fixture")
    except RuntimeError as error:
        assert "intervention/runtime recount changed" in str(error)
    else:
        raise AssertionError("an explicitly visible role must fail verification")
