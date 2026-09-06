import json

import pytest

from MemNavData.freeze_route_tangent_confirmation import (
    first_strict_authorization,
)


def test_first_strict_authorization_reads_only_initial_certificate_contract():
    plan = {
        "analysis_role_not_forwarded": True,
        "query_leg": [{
            "anchor": 67,
            "certified_relocalization_accepted": True,
            "certified_relocalization_authority_policy": "strict_certificate",
            "certified_relocalization_reason": "certificate_accepted",
            "evaluation_gt_goal_distance_m": 0.0,
        }],
        "query_result": {"success": False},
    }
    assert first_strict_authorization(plan) == {
        "authorized_anchor": 67,
        "authority_policy": "strict_certificate",
        "authority_reason": "certificate_accepted",
        "analysis_role_not_forwarded": True,
    }


def test_first_strict_authorization_fails_closed():
    with pytest.raises(RuntimeError, match="lacks an initial CEC"):
        first_strict_authorization({
            "analysis_role_not_forwarded": True,
            "query_leg": [{
                "anchor": 2,
                "certified_relocalization_accepted": False,
            }],
        })
