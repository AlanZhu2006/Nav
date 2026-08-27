"""Contract tests for the forced-reject-native lifelong baseline arm."""

import pytest

from MemNavData.independent_verify_shared_online_lifelong_nnr import (
    verify_forced_reject_plans,
)


def forced_plan(**overrides):
    plan = {
        "cec_takeover": False,
        "cec_shadow_takeover": False,
        "cec_forced_reject_native": True,
        "cec_action_state": "fallback",
    }
    plan.update(overrides)
    return plan


def test_clean_forced_plans_count_decisions_and_shadow():
    plans = [
        forced_plan(),
        forced_plan(cec_shadow_takeover=True,
                    cec_action_state="forced_reject"),
        {"memory_frame_idx": 7},  # non-decision row is ignored
    ]
    out = verify_forced_reject_plans(plans)
    assert out == {"decisions": 2, "shadow_takeovers": 1}


def test_granted_takeover_fails():
    with pytest.raises(RuntimeError, match="granted a takeover"):
        verify_forced_reject_plans([
            forced_plan(cec_takeover=True, cec_action_state="takeover"),
        ])


def test_missing_attestation_fails():
    with pytest.raises(RuntimeError, match="attestation"):
        verify_forced_reject_plans([
            forced_plan(cec_forced_reject_native=False),
        ])


def test_wrong_action_state_fails():
    with pytest.raises(RuntimeError, match="left the shared fallback"):
        verify_forced_reject_plans([
            forced_plan(cec_action_state="takeover"),
        ])


def test_eval2leg_plan_row_whitelist_carries_forced_receipts():
    """The 2026-08-22 HPC failure: the hub emitted the forced receipts but
    eval_2leg's whitelisted plan-row copy dropped them, so the contract check
    saw None.  Guard the whitelist itself."""
    import re
    from pathlib import Path
    source = Path(__file__).with_name("eval_2leg_habitat.py").read_text()
    for key in ("cec_forced_reject_native", "cec_shadow_takeover"):
        assert re.search(
            rf"{key}=response\.get\(\s*[\"']{key}[\"']\s*\)", source
        ), f"eval_2leg plan-row whitelist is missing {key}"
