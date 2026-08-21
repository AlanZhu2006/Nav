import json

import pytest

from audit_final14_cec_latency import audit


def _write_query(root, protocol, role, plans):
    path = (
        root / "evaluation" / protocol / "000_scene_episode"
        / "certified" / f"episode_pair_00_{role}_plans.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"query_leg": plans}))


def _plan(*, cached, elapsed, first_elapsed=10.0, accepted=True, anchor=8):
    return {
        "certified_relocalization_cached": cached,
        "certified_relocalization_ms": elapsed,
        "certified_relocalization_uncached_ms": first_elapsed,
        "certified_relocalization_accepted": accepted,
        "router_selected_anchor": anchor,
    }


def test_audit_separates_first_query_from_carried_latency(tmp_path):
    for protocol in ("natural_direction", "hard_support"):
        _write_query(tmp_path, protocol, "novel", [
            _plan(cached=False, elapsed=10.0, accepted=False),
            _plan(cached=True, elapsed=0.1, accepted=False),
            _plan(cached=True, elapsed=0.2, accepted=False),
        ])
        _write_query(tmp_path, protocol, "revisit", [
            _plan(cached=False, elapsed=20.0, first_elapsed=20.0,
                  anchor=16),
            _plan(cached=True, elapsed=0.3, first_elapsed=20.0,
                  anchor=16),
        ])

    result = audit(tmp_path)
    natural = result["protocols"]["natural_direction"]
    assert natural["first_query_uncached_ms"]["n"] == 2
    assert natural["first_query_uncached_ms"]["median"] == 15.0
    assert natural["cached_update_ms"]["n"] == 3
    assert natural["cached_update_ms"]["median"] == pytest.approx(0.2)
    assert natural["legacy_carried_uncached_ms"]["n"] == 5
    assert natural["legacy_to_true_sample_ratio"] == 2.5


def test_audit_rejects_a_second_uncached_request(tmp_path):
    duplicate = [
        _plan(cached=False, elapsed=10.0),
        _plan(cached=False, elapsed=11.0),
    ]
    for protocol in ("natural_direction", "hard_support"):
        for role in ("novel", "revisit"):
            _write_query(tmp_path, protocol, role, duplicate)
    with pytest.raises(RuntimeError, match="expected one first query"):
        audit(tmp_path)
