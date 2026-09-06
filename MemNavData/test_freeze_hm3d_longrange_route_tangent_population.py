import hashlib
import json

from MemNavData.freeze_hm3d_longrange_route_tangent_population import (
    deterministic_scene_cap,
    eligible_rows,
)


def _candidate(tmp_path, *, scene, identity, vertical, constructed=True,
               bin_name="20_to_30_m"):
    root = tmp_path / identity
    root.mkdir()
    payload = {
        "candidate_identity_sha256": identity,
        "online_a_endpoint": {"floor_position": [0.0, 1.0, 0.0]},
        "pairs": [{
            "queries": [
                {"analysis_role": "novel", "floor_position": [1, 4, 1]},
                {
                    "analysis_role": "revisit",
                    "floor_position": [2.0, 1.0 + vertical, 2.0],
                },
            ],
        }],
    }
    sidecar = root / "role_pairs.json"
    sidecar.write_text(json.dumps(payload, sort_keys=True))
    return {
        "constructed": constructed,
        "bin_name": bin_name,
        "candidate_identity_sha256": identity,
        "role_pair_candidate": str(root),
        "role_pairs_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "scene": scene,
    }


def test_eligibility_is_unused_midrange_and_revisit_same_floor(tmp_path):
    rows = [
        _candidate(tmp_path, scene="a", identity="1" * 64, vertical=0.5),
        _candidate(tmp_path, scene="a", identity="2" * 64, vertical=0.5001),
        _candidate(tmp_path, scene="b", identity="3" * 64, vertical=0.0,
                   bin_name="30_to_50_m"),
        _candidate(tmp_path, scene="c", identity="4" * 64, vertical=0.0,
                   constructed=False),
        _candidate(tmp_path, scene="d", identity="5" * 64, vertical=0.0),
    ]
    selected = eligible_rows(
        rows,
        consumed_identities={"5" * 64},
        distance_bin="20_to_30_m",
        maximum_vertical_error_m=0.5,
    )
    assert [row["candidate_identity_sha256"] for row in selected] == [
        "1" * 64,
    ]
    assert selected[0]["revisit_vertical_error_m"] == 0.5


def test_scene_cap_is_round_robin_and_hash_deterministic():
    rows = [
        {"scene": "b", "candidate_identity_sha256": "e" * 64},
        {"scene": "a", "candidate_identity_sha256": "d" * 64},
        {"scene": "a", "candidate_identity_sha256": "a" * 64},
        {"scene": "a", "candidate_identity_sha256": "c" * 64},
        {"scene": "b", "candidate_identity_sha256": "b" * 64},
    ]
    selected = deterministic_scene_cap(rows, maximum_per_scene=2)
    assert [(row["scene"], row["candidate_identity_sha256"][0])
            for row in selected] == [
        ("a", "a"), ("b", "b"), ("a", "c"), ("b", "e"),
    ]
