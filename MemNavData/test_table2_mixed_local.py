from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from MemNavData.table2_mixed_local import compose_prefix, support_sources, runtime_spec, query_command
from MemNavData.test_hm3d_table2_leg3_mixed_role import trace


def traces():
    a = trace("episode_0000", start=0, count=3)
    b = trace("episode_0000", start=2, count=2)
    return a, b


def test_b_branches_share_only_a_and_do_not_mutate_each_other():
    a, bn = traces()
    br = deepcopy(bn)
    br["poses"][1]["z"] = 2.
    br["poses"][1]["jpg_sha256"] = "f"*64
    br["end_position"][2] = 2.
    originals = deepcopy((a, bn, br))
    n = compose_prefix(a, bn)
    r = compose_prefix(a, br)
    assert n["poses"][:3] == r["poses"][:3] == a["poses"]
    assert n["poses"][4]["jpg_sha256"] != r["poses"][4]["jpg_sha256"]
    assert "f"*64 not in {p["jpg_sha256"] for p in n["poses"]}
    assert (a, bn, br) == originals
    assert n["steps"] == r["steps"] == 5
    assert n["prefix_A_steps"] == r["prefix_A_steps"] == 3


@pytest.mark.parametrize("which", ["position", "yaw", "failed", "gem", "scene", "seed"])
def test_c_rejects_wrong_branch_state_or_collector(which):
    a, b = traces()
    if which == "position":
        b["poses"][0]["x"] += .1
    elif which == "yaw":
        b["poses"][0]["yaw"] += .1
    elif which == "failed":
        b["reached"] = False
    elif which == "gem":
        b["source_hybrid_route"] = "certified_relocalization"
    elif which == "scene":
        b["source_scene"] = "wrong"
    else:
        b["episode_seed"] += 1
    with pytest.raises(ValueError):
        compose_prefix(a, b)


@pytest.mark.parametrize("curve,expected", [
    ([.7, .2, .05, .09], "A_only"),
    ([.01, .08, .6, .2], "B_only"),
    ([.7, .2, .6, .1], "both"),
    ([.7, .2, .3, .1], "intermediate"),
])
def test_support_origin_uses_both_segments_not_sampling_frame(curve, expected):
    assert support_sources(curve, 2)["support_source"] == expected


@pytest.mark.parametrize("curve", [[float("nan")], [-.1], [1.1], [[.5]]])
def test_invalid_support_does_not_get_a_role(curve):
    with pytest.raises(ValueError):
        support_sources(curve, 0)


def query():
    return dict(schema="local", scene="s", episode="episode_0000", asset="s.glb", seed=12,
                stage_number=2, start_position=[0,0,0], start_yaw=.2, camera_height_m=.5,
                camera_intrinsic=np.eye(3).tolist(), goal_rgb="goal.jpg", goal_rgb_sha256="a"*64,
                floor_position=[2,0,0], yaw_rad=1., geodesic_m=2., prefix_root="prefix_AB",
                prefix_trace_sha256="b"*64, analysis_role="revisit", b_role="novel",
                covis_curve=[.8], max_history_covis=.8, support_source="A_only")


def test_runtime_projection_excludes_roles_and_gt_support():
    result = runtime_spec(query())
    for key in ("analysis_role", "b_role", "covis_curve", "max_history_covis", "support_source"):
        assert key not in result
    assert result["goal_rgb"] == "goal.jpg" and result["stage_number"] == 2


def test_query_command_uses_same_repaired_profile_and_declared_seed():
    q = query()
    source = dict(scene="s", episode="episode_0000", source_episode="/root/s/episode_0000",
                  asset="/root/s.glb", seed=0)
    commands = [query_command(source, q, Path("/tmp/unused-table2-output"), 19781, 19782, arm)
                for arm in ("native", "cec")]
    for c in commands:
        assert c[3] == "eval" and "table2_mixed_local.py" in c[2]
        assert "--role_pair_query_role" not in c
        assert c[c.index("--seed")+1] == "12"
        assert c[c.index("--max_steps")+1] == "600"
        assert c[c.index("--exec_horizon")+1] == "8"
        assert c[c.index("--navdp_depth_source")+1] == "monocular_sidecar"
        assert c[c.index("--retrieval_override")+1] == "off"
        assert c[c.index("--terminal_uturn")+1] == "off"
    assert commands[0][commands[0].index("--hybrid_route")+1] == "native_sidecar"
    assert commands[1][commands[1].index("--hybrid_route")+1] == "certified_relocalization"


def test_current_construction_uses_original_covisibility_depth_tolerance():
    from MemNavData.final14_role_pair_contract import DEPTH_TOLERANCE_M
    assert DEPTH_TOLERANCE_M == .30
    source = Path(__file__).with_name("table2_mixed_local.py").read_text()
    assert source.count("tol=DEPTH_TOLERANCE_M") == 2


def test_source_readout_does_not_import_the_simulator_in_supervisor():
    from MemNavData.table2_mixed_local import base_source
    from MemNavData.run_repaired_fullmono_local import sources
    for rank, source in enumerate(sources()):
        spec = base_source(source, rank)
        assert spec["seed"] == 2026091100 + rank
        assert spec["prefix_root"] is None
        assert np.asarray(spec["camera_intrinsic"]).shape == (3, 3)
        assert len(spec["start_position"]) == 3


def test_revisit_addresses_are_not_filtered_by_unperturbed_distance():
    from MemNavData.table2_mixed_local import revisit_source_frames
    # The actual 122-frame A has source 39 at 1.866 m, but its perturbed goal
    # is at 2.086 m with covisibility .707. Retain it for final-goal checking.
    assert revisit_source_frames(122) == [39, 47, 55, 63, 71, 79, 87, 95, 103]
    assert revisit_source_frames(55) == []
    assert revisit_source_frames(56) == [39]


def test_c_source_sampling_reaches_actual_b_not_only_early_a():
    from MemNavData.table2_mixed_local import revisit_source_frames
    # The new actual A has 122 frames and native Revisit-B adds 358.
    frames = revisit_source_frames(480)
    assert any(frame < 122 for frame in frames)
    assert any(frame >= 122 for frame in frames)
    assert frames == list(range(39, 464, 8))
