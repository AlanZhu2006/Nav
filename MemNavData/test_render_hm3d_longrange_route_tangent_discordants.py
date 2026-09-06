import json

from MemNavData.render_hm3d_longrange_route_tangent_discordants import (
    ARMS,
    render_one,
)


def test_render_one_sealed_discordant(tmp_path):
    item = {
        "scene": "scene_a",
        "episode": "episode_a",
        "pairs": [{"queries": [
            {"analysis_role": "novel", "floor_position": [8.0, 0.0, 8.0]},
            {"analysis_role": "revisit", "floor_position": [1.0, 0.0, 0.0]},
        ]}],
    }
    episode_root = tmp_path / "run/evaluation/000_scene_a_episode_a"
    outcomes = {
        "mono_native": 0,
        "mono_cec_endpoint": 0,
        "mono_cec_route_tangent": 1,
    }
    for arm in ARMS:
        root = episode_root / arm
        root.mkdir(parents=True)
        end = [0.5, 0.0, 0.0] if outcomes[arm] else [-1.0, 0.0, 0.0]
        plan = {"step": 0, "navdp_critic_max": -0.25}
        if arm == "mono_cec_route_tangent":
            plan.update({
                "local_tangent_status": "active",
                "local_tangent_progress_fraction": 0.5,
                "local_tangent_unit_bearing": [1.0, 0.0],
            })
        payload = {
            "query_leg": [plan],
            "rollout_traces": {"query": [
                {"step": 0, "x": 0.0, "y": 0.0, "z": 0.0},
                {"step": 1, "x": end[0], "y": end[1], "z": end[2]},
            ]},
            "query_result": {
                "end_position": end,
                "steps": 1,
                "reached": outcomes[arm],
                "final_goal_3d_dist_m": 0.5 if outcomes[arm] else 2.0,
            },
        }
        (root / "episode_a_plans.json").write_text(json.dumps(payload))

    out = tmp_path / "discordant.mp4"
    receipt = render_one(
        item=item,
        discordant={
            "history_index": 0,
            "endpoint": 0,
            "route_tangent": 1,
        },
        run_root=tmp_path / "run",
        out_path=out,
        fps=2,
        maximum_video_frames=2,
    )
    assert out.is_file() and out.stat().st_size > 0
    assert receipt["outcomes"] == outcomes
    assert receipt["frames"] == 5
