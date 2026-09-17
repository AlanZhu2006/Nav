import json
from types import SimpleNamespace

import pytest

from MemNavData.dino_imagegoal_substitution import (
    ImageGoalSubstitution, digest, select_history_image,
)
from MemNavData import run_dino_imagegoal_local as pilot
from MemNavData.test_repaired_fullmono_local import SOURCE, option


@pytest.fixture
def history(tmp_path):
    paths = [tmp_path / f"{i:06d}.jpg" for i in range(12)]
    images = [f"causal rgb {i}".encode() for i in range(12)]
    for p, b in zip(paths, images):
        p.write_bytes(b)
    return paths, [digest(b) for b in images]


def probe():
    return dict(frame_idx=12, candidate_ceiling=11,
        certified_visual_candidates=[dict(anchor=9, score=.9), dict(anchor=8, score=.8)],
        gate=.0, predicted_gate=1., anchor=11, aux_pose=[999, 999])


def test_raw_visual_selection_ignores_gate_and_pose(history):
    paths, hashes = history
    image, row = select_history_image(probe(), paths, hashes)
    assert image == paths[9].read_bytes()
    assert row["anchor"] == 9
    assert not row["learned_gate_consumed"] and not row["geometric_pose_consumed"]


@pytest.mark.parametrize("changed", [
    {"frame_idx": 13}, {"candidate_ceiling": 12},
    {"certified_visual_candidates": []},
    {"certified_visual_candidates": [dict(anchor=12, score=1.)]},
    {"certified_visual_candidates": [dict(anchor=7, score=1.)]},
])
def test_no_current_future_or_ineligible_image(history, changed):
    with pytest.raises(ValueError):
        select_history_image(dict(probe(), **changed), *history)


def test_exact_replayed_image_identity_required(history):
    paths, hashes = history
    paths[9].write_bytes(b"changed")
    with pytest.raises(ValueError, match="replayed"):
        select_history_image(probe(), paths, hashes)


def test_ties_use_first_frame_without_confidence_threshold(history):
    p = dict(probe(), certified_visual_candidates=[dict(anchor=10, score=-.7), dict(anchor=8, score=-.7)])
    assert select_history_image(p, *history)[1]["anchor"] == 8


def test_one_append_fixed_retrieved_goal_and_unchanged_native_fields(history, tmp_path):
    calls, original_goals = [], []

    def transport(url, *args, **kw):
        calls.append((url, kw))
        payload = probe() if url.endswith("retrieval_probe_step") else {}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    base = SimpleNamespace(BASE="http://memory", NOVEL_BASE="http://navdp",
                           requests=SimpleNamespace(post=transport))

    def native_plan(image, goal, **kw):
        original_goals.append(goal)
        base.requests.post(base.BASE + "/memory_step",
            files={"image": ("image.jpg", image)}, data={"materialize_monocular_depth": "1"})
        base.requests.post(base.NOVEL_BASE + "/imagegoal_step",
            files={"image": ("image.jpg", image), "goal": ("goal.jpg", goal),
                   "depth": ("depth.png", b"original depth payload")},
            data={"diffusion_seed": str(kw["diffusion_seed"]),
                  "monocular_depth_transaction_token": "original frame token"})
        return {"trajectory": [[0., 0., 0.]]}

    base.srv_plan = native_plan
    adapter = ImageGoalSubstitution(base, *history, tmp_path / "output")
    adapter.install()
    # Prefix replay is unaffected by the query-only hook.
    base.requests.post(base.BASE + "/memory_step", files={"image": ("old.jpg", b"prefix")})
    base.srv_plan(b"current0", b"original goal", diffusion_seed=1)
    base.srv_plan(b"current1", b"original goal", diffusion_seed=2)
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == [
        "memory_step", "retrieval_probe_step", "imagegoal_step", "memory_step", "imagegoal_step"]
    assert original_goals == [b"original goal", b"original goal"]
    assert calls[1][1]["files"]["goal"][1] == b"original goal"
    assert calls[3][1]["files"] == {"image": ("image.jpg", b"current1")}
    for index, seed in ((2, "1"), (4, "2")):
        kw = calls[index][1]
        assert kw["files"]["goal"][1] == history[0][9].read_bytes()
        assert kw["files"]["depth"][1] == b"original depth payload"
        assert kw["data"] == {"diffusion_seed": seed, "monocular_depth_transaction_token": "original frame token"}
    assert len(adapter.rows) == 2
    assert all(not row["pointgoal_supplied"] for row in adapter.rows)
    assert json.loads((tmp_path / "output/retrieval_selection.json").read_text())["anchor"] == 9


def test_pilot_native_stack_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot.coverage, "task", lambda *_: (
        {}, {}, {"analysis_role": "revisit"}, {}, SOURCE, {}))
    native = pilot.command(tmp_path, 0, "native", tmp_path / "n", 21910, 21911)
    image = pilot.command(tmp_path, 0, "dino_imagegoal", tmp_path / "i", 21910, 21911)
    for flag in ("--hybrid_route", "--revisit_adapter", "--navdp_depth_source", "--max_steps",
                 "--seed", "--exec_horizon", "--revisit_controller", "--role_pair_query_role"):
        assert option(image, flag) == option(native, flag)
    assert option(image, "--hybrid_route") == "native_sidecar"
    assert pilot.ARMS == ("native", "dino_imagegoal", "raw_fixed", "cec")
    orders = [pilot.ARMS[i:] + pilot.ARMS[:i] for i in range(4)]
    assert all(set(slot) == set(pilot.ARMS) for slot in zip(*orders))
