from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from MemNavData.table2_continuous_expansion import declarations, check_delta, sbatch
from MemNavData.table2_mixed_local import dump


def fixture():
    sources, tasks = [], []
    for scene_index in range(18):
        scene = f"scene{scene_index:02d}"
        for index in range(8):
            episode = f"episode_{index:04d}"
            identity = f"{scene}/{episode}"
            spec = dict(scene=scene, episode=episode, asset=f"/local/{scene}.glb",
                seed=2026091100 + len(sources), start_position=[index, 0., scene_index],
                start_yaw=.1, camera_height_m=.5, camera_intrinsic=[[1,0,0],[0,1,0],[0,0,1]],
                prefix_root=None, prefix_trace_sha256=None, schema="forward")
            sources.append(dict(scene=scene, episode=episode, source_id=identity, seed=spec["seed"],
                asset=spec["asset"], initial_state=spec))
        tasks.append(dict(scene=scene, source=dict(asset=f"/remote/{scene}.glb", asset_sha256=scene)))
    return dict(schema="forward", protocol_sha256="frozen", sources=sources), dict(tasks=tasks)


def test_fixed_new_cases_keep_physical_and_sensor_recipe():
    previous, old_population = fixture()
    untouched = deepcopy(previous)
    local, remote = declarations(previous, old_population)
    assert previous == untouched
    assert len(local["sources"]) == 576
    assert Counter(r["scene"] for r in local["sources"]) == {f"scene{i:02d}": 32 for i in range(18)}
    assert len({r["source_id"] for r in local["sources"]}) == 576
    assert not ({r["source_id"] for r in previous["sources"]} & {r["source_id"] for r in local["sources"]})
    assert not ({r["seed"] for r in previous["sources"]} & {r["seed"] for r in local["sources"]})
    assert len({r["seed"] for r in local["sources"]}) == 576
    parents = {r["source_id"]: r for r in previous["sources"]}
    assert set(Counter(r["parent_scenario_source_id"] for r in local["sources"]).values()) == {4}
    for row, published in zip(local["sources"], remote["sources"]):
        source, parent = row["initial_state"], parents[row["parent_scenario_source_id"]]["initial_state"]
        for key in ("start_position", "camera_height_m", "camera_intrinsic"):
            assert source[key] == parent[key]
        assert source["prefix_root"] is None and source["prefix_trace_sha256"] is None
        assert source["start_yaw"] != parent["start_yaw"]
        assert published["initial_state"]["start_yaw"] == source["start_yaw"]
        assert published["asset"].startswith("/remote/")
        assert published["source_files"][published["asset"]] == published["scene"]
    assert (local, remote) == declarations(previous, old_population)


def test_input_order_or_unused_old_scores_does_not_select_new_cases():
    previous, old_population = fixture()
    expected, _ = declarations(previous, old_population)
    previous["sources"].reverse()
    for i, row in enumerate(previous["sources"]):
        row["ignored_previous_sr"] = i % 2
    actual, _ = declarations(previous, old_population)
    for row in actual["sources"]:
        row.pop("ignored_previous_sr")
    assert actual == expected


def test_expansion_cannot_be_changed_by_an_unrecorded_budget():
    previous, population = fixture()
    with pytest.raises(ValueError):
        declarations(previous, population, per_scene=64)
    previous["sources"].pop()
    with pytest.raises(ValueError):
        declarations(previous, population)


def test_delta_permits_only_the_offline_view_budget(tmp_path):
    path = tmp_path / "MemNavData/table2_balanced_sampling.py"
    path.parent.mkdir()
    original = "NOVEL_VIEWS_PER_CELL = 12\nOTHER_RULE = .55\n"
    path.write_text(original.replace("= 12", "= 48"))
    protected = tmp_path / "controller.py"
    protected.write_text("frozen = True\n")
    dump(tmp_path / "expansion_inputs/base_files.json", {
        "MemNavData/table2_balanced_sampling.py": hashlib.sha256(original.encode()).hexdigest(),
        "controller.py": hashlib.sha256(protected.read_bytes()).hexdigest()})
    assert check_delta(tmp_path)["controller_model_and_authorization_unchanged"]
    path.write_text(path.read_text().replace(".55", ".50"))
    with pytest.raises(ValueError, match="beyond"):
        check_delta(tmp_path)
    path.write_text(original.replace("= 12", "= 48"))
    protected.write_text("frozen = False\n")
    with pytest.raises(ValueError, match="Unexpected"):
        check_delta(tmp_path)


def test_summary_reports_C_sources_and_missing_coverage(tmp_path, monkeypatch):
    from MemNavData.table2_continuous_expansion import summarize_expansion
    import MemNavData.summarize_table2_continuous as original
    bundle, run = tmp_path / "bundle", tmp_path / "run"
    dump(bundle / "continuous_population/population.json", dict(tasks=[{}, {}]))
    def arm(stages):
        return dict(observed={s: dict(reached=r) for s, r in stages.items()})
    data = dict(technical_batch_complete=True, tasks=[
        dict(sequence="NNN", arms=dict(native=arm(dict(A=True,B=True,C=False)), cec=arm(dict(A=True,B=True,C=True)))),
        dict(sequence="NRR", arms=dict(native=arm(dict(A=True,B=False)), cec=arm(dict(A=True,B=True,C=True))))])
    monkeypatch.setattr(original, "summarize", lambda *a: deepcopy(data))
    result = tmp_path / "summary.json"
    assert summarize_expansion(bundle, run, result) == 0
    import json
    output = json.loads(result.read_text())
    assert output["stage_successes_queries"]["native/C/N"] == [0,1]
    assert output["c_provenance_successes_queries"]["cec/after_R/R"] == [1,1]
    assert "native/C/R" not in output["stage_successes_queries"]
    assert not any(output["C_at_least_10_issued_queries"].values())


def test_scheduler_probe_is_read_only_and_resource_values_explicit(monkeypatch):
    import MemNavData.table2_continuous_expansion as module
    seen = []
    class Result:
        stdout, stderr = "", "test-only succeeded"
    def run(command, **kwargs):
        seen.append(command)
        return Result()
    monkeypatch.setattr(module.subprocess, "run", run)
    sbatch(Path("test.sbatch"), ["--partition=a100_tandon"], dry_run=True)
    assert seen == [["sbatch", "--test-only", "--partition=a100_tandon", "test.sbatch"]]
    code = Path(module.__file__).read_text()
    for flag in ("--partition=a100_tandon", "--mem=128G", "--time=01:00:00", "--gres=gpu:1"):
        assert flag in code
    assert "--array=0-{count-1}%4" in code
    assert "--dependency=afterany:{job}" in code
    assert 'len(load(path)["tasks"])' in code
