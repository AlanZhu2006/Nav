from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from MemNavData.table2_mixed_hpc import (
    SCHEMA, source_id, task_root, task_source, validate_plan,
    select_c, population_at,
    arm_order,
)
from MemNavData.table2_mixed_local import dump, sha
from MemNavData.table2_task_storage import restore_task, safe_member


def test_runtime_repair_keeps_plan_identity_but_records_actual_bundle(tmp_path, monkeypatch):
    import MemNavData.table2_mixed_hpc as runtime
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    (tmp_path / "SOURCE_BUNDLE.sha256").write_text("sealed repair\n")
    p = {"runtime_sha256": "original"}
    with pytest.raises(FileNotFoundError):
        runtime.runtime_provenance(p)
    repair = dict(original_bundle_sha256="original", model_parameters_changed=False,
                  controller_changed=False, jpeg_validation_removed=False)
    path = tmp_path / "multipart_repair_manifest.json"
    path.write_text(json.dumps(repair))
    result = runtime.runtime_provenance(p)
    assert result["runtime_sha256"] != "original"
    assert result["planned_runtime_sha256"] == p["runtime_sha256"] == "original"
    repair["controller_changed"] = True
    path.write_text(json.dumps(repair))
    with pytest.raises(ValueError, match="authorized transport repair"):
        runtime.runtime_provenance(p)


def plan(tmp_path):
    sources = [dict(index=i, scene="same", episode=f"episode_{i:04d}",
                    source_id=f"same/episode_{i:04d}") for i in range(2)]
    return dict(schema=SCHEMA, formal_result=False, sources=sources,
                work_root=str(tmp_path/"virtual"), arms=["native", "cec"],
                c_selection="global_equal_native_B_roles")


def test_duplicate_scene_different_episode_has_distinct_keys(tmp_path):
    p = plan(tmp_path)
    assert source_id(p["sources"][0]) != source_id(p["sources"][1])
    assert task_root(p, "A", 0) != task_root(p, "A", 1)
    assert task_source(p, "B", 2) == (p["sources"][1], "novel")
    p["work_root"] = "/tmp/memnav_table2_test"
    validate_plan(p)
    p["sources"][1].update(scene="same", episode="episode_0000", source_id="same/episode_0000")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_plan(p)


@pytest.mark.parametrize("value", ["..", "a/b", "/tmp", "s;cmd"])
def test_source_id_cannot_escape(value):
    with pytest.raises(ValueError):
        source_id(dict(scene=value, episode="episode_0000"))


def completed_b(root, index, p, digest, *, reached, constructible):
    source, role = task_source(p, "B", index)
    folder = root/"B"/f"task_{index:03d}"
    candidate = dict(source_id=source["source_id"], source_index=source["index"],
                     b_task_index=index, collector="native", b_role=role,
                     b_reached=reached, both_c_constructed=constructible)
    summary = dict(completed=True, plan_sha256=digest, stage="B", index=index, c_source=candidate)
    dump(folder/"summary.json", summary)
    dump(folder/"independent_verification.json", dict(verified=True, summary_sha256=sha(folder/"summary.json")))
    dump(folder/"archive_receipt.json", dict(completed=True, all_member_hashes_readback_verified=True))


def test_global_c_mix_does_not_require_both_roles_on_same_a(tmp_path):
    p = plan(tmp_path)
    p["work_root"] = "/tmp/memnav_table2_test"
    path = tmp_path/"plan.json"
    dump(path, p)
    # A0 only has successful Novel-B; A1 only successful Revisit-B.
    # Per-array balancing would wrongly throw away both; global selection must not.
    for i in range(4):
        completed_b(tmp_path, i, p, sha(path), reached=i in (0,3), constructible=i in (0,3))
    result = select_c(path, tmp_path, tmp_path/"c_population.json")
    assert [s["b_task_index"] for s in result["selected"]] == [0,3]
    assert result["counts"] == dict(novel=1, revisit=1)
    assert result["c_query_tasks"] == 4 and not result["C_navigation_outcomes_read"]
    assert task_source(p, "C", 2, result) == (p["sources"][1], "novel")
    assert population_at(path, tmp_path) == result
    with pytest.raises(FileExistsError):
        select_c(path, tmp_path, tmp_path/"c_population.json")


def test_one_sided_c_supply_remains_empty(tmp_path):
    p = plan(tmp_path)
    p["work_root"] = "/tmp/memnav_table2_test"
    path = tmp_path/"plan.json"
    dump(path, p)
    for i in range(4):
        completed_b(tmp_path, i, p, sha(path), reached=i%2==0, constructible=i%2==0)
    result = select_c(path, tmp_path, tmp_path/"c_population.json")
    assert result["selected"] == [] and result["c_query_tasks"] == 0


def test_missing_b_is_not_a_navigation_failure(tmp_path):
    p = plan(tmp_path)
    p["work_root"] = "/tmp/memnav_table2_test"
    path = tmp_path/"plan.json"
    dump(path, p)
    with pytest.raises(FileNotFoundError):
        select_c(path, tmp_path, tmp_path/"c_population.json")
    assert not (tmp_path/"c_population.json").exists()


def test_archive_restores_original_receipts_and_hardlinks(tmp_path):
    original = tmp_path/"virtual"/"A"/"task_000"/"task"
    staging = tmp_path/"staging"/"task"
    staging.mkdir(parents=True)
    payload = dict(prefix_root=str(original/"prefix_A"), source_id="same/episode_0000")
    dump(staging/"receipt.json", payload)
    (staging/"image.jpg").write_bytes(b"causal RGB bytes")
    import os
    os.link(staging/"image.jpg", staging/"copy.jpg")
    files = [dict(path=f.name, bytes=f.stat().st_size, sha256=sha(f)) for f in staging.iterdir()]
    dump(staging/"artifact_index.json", dict(original_work_root=str(original), files=files))
    archive = tmp_path/"artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(staging, arcname="task")
    receipt = tmp_path/"archive_receipt.json"
    dump(receipt, dict(completed=True, all_member_hashes_readback_verified=True,
         original_work_root=str(original), archive=str(archive), archive_bytes=archive.stat().st_size,
         archive_sha256=sha(archive)))
    result = restore_task(receipt, original)
    assert result["verified"] and result["files"] == 3
    assert (original/"receipt.json").read_bytes() == (staging/"receipt.json").read_bytes()
    assert (original/"image.jpg").read_bytes() == (original/"copy.jpg").read_bytes()
    with pytest.raises(ValueError, match="existing"):
        restore_task(receipt, original)
    with pytest.raises(ValueError, match="original work root"):
        restore_task(receipt, tmp_path/"other"/"task")


@pytest.mark.parametrize("name", ["/tmp/escape", "task/../../escape", "other/file"])
def test_bad_archive_paths_are_rejected(name):
    with pytest.raises(ValueError, match="escapes"):
        safe_member(tarfile.TarInfo(name))


def test_new_runner_does_not_use_local_fixed_sources():
    text = Path(__file__).with_name("table2_mixed_hpc.py").read_text()
    assert "source_rows = sources()" not in text
    assert "selected = balanced_c_sources(candidates)" in text
    assert 'key = (row["source_id"], row["phase"], row["role"])' in text


def test_arm_order_is_balanced_within_each_role():
    for role_index in (0,1):
        first = [arm_order(2*parent+role_index)[0] for parent in range(8)]
        assert first.count("native") == first.count("cec") == 4


def test_bundle_and_wrappers_cover_the_new_entrypoint():
    root = Path(__file__).parent
    builder = (root/"prepare_table2_mixed_bundle.py").read_text()
    for name in ("TABLE2_MIXED_LOCAL_PROTOCOL_20260911.md", "TABLE2_MIXED_HPC_PILOT_PROTOCOL_20260911.md",
                 "run_table2_mixed_hpc.sh", "submit_table2_mixed_hpc.sh",
                 "slurm_table2_mixed_eval.sbatch", "slurm_table2_mixed_reduce.sbatch"):
        assert name in builder and (root/name).is_file()
    submit = (root/"submit_table2_mixed_hpc.sh").read_text()
    assert "--partition=a100_tandon" in submit and "--time=01:00:00" in submit
    assert '--dependency="afterok:${TABLE2_A_GATE}:${TABLE2_A}"' in submit
    assert '--dependency="afterok:${TABLE2_B}"' in submit
    wrapper = (root/"slurm_table2_mixed_eval.sbatch").read_text()
    assert '-B "${TABLE2_HOST_TMP}:${TABLE2_WORK_ROOT}"' in wrapper
    preflight = (root/"preflight_table2_mixed_hpc.py").read_text()
    assert "query_command" in preflight and "--contract_dry_run" in preflight
