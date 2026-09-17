"""Fixed new initial conditions for continuous Table II; no outcome-driven refill.

The known runtime is copied from its sealed receipt. Only an explicitly audited
offline candidate-budget edit is allowed before sealing the bootstrap bundle.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from MemNavData.table2_mixed_local import load, dump, sha
from MemNavData.table2_novel_sampling import initial_a_yaw
from MemNavData.table2_balanced_sampling import SCHEMA as SOURCE_RECIPE

BASE_SHA = "9b988e7d335a4bea3329db29a6abc967321cfd4e33735e10690ea15859ea1950"
SCHEMA = "table2_continuous_expansion_20260912_v2"
PER_SCENE = 32
NEW_FILES = (
    "table2_continuous_expansion.py", "test_table2_continuous_expansion.py",
    "slurm_table2_expansion_construct.sbatch", "slurm_table2_expansion_freeze.sbatch",
    "slurm_table2_expansion_summary.sbatch", "TABLE2_CONTINUOUS_EXPANSION_PROTOCOL_20260912.md",
)


def declarations(previous, old_population, *, per_scene=PER_SCENE):
    """Reuse only physical scenario carriers, with new goal keys, yaw and seed."""
    if per_scene != PER_SCENE:
        raise ValueError("The declared expansion is exactly 32 new cases per scene")
    by_scene = defaultdict(list)
    for row in previous["sources"]:
        by_scene[row["scene"]].append(row)
    if len(by_scene) != 18 or any(len(rows) != 8 for rows in by_scene.values()):
        raise ValueError("Expected the existing 18-scene, eight-carrier declaration")
    assets = {}
    for task in old_population["tasks"]:
        value = (task["source"]["asset"], task["source"]["asset_sha256"])
        if task["scene"] in assets and assets[task["scene"]] != value:
            raise ValueError("Inconsistent asset binding")
        assets[task["scene"]] = value
    sources, portable = [], []
    for scene in sorted(by_scene):
        carriers = sorted(by_scene[scene], key=lambda r: r["episode"])
        for replicate in range(per_scene):
            carrier = carriers[replicate % len(carriers)]
            index = len(sources)
            episode = f"episode_{8 + replicate:04d}"
            identity = f"{scene}/{episode}"
            row = deepcopy(carrier)
            source = row["initial_state"]
            source.update(episode=episode, seed=2026120000 + index,
                start_yaw=initial_a_yaw(f"{SOURCE_RECIPE}/{identity}"),
                prefix_root=None, prefix_trace_sha256=None, stage_number=0)
            row.update(index=index, source_id=identity, episode=episode,
                seed=source["seed"], seed_rank=index, initial_state=source,
                parent_scenario_source_id=carrier["source_id"],
                reuse_index=replicate // len(carriers), expert_history_used=False)
            sources.append(row)
            remote = deepcopy(row)
            asset, checksum = assets[scene]
            remote["asset"] = remote["initial_state"]["asset"] = asset
            remote["initial_state"]["asset_sha256"] = checksum
            remote["source_files"] = {asset: checksum}
            portable.append(remote)
    if {s["source_id"] for s in sources} & {s["source_id"] for s in previous["sources"]}:
        raise ValueError("Expansion reuses an old task identity")
    if len({s["seed"] for s in sources}) != len(sources):
        raise ValueError("Seeds are not unique")
    common = dict(schema=previous["schema"], protocol_sha256=previous["protocol_sha256"],
        source_scenes=18, source_budget=len(sources), expansion_schema=SCHEMA,
        construction_novel_views_per_cell=48, source_selection="all existing scenario carriers, balanced reuse; new yaw/goal key/seed",
        prior_navigation_outcomes_read=False, expert_history_used=False,
        population="new initial-condition cases in previously used MP3D scenes; not new-scene confirmation")
    return dict(common, sources=sources), dict(common, sources=portable)


def copy_receipted(base, out, *, omit_population=False):
    out.mkdir(parents=True, exist_ok=False)
    copied = {}
    for line in (base / "SOURCE_BUNDLE.sha256").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if sha(base / name) != checksum:
            raise ValueError(f"Sealed input changed: {name}")
        if omit_population and (name.startswith("continuous_population/") or name == "continuous_runtime_delta.json"):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
        copied[name] = checksum
    return copied


def prepare(base, out):
    if sha(base / "SOURCE_BUNDLE.sha256") != BASE_SHA:
        raise ValueError("Expected the completed 97-source continuous runtime")
    copied = copy_receipted(base, out, omit_population=True)
    for name in NEW_FILES:
        shutil.copyfile(Path(__file__).with_name(name), out / "MemNavData" / name)
    previous = load(base / "continuous_population/source_declaration.json")
    old_population = load(base / "continuous_population/population.json")
    local, remote = declarations(previous, old_population)
    inputs = out / "expansion_inputs"
    dump(inputs / "local_source_declaration.json", local)
    dump(inputs / "source_declaration.json", remote)
    dump(inputs / "portable_plan.json", remote)
    dump(inputs / "base_files.json", copied)
    dump(inputs / "protocol.json", dict(schema=SCHEMA, base_runtime_sha256=BASE_SHA,
        source_declaration_sha256=sha(inputs / "source_declaration.json"), declared_sources=576,
        declared_scenes=18, per_scene=PER_SCENE, new_seed_and_yaw=True,
        initial_physical_positions_reused=True, novel_visual_views_per_cell=48,
        all_other_constructor_eligibility_unchanged=True, strict_area_condition_retained=True,
        prior_results_combined=False, success_rate_release_gate=False,
        desired_C_queries_per_role=10, desired_coverage_is_not_a_guarantee=True))
    print(dict(bootstrap=str(out), declared_sources=576, next="apply audited 12-to-48 constructor-only edit, check, seal"))


def check_delta(bundle):
    before = load(bundle / "expansion_inputs/base_files.json")
    changed = [name for name, checksum in before.items() if sha(bundle / name) != checksum]
    if changed != ["MemNavData/table2_balanced_sampling.py"]:
        raise ValueError(f"Unexpected runtime changes: {changed}")
    content = (bundle / changed[0]).read_text()
    if content.count("NOVEL_VIEWS_PER_CELL = 48") != 1:
        raise ValueError("Wrong candidate budget")
    import hashlib
    original = content.replace("NOVEL_VIEWS_PER_CELL = 48", "NOVEL_VIEWS_PER_CELL = 12")
    if hashlib.sha256(original.encode()).hexdigest() != before[changed[0]]:
        raise ValueError("Offline constructor changed beyond its candidate budget")
    return dict(verified=True, modified_file=changed[0], change="NOVEL_VIEWS_PER_CELL: 12 -> 48",
        other_base_files_byte_identical=len(before) - 1,
        controller_model_and_authorization_unchanged=True)


def construct_scene(bundle, run, scene_index):
    from MemNavData.table2_formal_population import prepare_a
    source_file = bundle / "expansion_inputs/source_declaration.json"
    declared = load(source_file)
    scenes = sorted({r["scene"] for r in declared["sources"]})
    if not 0 <= scene_index < len(scenes):
        raise ValueError("Scene shard outside frozen declaration")
    selected = [r["index"] for r in declared["sources"] if r["scene"] == scenes[scene_index]]
    prepare_a(source_file, run / "A_menus", indices=selected)
    dump(run / "construction_receipts" / f"scene_{scene_index:02d}.json", dict(
        scene=scenes[scene_index], indices=selected, source_declaration_sha256=sha(source_file),
        completed=True, navigation_started=False))


def sbatch(script, flags, *, dry_run=False):
    command = ["sbatch", "--test-only" if dry_run else "--parsable", *flags, str(script)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return dict(command=command, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def freeze_and_submit(bundle, run, final, *, submit=False):
    from MemNavData.table2_continuous_population import build
    from MemNavData.prepare_repaired_fullmono_bundle import seal
    declaration = bundle / "expansion_inputs/source_declaration.json"
    declared = load(declaration)
    for i, scene in enumerate(sorted({r["scene"] for r in declared["sources"]})):
        receipt = load(run / "construction_receipts" / f"scene_{i:02d}.json")
        if not receipt["completed"] or receipt["scene"] != scene or receipt["source_declaration_sha256"] != sha(declaration):
            raise ValueError("Incomplete or mismatched construction shard")
    delta = check_delta(bundle)
    copy_receipted(bundle, final)
    population = build(declaration, run / "A_menus", bundle / "expansion_inputs/portable_plan.json",
                       final / "continuous_population")
    dump(final / "expansion_inputs/runtime_delta_verified.json", delta)
    seal(final)
    population_file = final / "continuous_population/population.json"
    count = len(population["tasks"])
    info = dict(declared_sources=576, legal_A_sources=count, scenes=population["included_scenes"],
        sequence_counts=population["sequence_counts"], A_distance_counts=population["a_distance_counts"],
        excluded_sources=population["excluded_sources"], runtime_sha256=sha(final / "SOURCE_BUNDLE.sha256"),
        population_sha256=sha(population_file), frozen_bundle=str(final), navigation_outcomes_read=False)
    dump(run / "population_freeze.json", info)
    if not submit:
        print(info)
        return
    if os.environ.get("USER") != "yz11502":
        raise ValueError("Submission requires the project account")
    exports = ["ALL", f"REPAIRED_BUNDLE={final}", f"TABLE2_CONTINUOUS_RUN={run}",
        f"EXPECTED_RUNTIME_SHA={info['runtime_sha256']}", f"EXPECTED_POPULATION_SHA={info['population_sha256']}"]
    flags = ["--partition=a100_tandon", "--account=torch_pr_769_tandon_advanced", "--qos=gpu48",
        "--gres=gpu:1", "--cpus-per-task=12", "--mem=128G", "--time=01:00:00",
        "--job-name=table2_expand_eval", f"--array=0-{count-1}%4", "--export=" + ",".join(exports)]
    script = final / "MemNavData/slurm_table2_continuous_eval.sbatch"
    validation = sbatch(script, flags, dry_run=True)
    evaluation = sbatch(script, flags)
    job = evaluation["stdout"].split(";")[0]
    dump(run / "eval_submission.json", dict(**info, test_only=validation, evaluation=evaluation, eval_job_id=job))
    summary_flags = ["--partition=cpu_short", "--account=torch_pr_769_tandon_advanced",
        "--cpus-per-task=2", "--mem=8G", "--time=00:15:00", f"--dependency=afterany:{job}",
        "--kill-on-invalid-dep=yes", "--export=" + ",".join(exports)]
    summary = sbatch(final / "MemNavData/slurm_table2_expansion_summary.sbatch", summary_flags)
    dump(run / "summary_submission.json", summary)
    print(dict(eval_job_id=job, summary_job_id=summary["stdout"].split(";")[0], legal_sources=count), flush=True)


def summarize_expansion(bundle, run, out):
    from MemNavData.summarize_table2_continuous import summarize
    path = bundle / "continuous_population/population.json"
    result = summarize(path, run, len(load(path)["tasks"]))
    result["note"] = "Separate fixed expansion cohort. Construction losses remain unobserved; no old97 pooling or SR-based refill."
    stage_counts, c_sources = {}, {}
    for row in result["tasks"]:
        for arm in ("native", "cec"):
            for stage, measurement in row["arms"][arm]["observed"].items():
                role = row["sequence"]["ABC".index(stage)]
                key = f"{arm}/{stage}/{role}"
                total = stage_counts.setdefault(key, [0, 0])
                total[0] += int(measurement["reached"])
                total[1] += 1
                if stage == "C":
                    total = c_sources.setdefault(f"{arm}/after_{row['sequence'][1]}/{role}", [0, 0])
                    total[0] += int(measurement["reached"])
                    total[1] += 1
    result.update(stage_successes_queries=stage_counts, c_provenance_successes_queries=c_sources,
        C_at_least_10_issued_queries={f"{a}/{r}": stage_counts.get(f"{a}/C/{r}", [0, 0])[1] >= 10
                                   for a in ("native", "cec") for r in ("N", "R")})
    dump(out, result)
    print({k: v for k, v in result.items() if k != "tasks"}, flush=True)
    return 0 if result["technical_batch_complete"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check-delta", "construct", "freeze", "summary"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--submit", action="store_true")
    a = parser.parse_args()
    if a.mode == "prepare": prepare(a.base.resolve(), a.out.resolve())
    elif a.mode == "check-delta": print(check_delta(a.bundle.resolve()))
    elif a.mode == "construct": construct_scene(a.bundle.resolve(), a.run.resolve(), a.scene_index)
    elif a.mode == "freeze": freeze_and_submit(a.bundle.resolve(), a.run.resolve(), a.final.resolve(), submit=a.submit)
    else: sys.exit(summarize_expansion(a.bundle.resolve(), a.run.resolve(), a.out.resolve()))
