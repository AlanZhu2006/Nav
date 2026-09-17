"""Second, disjoint cleanup batch; reuse the first batch's verified archiver.

Only old runtime buffers are eligible. Actual source histories and all result
tables, logs, traces, datasets, models and code snapshots remain in place.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve()
BASE = HERE.with_name("archive_old_hpc_buffers_20260909.py")
BASE_SHA = "521f0fcdcc3537b2927dd5b219865759c396b6a40c7179bb670a21245bdddb4c"
spec = importlib.util.spec_from_file_location("frozen_buffer_archiver", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.require(base.digest(BASE) == BASE_SHA, "first-batch archiver changed")
from finish_verified_buffer_prune_20260909 import finish, write_pointer

base.EXPERIMENTS = (
    "hm3d_fresh_fullmono_mixed_role_20260820",
    "lifelong_nnr_controller_20260822",
    "final14_mono_factorial_20260819",
    "final14_cec_authority_20260828",
    "hm3d_vint_controller_native_cec_20260828",
    "mdtec_monocular_cec_composition_20260819",
    "hm3d_fullmono_lifelong_20260824",
    "hm3d_longrange_oracle_attribution_20260903",
)
base.COLD = base.ROOT / "maintenance/cold_buffers_phase2_20260909/verified_archives"
base.FINAL_COLD = Path("/archive/yz11502/nav_runtime_buffers_phase2_20260909")
POPULATION = base.ROOT / (
    "hm3d_covisibility_repaired_20260909/run_bfb8fc3d5a1bb081/construction/population.json")
POPULATION_SHA = "182f3a6d2519d5b2c178b88345db4d0bb678088dd88487cfecaad9814c6fdaaf"
original_write = base.write


def write(path, value):
    if Path(path).name.endswith(".ARCHIVED_20260909.json"):
        return write_pointer(path, value, original_write, base)
    return original_write(path, value)


base.write = write
# Export the same small interface used by the existing cold-transfer worker.
ROOT, COLD, FINAL_COLD = base.ROOT, base.COLD, base.FINAL_COLD
EXPERIMENTS = base.EXPERIMENTS
digest, require = base.digest, base.require
validate_target, inventory, migrate = base.validate_target, base.inventory, base.migrate


def source_inputs():
    require(digest(POPULATION) == POPULATION_SHA, "active population changed")
    protected = {POPULATION.parent}
    for task in json.loads(POPULATION.read_text())["tasks"]:
        construction = Path(task["construction"])
        payload = json.loads(construction.read_text())
        require(digest(construction) == task["construction_sha256"], "construction changed")
        online = Path(payload["online_a_episode"])
        receipt = json.loads((online / "receipt.json").read_text())
        require(digest(online / "receipt.json") == payload["online_a_receipt_sha256"], "history changed")
        protected.update((construction.parent, online, Path(receipt["source_episode"]),
                          Path(receipt["source_asset"])))
    return sorted(p.resolve() for p in protected)


def discover(plan):
    base.discover(plan)
    payload = json.loads(plan.read_text())
    protected = source_inputs()
    targets, excluded = [], []
    for text in payload["targets"]:
        p = Path(text)
        # Collection buffers are excluded even if no active input names them.
        collection = any(part.startswith("collect_") or part == "goal_a" for part in p.parts)
        overlap = any(p == q or p.is_relative_to(q) or q.is_relative_to(p) for q in protected)
        nonempty = any(files for _, _, files in os.walk(p))
        if collection or overlap or not nonempty:
            excluded.append({"target": text, "collection": collection,
                             "active_input_overlap": overlap, "empty": not nonempty})
        else:
            targets.append(text)
    payload.update(targets=targets, wrapper_sha256=digest(HERE),
        protected_input_roots=[str(p) for p in protected],
        protected_population_sha256=POPULATION_SHA, excluded=excluded,
        counts_by_experiment=dict(Counter(Path(p).relative_to(ROOT).parts[0] for p in targets)))
    write(plan, payload)
    print(json.dumps({"eligible_targets": len(targets), "excluded_targets": len(excluded),
        "protected_input_roots": len(protected), "plan_sha256": digest(plan),
        "counts_by_experiment": payload["counts_by_experiment"]}, indent=2), flush=True)


def execute(plan, plan_sha, index, receipts):
    payload = json.loads(plan.read_text())
    require(payload["wrapper_sha256"] == digest(HERE), "cleanup wrapper changed")
    require(payload["protected_population_sha256"] == digest(POPULATION), "active population changed")
    try:
        base.archive_one(plan, plan_sha, index, receipts)
    except PermissionError:
        # A verified archive must exist before permission repair can delete
        # anything. The helper verifies the remaining files and restores modes.
        matches = list(receipts.glob(f"{index:03d}_*.json"))
        require(len(matches) == 1, "permission error occurred before a durable archive receipt")
        row = json.loads(matches[0].read_text())
        require(row["status"] == "archive_verified_before_prune", "not a resumable prune")
        finish(matches[0], base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("discover", "execute", "migrate"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args()
    require(os.environ.get("USER") == "yz11502", "wrong account")
    if args.mode == "discover":
        discover(args.plan)
    elif args.mode == "execute":
        execute(args.plan, args.plan_sha, args.index, args.receipts)
    else:
        migrate(args.receipts)
