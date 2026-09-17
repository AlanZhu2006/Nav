"""CPU-only re-verification of a completed rollout rejected by mesh reload.

Preserve the original failed archive. Change neither navigation nor its logs.
The frozen task auditor is reused with one explicit dependency: its rollout
auditor receives a byte-verified, freshly reconstructed execution PathFinder.
"""
import argparse
import importlib.util
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

from MemNavData.table2_mixed_local import dump, load, sha
from MemNavData import table2_mixed_hpc as pipeline
from MemNavData.table2_task_storage import restore_task, safe_member
from MemNavData.covisibility_task_archive import archive


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_index(task, index):
    for row in index["files"]:
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe raw artifact path")
        p = task / rel
        if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError(f"Original artifact changed: {rel}")


def recover(args):
    plan = pipeline.validate_plan(load(args.plan))
    durable = pipeline.durable_root(args.run, args.stage, args.index)
    task = pipeline.task_root(plan, args.stage, args.index)
    receipt_path = durable / "archive_receipt.json"
    receipt = load(receipt_path)
    if receipt["completed"] or receipt["exit_code"] != 1 or receipt["verification"] is not None:
        raise ValueError("Only an archived failed verification may be recovered")
    if not receipt["all_member_hashes_readback_verified"]:
        raise ValueError("Original archive integrity was not verified")
    source_archive = Path(receipt["archive"])
    if source_archive != durable / "artifacts.tar.gz":
        raise ValueError("Archive is not the resolved task target")
    if (sha(source_archive) != args.expected_archive_sha
            or receipt["archive_sha256"] != args.expected_archive_sha
            or source_archive.stat().st_size != receipt["archive_bytes"]):
        raise ValueError("Original failed archive differs from the frozen repair target")
    if task.exists() or str(task) != receipt["original_work_root"]:
        raise ValueError("Use a fresh bind at the original virtual work root")
    task.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            safe_member(member)
        stream.extractall(task.parent, members=members)
    raw_index = load(task / "artifact_index.json")
    check_index(task, raw_index)
    log = (task / "logs/verify.log").read_text()
    if 'action["actual_position"]' not in log or "assert_allclose" not in log:
        raise ValueError("This is not the diagnosed collision replay failure")
    manifest, summary = load(task / "manifest.json"), load(task / "summary.json")
    if not summary["completed"] or summary["plan_sha256"] != sha(args.plan):
        raise ValueError("Navigation did not complete under the frozen plan")
    if (manifest["stage"], manifest["index"]) != (args.stage, args.index):
        raise ValueError("Repair task identity differs")
    source = manifest["source"]
    # Recreate causal ancestors only for the task auditor; no policy replay.
    if args.stage in ("B", "C"):
        restore_task(pipeline.durable_root(args.run, "A", source["index"]) / "archive_receipt.json",
                     pipeline.task_root(plan, "A", source["index"]))
    if args.stage == "C":
        b_index = summary["c_source"]["b_task_index"]
        restore_task(pipeline.durable_root(args.run, "B", b_index) / "archive_receipt.json",
                     pipeline.task_root(plan, "B", b_index))

    repair_root = Path(__file__).resolve().parent
    updated = module_at("rollout_with_execution_context", repair_root / "verify_repaired_fullmono_local.py")
    context = module_at("rebuilt_execution_context", repair_root / "verify_rebuilt_collision_context.py")
    proofs = task / "collision_reverification"
    proofs.mkdir()
    contexts = []

    def verified_rollout(row):
        folder = Path(row["directory"])
        mesh_copy = proofs / f"{len(contexts):03d}_rebuilt.navmesh"
        with context.rebuilt_execution_context(
                source["asset"], source["source_files"][source["asset"]],
                folder / "execution.navmesh", mesh_copy) as (pf, evidence):
            result = updated.verify_rollout(row, execution_pathfinder=pf)
            contexts.append(dict(evidence, arm=row["arm"], steps=row["steps"],
                                 all_recorded_commands_and_positions_verified=True))
            return result

    # Only the offline audit dependency is substituted. The frozen controller,
    # model requests, goals, observations and measurements are never executed.
    from MemNavData import verify_repaired_fullmono_local as frozen_auditor
    original = frozen_auditor.verify_rollout
    frozen_auditor.verify_rollout = verified_rollout
    try:
        pipeline.verify_task(task)
    finally:
        frozen_auditor.verify_rollout = original
    check_index(task, raw_index)
    evidence = dict(verified=True, stage=args.stage, index=args.index,
                    original_archive_sha256=args.expected_archive_sha,
                    original_receipt_sha256=sha(receipt_path),
                    runtime_sha256=manifest["runtime_sha256"], plan_sha256=sha(args.plan),
                    navigation_rerun=False, controller_changed=False,
                    all_original_indexed_files_unchanged=True, contexts=contexts,
                    repair_files={p.name: sha(p) for p in repair_root.iterdir()
                                  if p.suffix in (".py", ".sbatch")},
                    reason="rebuild original collision context; no tolerance relaxation")
    dump(proofs / "receipt.json", evidence)
    # Keep both the original archive and its original internal index. The
    # archive writer creates a fresh index and must not index its own old copy.
    (task / "artifact_index.json").rename(proofs / "original_artifact_index.json")
    preserved = args.run / "failed_attempts" / args.preserve_name
    if preserved.exists():
        raise ValueError("Do not overwrite an earlier failed attempt")
    preserved.parent.mkdir(parents=True, exist_ok=True)
    durable.rename(preserved)
    durable.mkdir()
    archive(task, durable, 0)
    dump(durable / "collision_reverification.json",
         dict(evidence, preserved_original_directory=str(preserved)))
    print(json.dumps(dict(verified=True, stage=args.stage, index=args.index,
                          navigation_rerun=False, preserved_original=str(preserved))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("A", "B", "C"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--expected-archive-sha", required=True)
    parser.add_argument("--preserve-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.preserve_name):
        parser.error("preserve-name must identify one failed attempt")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_archive_sha):
        parser.error("expected-archive-sha must be a SHA-256 digest")
    recover(args)
