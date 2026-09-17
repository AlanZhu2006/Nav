"""Package new Table-II orchestration with the unchanged repaired runtime."""
import argparse
import json
from pathlib import Path
import shutil

from MemNavData.prepare_repaired_fullmono_bundle import ROOT, digest, prepare, seal


FORWARD_OVERLAY = (
    "preflight_table2_mixed_hpc.py", "prepare_table2_mixed_bundle.py",
    "submit_table2_full_hpc.sh", "table2_balanced_sampling.py",
    "table2_formal_population.py", "table2_mixed_hpc.py",
    "verify_table2_full_setup.py", "table2_sampling_profiles.py",
    "TABLE2_FORWARD_NOVEL_PROTOCOL_20260911.md",
)


def prepare_forward(destination, base_bundle):
    """Copy the sealed v2 runtime, overlaying only the direction experiment."""
    receipt = base_bundle / "SOURCE_BUNDLE.sha256"
    entries = [line.split("  ", 1) for line in receipt.read_text().splitlines()]
    for expected, relative in entries:
        path = base_bundle / relative
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(base_bundle.resolve())
                or digest(path) != expected):
            raise ValueError(f"Invalid baseline bundle member: {relative}")
    destination.mkdir(parents=True, exist_ok=False)
    for _, relative in entries:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # copyfile keeps the new staging copy writable, not the sealed source.
        shutil.copyfile(base_bundle / relative, target)
        target.chmod((base_bundle / relative).stat().st_mode | 0o200)
    changes = []
    for name in FORWARD_OVERLAY:
        relative = "MemNavData/" + name
        source, target = ROOT / relative, destination / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(source)
        before = digest(target) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        changes.append(dict(file=relative, before=before, after=digest(target)))
    with (destination / "FORWARD_OVERLAY.json").open("x") as stream:
        json.dump(dict(base_receipt_sha256=digest(receipt), files=changes,
                       other_baseline_files_unchanged=True), stream, indent=2)
        stream.write("\n")


def build(destination, base_bundle=None):
    if base_bundle is not None:
        prepare_forward(destination, base_bundle)
        seal(destination)
        return
    prepare(destination)
    names = ("TABLE2_MIXED_LOCAL_PROTOCOL_20260911.md", "TABLE2_MIXED_HPC_PILOT_PROTOCOL_20260911.md",
             "run_table2_mixed_hpc.sh", "slurm_table2_mixed_eval.sbatch",
             "slurm_table2_mixed_reduce.sbatch", "submit_table2_mixed_hpc.sh",
             "TABLE2_FULL_RERUN_PROTOCOL_20260911.md", "submit_table2_full_hpc.sh",
             "TABLE2_FORWARD_NOVEL_PROTOCOL_20260911.md",
             "slurm_table2_full_reduce.sbatch", "run_table2_full_preflight.sh",
             "slurm_table2_full_preflight.sbatch")
    for name in names:
        source = ROOT/"MemNavData"/name
        if not source.is_file() or source.is_symlink():
            raise ValueError(source)
        shutil.copy2(source, destination/"MemNavData"/name)
    seal(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--base-bundle", type=Path,
                        help="For forward-Novel, inherit this sealed v2 runtime exactly")
    args = parser.parse_args()
    build(args.destination.resolve(), args.base_bundle.resolve() if args.base_bundle else None)
