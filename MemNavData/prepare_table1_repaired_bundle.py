"""Extend the existing source-only runtime bundle with RGB controllers."""
import argparse
from pathlib import Path
import shutil

from prepare_repaired_fullmono_bundle import ROOT, EXTENSIONS, prepare, seal


def build(destination):
    prepare(destination)
    files = []
    for controller in ("vint", "nomad"):
        folder = ROOT / "NavDP/baselines" / controller
        files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in EXTENSIONS
                     and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(folder).parts))
    files.extend(ROOT / "MemNavData" / name for name in (
        "TABLE1_REPAIRED_THREE_CONTROLLER_PROTOCOL_20260910.md",
        "run_table1_repaired_hpc.sh", "slurm_table1_repaired_eval.sbatch",
        "slurm_table1_repaired_summary.sbatch", "submit_table1_repaired_hpc.sh"))
    for source in files:
        if source.is_symlink() or not source.is_file():
            raise ValueError(source)
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    seal(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination.resolve())
