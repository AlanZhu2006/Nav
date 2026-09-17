"""Overlay experiment orchestration onto the already tested immutable runtime."""
import argparse
from pathlib import Path
import shutil

from MemNavData.prepare_repaired_fullmono_bundle import ROOT, seal, digest
from MemNavData.table2_continuous_population import build
from MemNavData.table2_mixed_local import dump

OVERLAY = (
    "table2_continuous_population.py", "table2_continuous_paired.py",
    "table2_continuous_local.py", "verify_table2_continuous_local.py",
    "verify_table2_common_pair.py", "prepare_table2_continuous_bundle.py",
    "run_table2_continuous_hpc.sh", "slurm_table2_continuous_eval.sbatch",
    "summarize_table2_continuous.py", "slurm_table2_continuous_summary.sbatch",
    "TABLE2_CONTINUOUS_POPULATION_PROTOCOL_20260912.md", "test_table2_continuous_population.py",
)


def prepare(base, out, declaration, menus, portable_plan):
    receipt = base / "SOURCE_BUNDLE.sha256"
    expected = "150243d7f415a4230ea3a1268d79dc9fef2216a7d6341e958a596cef34cee991"
    if digest(receipt) != expected:
        raise ValueError("Use the verified dual-live A100 runtime as the base")
    out.mkdir(parents=True, exist_ok=False)
    baseline = {}
    for line in receipt.read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if digest(base / name) != checksum:
            raise ValueError(f"Base runtime changed: {name}")
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, dst)
        baseline[name] = checksum
    for name in OVERLAY:
        shutil.copyfile(ROOT / "MemNavData" / name, out / "MemNavData" / name)
    changes = [name for name, old in baseline.items() if digest(out / name) != old]
    if not set(changes).issubset({"MemNavData/" + n for n in OVERLAY}):
        raise ValueError("An undeclared runtime change entered the formal bundle")
    build(declaration, menus, portable_plan, out / "continuous_population")
    dump(out / "continuous_runtime_delta.json", dict(
        base_receipt_sha256=expected, modified_existing_files=sorted(changes),
        added_files=sorted("MemNavData/"+n for n in OVERLAY if "MemNavData/"+n not in baseline),
        untouched_base_files=len(baseline)-len(changes), controller_and_model_sources_unchanged=True))
    seal(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "out", "declaration", "menus", "portable-plan"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.base.resolve(), a.out.resolve(), a.declaration.resolve(), a.menus.resolve(), a.portable_plan.resolve())
