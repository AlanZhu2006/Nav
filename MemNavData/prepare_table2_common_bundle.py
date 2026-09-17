"""Seal the tested common-goal runner plus one portable, already-consumed source."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from MemNavData.prepare_repaired_fullmono_bundle import prepare, seal, ROOT
from MemNavData.table2_mixed_local import load, dump
from MemNavData.table2_sampling_profiles import FORWARD_SCHEMA


def build(out):
    prepare(out)
    for name in (
        "run_table2_common_hpc.sh", "slurm_table2_common_gate.sbatch",
        "TABLE2_COMMON_PAIR_LOCAL_PROTOCOL_20260911.md",
        "TABLE2_CONTINUOUS_COMMON_TASK_DRAFT_20260911.md",
        "TABLE2_CONTINUOUS_LOCAL_PROTOCOL_20260911.md",
        "TABLE2_FORWARD_NOVEL_PROTOCOL_20260911.md",
    ):
        shutil.copy2(ROOT / "MemNavData" / name, out / "MemNavData" / name)
    plan_path = ROOT / ".diagnostics/table2_full_rerun_20260911_v2/plan.json"
    record = load(plan_path)["sources"][0]
    source = deepcopy(record["initial_state"])
    source["schema"] = FORWARD_SCHEMA
    source["asset_sha256"] = record["source_files"][source["asset"]]
    assert source["scene"] == "1LXtFkjw3qL" and source["seed"] == 2026091100
    dump(out / "common_source.json", source)
    seal(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    build(p.parse_args().out.resolve())
