"""Read-only metadata export for the main-table old-memory/offset audit.

Run on the HPC login node; no simulator, model, training, or GPU is invoked.
Only JSON is written to stdout. Truth fields remain offline evaluation data.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(planpath):
    plan = json.loads(planpath.read_text())
    cells = [c for c in plan["cells"] if c["controller"] == "navdp"]
    rows = []
    for c in cells:
        root = Path(c["benchmark"])
        assert sha(root / "manifest.json") == c["benchmark_sha256"]
        path = root / c["scene"] / c["episode"] / "role_pairs.json"
        assert sha(path) == c["role_pairs_sha256"]
        pair = json.loads(path.read_text())
        source = Path(pair["online_a_episode"])
        tracepath = source / "online_a_trace.json"
        assert sha(tracepath) == pair["online_a_trace_sha256"]
        assert sha(source / "receipt.json") == pair["online_a_receipt_sha256"]
        trace = json.loads(tracepath.read_text())
        rows.append(dict(cell=c, role_pairs_path=str(path), role_pairs_sha256=sha(path),
                         trace_path=str(tracepath), trace_sha256=sha(tracepath),
                         poses=[{k: p[k] for k in ("step", "x", "y", "z", "yaw")} for p in trace["poses"]],
                         decision_steps=[p["step"] for p in trace["plans"]],
                         end_position=trace["end_position"], end_yaw=trace["end_yaw"],
                         queries=[q for p in pair["pairs"] for q in p["queries"]]))
    return dict(schema="main_table_reviewer_history_offline_v1", plan=str(planpath),
                plan_sha256=sha(planpath), rows=rows, histories=len(rows),
                control_or_model_calls=0, ground_truth_scope="offline evaluation only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(
        "/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/run_7b6bffa77891ef4f/plan.json"))
    args = parser.parse_args()
    print(json.dumps(export(args.plan), separators=(",", ":"), allow_nan=False))
