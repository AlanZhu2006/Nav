#!/usr/bin/env python3
"""Run the frozen four consumed histories with CEC/MPC alignment off/on."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.run_cec_stream_depth_closed_loop import BENCH, MEM_PY
from MemNavData.run_habitat_bullet_query_local import BULLET_PY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primitive", type=Path, required=True)
    parser.add_argument("--memnav-port", type=int, default=21680)
    parser.add_argument("--navdp-port", type=int, default=21681)
    args = parser.parse_args()
    primitive = args.primitive.resolve() / "summary.json"
    if json.loads(primitive.read_text())["passed"] is not True:
        raise RuntimeError("Physical turn / zero-reference preflight has not passed")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    histories = json.loads((BENCH / "manifest.json").read_text())["episodes"]
    if [h["scene"] for h in histories] != ["gxdoqLR6rwA", "pLe4wQe7qrG", "yqstnuAEVhm", "mJXqzFtmKg4"]:
        raise RuntimeError("The consumed four-history population changed")
    with (out / "logs/unit_preflight.log").open("x") as stream:
        subprocess.run([MEM_PY, "-m", "unittest", "MemNavData.test_habitat_bullet_alignment", "-v"],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    runner = ROOT / "MemNavData/run_habitat_bullet_query_local.py"
    plan = [{"index": i, "scene": h["scene"], "episode": h["episode"],
             "history_sha256": h["online_a_trace_sha256"]} for i, h in enumerate(histories)]
    dump(out / "manifest.json", {
        "scope": "consumed query-stage mechanism diagnostic, not formal SR",
        "histories": plan, "arms": ["cec_mpc", "cec_mpc_first_rear_physical_alignment"],
        "source_manifest_sha256": sha(BENCH / "manifest.json"),
        "runner_sha256": sha(runner), "batch_sha256": sha(Path(__file__)),
        "primitive_result": str(primitive), "primitive_sha256": sha(primitive),
        "max_commands_including_turn": 600, "residual_m": 2.5,
        "no_navmesh_motion_projection": True, "new_method_training": False,
        "query_only_history_A_was_old_online_metric": True,
        "no_robot_or_HPC_access": True, "posthoc_population_no_inferential_claim": True,
    })
    results = []
    for item in plan:
        run = out / f"history_{item['index']:02d}_{item['scene']}"
        dump(out / "progress.json", {"active": item, "active_run": str(run),
                                     "completed_histories": results, "stage": "navigation"})
        print(f"START {item['index']} {item['scene']}: CEC/MPC alignment off/on", flush=True)
        started = time.monotonic()
        command = [str(BULLET_PY), "-u", str(runner), "local", "--out", str(run),
                   "--history-index", str(item["index"]), "--max-steps", "600",
                   "--alignment-pair", "--continue-after-invalid-physics",
                   "--memnav-port", str(args.memnav_port), "--navdp-port", str(args.navdp_port)]
        with (out / f"logs/history_{item['index']:02d}.log").open("x") as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        row = dict(**item, run=str(run), exit_code=code, elapsed_s=time.monotonic()-started)
        if (run / "summary.json").exists():
            row["summary"] = json.loads((run / "summary.json").read_text())
        if (run / "failure.json").exists():
            row["failure"] = json.loads((run / "failure.json").read_text())
        results.append(row)
        dump(out / "partial_results.json", results)
        print(f"DONE {item['index']} {item['scene']}: exit={code}", flush=True)
        if code not in (0, 2):
            dump(out / "failure.json", {"history": item, "exit_code": code,
                                         "message": "Infrastructure error, no retuning or silent fallback"})
            return code
    dump(out / "summary.json", {"all_histories_attempted": True, "results": results,
                                 "formal_SR_claim": False})
    dump(out / "progress.json", {"stage": "navigation_complete", "completed_histories": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
