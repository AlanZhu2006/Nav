#!/usr/bin/env python3
"""Complete the three remaining frozen Bullet histories; never rerun history 0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.run_cec_stream_depth_closed_loop import BENCH, MEM_PY, hab_env

HERE = Path(__file__).resolve()
RUNNER = ROOT / "MemNavData/run_habitat_bullet_query_local.py"
BULLET_PY = ROOT / ".diagnostics/habitat_bullet_env_20260908/bin/python"
PRIOR = ROOT / ".diagnostics/habitat_physics_executor_20260908/query_four_arm_v3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memnav-port", type=int, default=21680)
    parser.add_argument("--navdp-port", type=int, default=21681)
    args = parser.parse_args()
    out = args.out.resolve()
    histories = json.loads((BENCH / "manifest.json").read_text())["episodes"]
    expected = ["gxdoqLR6rwA", "pLe4wQe7qrG", "yqstnuAEVhm", "mJXqzFtmKg4"]
    assert [row["scene"] for row in histories] == expected
    prior_manifest = json.loads((PRIOR / "manifest.json").read_text())
    prior_verification = json.loads((PRIOR / "independent_verification.json").read_text())
    assert prior_verification["verified"]
    assert sha(BENCH / "manifest.json") == prior_manifest["source_manifest_sha256"]
    # The only runtime-source change is selecting a different existing history.
    for name, digest in prior_manifest["sources"].items():
        if Path(name) != RUNNER:
            assert sha(Path(name)) == digest, f"Shared execution/model source changed: {name}"
    for name, digest in prior_manifest["weights"].items():
        assert sha(Path(name)) == digest, f"Model weights changed: {name}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    plan = [dict(index=i, scene=histories[i]["scene"], episode=histories[i]["episode"],
                 history_sha256=histories[i]["online_a_trace_sha256"])
            for i in (1, 2, 3)]
    dump(out / "manifest.json", {
        "scope": "remaining three consumed histories; query-stage physical execution diagnostic",
        "reused_history_zero": str(PRIOR),
        "reused_verification_sha256": sha(PRIOR / "independent_verification.json"),
        "source_manifest_sha256": sha(BENCH / "manifest.json"),
        "runner_sha256": sha(RUNNER), "batch_source_sha256": sha(HERE),
        "histories": plan, "max_steps": 600, "trackers": ["pure_pursuit", "mpc"],
        "policies": ["native", "cec"], "ports": [args.memnav_port, args.navdp_port],
        "arm_order": prior_manifest["variants"], "no_method_tuning": True,
        "inferential_statistical_claim": False,
        "invalid_physics_handling": "retain error; do not count as navigation failure or substitute old executor",
    })
    results = []
    for item in plan:
        index, scene = item["index"], item["scene"]
        run = out / f"history_{index:02d}_{scene}"
        started = time.monotonic()
        print(f"START history {index} {scene}; four arms", flush=True)
        command = [str(BULLET_PY), "-u", str(RUNNER), "local", "--out", str(run),
                   "--history-index", str(index), "--max-steps", "600", "--trackers", "both",
                   "--memnav-port", str(args.memnav_port), "--navdp-port", str(args.navdp_port)]
        dump(out / "progress.json", {"active": item, "completed_histories": results,
                                    "stage": "navigation", "active_run": str(run)})
        with (out / "logs" / f"history_{index:02d}.log").open("x") as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT).returncode
        row = dict(**item, run=str(run), navigation_exit_code=code,
                   navigation_wall_seconds=time.monotonic()-started)
        if code == 0:
            with (out / "logs" / f"verify_{index:02d}.log").open("x") as stream:
                checked = subprocess.run([MEM_PY, str(ROOT / "MemNavData/verify_habitat_bullet_query.py"), str(run)],
                                         cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            row["verification_exit_code"] = checked.returncode
            if checked.returncode == 0:
                verified = json.loads((run / "independent_verification.json").read_text())
                row["verified"] = verified["verified"]
                row["results"] = verified["results"]
                dump(out / "progress.json", {"active": item, "completed_histories": results,
                                            "stage": "first_person_render", "active_run": str(run)})
                with (out / "logs" / f"render_{index:02d}.log").open("x") as stream:
                    rendered = subprocess.run([str(BULLET_PY), "-u", str(ROOT / "MemNavData/render_habitat_bullet_query.py"),
                                               "--run", str(run), "--out", str(run / "first_person_v1")],
                                              cwd=ROOT, env=hab_env(), stdout=stream, stderr=subprocess.STDOUT)
                row["render_exit_code"] = rendered.returncode
        elif (run / "failure.json").exists():
            row["failure"] = json.loads((run / "failure.json").read_text())
        row["total_wall_seconds"] = time.monotonic()-started
        results.append(row)
        dump(out / "partial_results.json", results)
        print(f"DONE history {index} {scene}: navigation={code}, verified={row.get('verified', False)}", flush=True)
    completed = all(row.get("verified") and row.get("render_exit_code") == 0 for row in results)
    dump(out / "summary.json", {"completed": completed, "attempted_histories": 3,
         "results": results, "reused_history_zero": str(PRIOR),
         "scope": "four consumed histories in total, not formal SR confirmation",
         "inferential_statistical_claim": False})
    dump(out / "progress.json", {"active": None, "stage": "finished", "completed": completed,
                                "completed_histories": results})
    print(f"BATCH FINISHED completed={completed}", flush=True)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
