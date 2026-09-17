#!/usr/bin/env python3
"""Finish measurement verification/video export once the local batch closes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.run_cec_stream_depth_closed_loop import MEM_PY, hab_env
from MemNavData.run_habitat_bullet_query_local import BULLET_PY
from MemNavData.habitat_executor_audit import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--batch-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    dump(root/"finalizer_process.json", dict(pid=os.getpid(), batch_pid=args.batch_pid))
    while not (root/"summary.json").exists():
        if (root/"failure.json").exists():
            dump(root/"finalization_status.json", dict(status="batch_interface_failure", evaluated=False))
            return 1
        try:
            os.kill(args.batch_pid, 0)
        except ProcessLookupError:
            dump(root/"finalization_status.json", dict(status="batch_ended_without_summary", evaluated=False))
            return 1
        time.sleep(20)
    with (root/"logs/independent_verification.log").open("x") as stream:
        code = subprocess.run([MEM_PY, str(ROOT/"MemNavData/verify_habitat_xnavdp_stack.py"), str(root)],
                              cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        dump(root/"finalization_status.json", dict(status="verification_failed", exit_code=code))
        return code
    env = dict(hab_env(), PYTHONPATH=f"{ROOT}:{ROOT}/MemNavData")
    for history in sorted(root.glob("history_*")):
        invalid = (history/"failure.json").exists()
        name = "render_habitat_bullet_partial.py" if invalid else "render_habitat_bullet_query.py"
        with (root/f"logs/render_{history.name}.log").open("x") as stream:
            code = subprocess.run([str(BULLET_PY), str(ROOT/"MemNavData"/name),
                                   "--run", str(history), "--out", str(history/"first_person_v1")],
                                  cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code:
            dump(root/"finalization_status.json", dict(status="verified_video_export_failed", history=history.name))
            return code
    dump(root/"finalization_status.json", dict(status="verified_and_videos_exported", formal_SR_claim=False))
    print("All measurements verified; first-person replays exported.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
