#!/usr/bin/env python3
"""Finish the interrupted fourth RGB pair, retaining the first three pairs.

This does not resume within a stochastic rollout. Both fourth-history arms
restart in their frozen order; the interrupted directory is left untouched.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.run_cec_stream_depth_closed_loop import MEM_PY
from MemNavData.run_habitat_bullet_query_local import BULLET_PY
from MemNavData.verify_habitat_xnavdp_stack import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.out.resolve()
    if args.detach:
        out.parent.mkdir(parents=True, exist_ok=True)
        log = out.with_name(out.name + "_launcher.log")
        with log.open("x") as stream:
            process = subprocess.Popen([MEM_PY, "-u", str(Path(__file__).resolve()),
                "--parent", str(parent), "--out", str(out)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps(dict(pid=process.pid, log=str(log), result=str(out))), flush=True)
        return 0
    frozen = json.loads((parent / "manifest.json").read_text())
    assert frozen["rgb_pair"] and len(frozen["histories"]) == 4
    interrupted = parent / "history_03_mJXqzFtmKg4"
    assert not (interrupted / "summary.json").exists()
    snapshot = json.loads((interrupted / "manifest.json").read_text())
    for name, digest in snapshot["sources"].items():
        assert sha(Path(name)) == digest, f"Frozen evaluation source changed: {name}"
    for name, digest in snapshot["weights"].items():
        assert sha(Path(name)) == digest, f"Frozen weights changed: {name}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    dump(out / "owned_process.json", dict(pid=os.getpid()))
    dump(out / "manifest.json", dict(**frozen, recovery=dict(
        parent=str(parent), parent_manifest_sha256=sha(parent/"manifest.json"),
        reason="Fourth pair interrupted without terminal summary; previous processes absent",
        reused_histories=[0, 1, 2], restarted_histories=[3],
        selection="All frozen histories retained; no selection by navigation outcome")))
    results = []
    for item in frozen["histories"][:3]:
        name = f"history_{item['index']:02d}_{item['scene']}"
        source = parent / name
        verified = audit(source)
        assert verified["verified"]
        (out / name).symlink_to(source, target_is_directory=True)
        results.append(dict(**item, run=str(source), reused=True,
                            summary=json.loads((source/"summary.json").read_text())))
    item = frozen["histories"][3]
    run = out / interrupted.name
    dump(out / "progress.json", dict(stage="navigation", active=item, completed=results))
    started = time.monotonic()
    with (out / "logs/history_03.log").open("x") as stream:
        code = subprocess.run([str(BULLET_PY), "-u", str(ROOT/"MemNavData/run_habitat_bullet_query_local.py"),
            "local", "--out", str(run), "--history-index", "3", "--max-steps", "600",
            "--xnavdp-rgb-pair", "--continue-after-invalid-physics"], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT).returncode
    if code not in (0, 2):
        dump(out/"failure.json", dict(stage="fourth_pair", exit_code=code))
        return code
    results.append(dict(**item, run=str(run), reused=False, exit_code=code,
        elapsed_s=time.monotonic()-started, summary=json.loads((run/"summary.json").read_text())))
    dump(out/"summary.json", dict(all_histories_attempted=True, results=results,
        formal_SR_claim=False, recovery_parent=str(parent)))
    dump(out/"progress.json", dict(stage="navigation_complete", completed=results))
    # The same supervisor executes finalization, without an orphan waiter.
    return subprocess.run([MEM_PY, str(ROOT/"MemNavData/finalize_habitat_xnavdp_stack.py"),
        str(out), "--batch-pid", str(os.getpid())], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
