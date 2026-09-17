"""Retry the observed cross-node receipt-read error without weakening checks.

The original worker, archive verifier and frozen cleanup plans are unchanged.
Only errno 521 on this batch's JSON receipts is retryable; archive/data errors
still stop. An optional predecessor PID prevents concurrent transfer workers.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import follow_hpc_cold_transfer_20260909 as worker


def retryable(error, receipts):
    return (error.errno == 521 and error.filename is not None
            and Path(error.filename).parent == receipts
            and Path(error.filename).suffix == ".json")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wait-pid", type=int)
    args, forwarded = parser.parse_known_args()
    receipts = Path(forwarded[forwarded.index("--receipts") + 1])
    summary = Path(forwarded[forwarded.index("--summary") + 1])
    if args.wait_pid:
        proc = Path(f"/proc/{args.wait_pid}")
        while proc.exists():
            try:
                cmd = (proc / "cmdline").read_bytes()
            except FileNotFoundError:
                break
            if b"follow_hpc_cold_transfer_20260909.py" not in cmd or str(receipts).encode() not in cmd:
                raise RuntimeError("predecessor PID no longer identifies this batch's transfer worker")
            time.sleep(20)
        print("PREDECESSOR_EXITED", args.wait_pid, flush=True)
    if summary.is_file():
        row = json.loads(summary.read_text())
        if row.get("complete"):
            print("ALREADY_COMPLETE", flush=True)
            return
    sys.argv = [sys.argv[0]] + forwarded
    for attempt in range(12):
        try:
            worker.main()
            return
        except OSError as error:
            if not retryable(error, receipts) or attempt == 11:
                raise
            print(f"RECEIPT_READ_DEFERRED attempt={attempt+1} path={error.filename}; retry in 20s", flush=True)
            time.sleep(20)


if __name__ == "__main__":
    main()
