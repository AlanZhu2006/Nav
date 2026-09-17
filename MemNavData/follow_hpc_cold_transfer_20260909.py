"""Bounded login-node file-transfer worker for the authorized cleanup batch.

Stops when the specified CPU array leaves the queue, or after two hours. It
only moves archives already individually verified by the unchanged cleaner.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    assert args.job.isdigit()
    assert hashlib.sha256(args.code.read_bytes()).hexdigest() == args.code_sha
    spec = importlib.util.spec_from_file_location("verified_cleanup", args.code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from finish_verified_buffer_prune_20260909 import finish, write_pointer, load_receipt
    original_write = module.write
    def write(path, value):
        if Path(path).name.endswith(".ARCHIVED_20260909.json"):
            return write_pointer(Path(path), value, original_write, module)
        return original_write(path, value)
    module.write = write
    class ReadyReceipts:
        def glob(self, pattern):
            for p in args.receipts.glob(pattern):
                row = load_receipt(p)
                target = Path(row["original_target"])
                # The marker is the CPU task's final write. Avoid racing its
                # last receipt/marker publication with the cold-location update.
                if target.with_name(target.name+".ARCHIVED_20260909.json").is_file():
                    yield p
    deadline = time.monotonic() + 7200
    while True:
        # Finish an interrupted location-pointer update only when both copies
        # are still available and match the recorded archive hash.
        for p in args.receipts.glob("*.json"):
            row = load_receipt(p)
            previous = row.get("previous_scratch_archive")
            if not row.get("cold_transfer_verified") or not previous or not Path(previous).exists():
                continue
            old, cold = Path(previous), Path(row["archive"])
            module.require(old.parent == module.COLD and cold.parent == module.FINAL_COLD, "wrong duplicate location")
            module.require(module.digest(old) == module.digest(cold) == row["archive_sha256"], "duplicate archive differs")
            target = Path(row["original_target"])
            module.write(target.with_name(target.name+".ARCHIVED_20260909.json"), row)
            old.unlink()
            print(f"COLD_POINTER_RECOVERED {cold.name}", flush=True)
        # Resume only terminated failed tasks. Never race a live archiver's
        # normal publication/pruning sequence.
        accounting = subprocess.run(["sacct", "-X", "-j", args.job, "--format=JobID,State", "-n", "-P"],
                                    capture_output=True, text=True, timeout=30)
        if accounting.returncode == 0:
            failed = set()
            for line in accounting.stdout.splitlines():
                fields = line.split("|")
                name, state = fields[:2]
                suffix = name.removeprefix(args.job+"_")
                if name.startswith(args.job+"_") and suffix.isdigit() and state == "FAILED":
                    failed.add(int(suffix))
            for p in args.receipts.glob("*.json"):
                if int(p.name.split("_", 1)[0]) in failed:
                    row = load_receipt(p)
                    if row["status"] == "archive_verified_before_prune":
                        finish(p, module)
        module.migrate(ReadyReceipts())
        rows = [load_receipt(p) for p in sorted(args.receipts.glob("*.json"))]
        done = [r for r in rows if r.get("status") == "archived_and_pruned"]
        moved = [r for r in done if r.get("cold_transfer_verified")]
        queue = subprocess.run(["squeue", "-h", "-j", args.job, "-o", "%T"],
                               capture_output=True, text=True, timeout=30)
        # A completed Slurm job may no longer be known to squeue. Confirm the
        # completed population from receipts; never turn missing receipts into success.
        active = bool(queue.stdout.strip())
        result = {"cleanup_job": args.job, "expected_nonempty_targets": args.expected,
            "archived_targets": len(done), "cold_moved_targets": len(moved),
            "source_files_removed": sum(r["deleted_files"] for r in done),
            "source_bytes_in_archived_targets": sum(r["source_bytes"] for r in done),
            "source_bytes_fully_moved_off_scratch": sum(r["source_bytes"] for r in moved),
            "cold_archive_bytes": sum(r["archive_bytes"] for r in moved),
            "complete": len(moved) == args.expected, "array_active": active,
            "squeue_returncode": queue.returncode,
            "updated_unix_time": time.time(), "worker_deadline_reached": time.monotonic() >= deadline}
        module.write(args.summary, result)
        print(json.dumps(result), flush=True)
        if result["complete"] or not active or result["worker_deadline_reached"]:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
