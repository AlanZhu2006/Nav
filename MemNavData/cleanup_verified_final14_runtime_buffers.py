#!/usr/bin/env python3
"""Remove only regenerable RGB buffers from one sealed Final14 run.

This maintenance utility never parses navigation outcomes.  It first verifies
the post-hoc result receipt, all 21 task completion markers, and the exact
``task/buffer`` path boundary.  Deletion requires an explicit ``--execute``;
both dry-run and execution emit a machine-readable receipt.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED_TASKS = 21
MIN_EXPECTED_FILES = 100_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_posthoc(root: Path) -> tuple[Path, Path]:
    posthoc = root / "POSTHOC"
    summary = posthoc / "final14_mono_factorial_summary.json"
    verification = posthoc / "final14_mono_factorial_independent_verification.json"
    receipt = posthoc / "output_receipt.sha256"
    require(summary.is_file() and verification.is_file() and receipt.is_file(),
            "sealed POSTHOC artifacts are incomplete")
    rows = [line.split(maxsplit=1) for line in receipt.read_text().splitlines()
            if line.strip()]
    expected = {
        str(summary.resolve()): sha256_file(summary),
        str(verification.resolve()): sha256_file(verification),
    }
    observed = {name: digest for digest, name in rows}
    require(observed == expected, "POSTHOC output receipt is invalid")
    return summary, verification


def audit_targets(root: Path) -> list[dict]:
    tasks_root = root / "tasks"
    tasks = sorted(path for path in tasks_root.iterdir() if path.is_dir())
    require(len(tasks) == EXPECTED_TASKS,
            f"expected {EXPECTED_TASKS} task roots, found {len(tasks)}")
    rows = []
    for index, task in enumerate(tasks):
        require(task.name.startswith(f"{index:03d}_"),
                f"task ordering changed at {task.name}")
        run_log = task / "run.log"
        server_receipt = task / "server_receipt.json"
        target = (task / "buffer").resolve()
        require(run_log.is_file() and server_receipt.is_file(),
                f"task receipt is incomplete: {task.name}")
        require(f"[complete] history={index} " in run_log.read_text(
            errors="replace"), f"completion marker missing: {task.name}")
        require(target.is_dir() and target.parent == task.resolve()
                and target.name == "buffer",
                f"unsafe cleanup target: {target}")
        files = 0
        bytes_ = 0
        for path in target.rglob("*"):
            require(not path.is_symlink(), f"buffer contains symlink: {path}")
            if path.is_file():
                files += 1
                bytes_ += path.stat().st_size
        require(files > 0, f"runtime buffer is empty: {target}")
        rows.append({
            "history_index": index,
            "task": task.name,
            "target": str(target),
            "files": files,
            "bytes": bytes_,
            "run_log_sha256": sha256_file(run_log),
            "server_receipt_sha256": sha256_file(server_receipt),
        })
    require(sum(row["files"] for row in rows) >= MIN_EXPECTED_FILES,
            "audited tree is smaller than the frozen maintenance scope")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    out = args.out.resolve()
    require(root.is_dir(), f"run root is missing: {root}")
    require(not out.exists(), f"refusing to overwrite receipt: {out}")
    summary, verification = verify_posthoc(root)
    targets = audit_targets(root)
    payload = {
        "schema_version": (
            "final14_mono_factorial_runtime_buffer_cleanup_v1_20260903"),
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "sealed_run_root": str(root),
        "task_count": len(targets),
        "targets": targets,
        "total_files_removed": (
            sum(row["files"] for row in targets) if args.execute else 0),
        "total_bytes_removed": (
            sum(row["bytes"] for row in targets) if args.execute else 0),
        "total_files_audited": sum(row["files"] for row in targets),
        "total_bytes_audited": sum(row["bytes"] for row in targets),
        "posthoc_summary_sha256": sha256_file(summary),
        "posthoc_independent_verification_sha256": sha256_file(verification),
        "posthoc_receipt_verified_before_cleanup": True,
        "task_completion_markers_verified_before_cleanup": True,
        "data_class": "regenerable_runtime_rgb_buffers",
        "navigation_outcomes_read": False,
        "scientific_execution_changed": False,
    }
    out.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.execute:
        for row in targets:
            target = Path(row["target"])
            shutil.rmtree(target)
            require(not target.exists(), f"cleanup failed: {target}")
    print(json.dumps({
        "mode": payload["mode"],
        "targets": len(targets),
        "files": payload["total_files_audited"],
        "bytes": payload["total_bytes_audited"],
        "receipt": str(out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
