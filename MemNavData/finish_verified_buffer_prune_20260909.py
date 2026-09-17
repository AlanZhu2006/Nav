"""Finish archive-backed pruning after a terminated task hit read-only modes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import time


def load_receipt(path):
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except OSError as error:
            if error.errno != 521 or attempt == 2:
                raise
            time.sleep(1)


def write_pointer(path, value, write, cleanup):
    path = Path(path)
    cleanup.require(path.name.endswith(".ARCHIVED_20260909.json") and path.is_relative_to(cleanup.ROOT),
                    "not a scoped archive pointer")
    cleanup.require(path.relative_to(cleanup.ROOT).parts[0] in cleanup.EXPERIMENTS, "unscoped pointer parent")
    parent = path.parent
    cleanup.require(parent.stat().st_uid == os.getuid(), "foreign pointer parent")
    mode = stat.S_IMODE(parent.stat().st_mode)
    try:
        parent.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
        write(path, value)
    finally:
        parent.chmod(mode)


def finish(receipt_path, cleanup):
    receipt = load_receipt(receipt_path)
    cleanup.require(receipt["status"] == "archive_verified_before_prune", "not a resumable verified prune")
    target, archive = Path(receipt["original_target"]), Path(receipt["archive"])
    cleanup.validate_target(target)
    cleanup.require(archive.parent == cleanup.COLD and archive.is_file(), "missing checked archive")
    cleanup.require(receipt["all_files_readback_verified"] and receipt["source_unchanged"], "archive not verified")
    cleanup.require(cleanup.digest(archive) == receipt["archive_sha256"], "archive changed")
    with tarfile.open(archive, "r:gz") as tar:
        index = json.load(tar.extractfile("archive_index.json"))
    cleanup.require(index["original_target"] == str(target), "archive target mismatch")
    expected = {r["path"]: r for r in index["files"]}
    remaining, directories = cleanup.inventory(target)
    for row in remaining:
        cleanup.require(row["path"] in expected, "new unarchived file appeared")
        old = expected[row["path"]]
        cleanup.require(all(row[k] == old[k] for k in ("bytes", "mtime_ns", "inode")), "source changed after failure")
        cleanup.require(cleanup.digest(target/row["path"]) == old["sha256"], "remaining file differs from archive")
    parent = target.parent
    cleanup.require(parent.stat().st_uid == os.getuid() and parent != cleanup.ROOT, "unsafe parent permission change")
    modes = {p: stat.S_IMODE(p.stat().st_mode) for p in [parent]+directories}
    try:
        # Only these owned directories gain owner write/execute temporarily.
        for p, mode in modes.items():
            p.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
        shutil.rmtree(target)
    finally:
        for p, mode in reversed(list(modes.items())):
            if p.exists():
                p.chmod(mode)
    cleanup.require(not target.exists(), "prune still incomplete")
    receipt.update(status="archived_and_pruned", deleted_files=receipt["source_files"],
        freed_scratch_bytes=receipt["source_bytes"], permission_repair=True,
        surviving_parent_mode_restored=stat.S_IMODE(parent.stat().st_mode) == modes[parent],
        permission_repair_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cleanup.write(receipt_path, receipt)
    write_pointer(target.with_name(target.name+".ARCHIVED_20260909.json"), receipt, cleanup.write, cleanup)
    print(f"PRUNE_RECOVERED {target}: files={receipt['source_files']} bytes={receipt['source_bytes']}", flush=True)
