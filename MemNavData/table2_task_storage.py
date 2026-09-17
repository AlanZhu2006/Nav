"""Restore an unchanged predecessor inside a per-job virtual work-root bind.

Every GPU job binds its own mktemp directory at the SAME experiment-specific
container path. Thus absolute receipt paths survive node changes without
editing JSON, changing RGB, or creating shared scratch JPEG buffers.
"""
from pathlib import Path, PurePosixPath
import tarfile

from MemNavData.table2_mixed_local import load, sha


def safe_member(member):
    def inside(name):
        path = PurePosixPath(name)
        return (not path.is_absolute() and path.parts and path.parts[0] == "task"
                and ".." not in path.parts)
    if not inside(member.name):
        raise ValueError("Archive path escapes the exact task root")
    if member.issym() or not (member.isdir() or member.isfile() or member.islnk()):
        raise ValueError("Unsupported archive member")
    if member.islnk() and not inside(member.linkname):
        raise ValueError("Archive hardlink escapes task")


def restore_task(receipt_path, target):
    receipt_path, target = Path(receipt_path), Path(target)
    receipt = load(receipt_path)
    if not (receipt["completed"] and receipt["all_member_hashes_readback_verified"]):
        raise ValueError("Only a verified complete predecessor can be restored")
    if str(target) != receipt["original_work_root"] or target.name != "task":
        raise ValueError("Bind the original work root; do not rewrite receipt paths")
    if target.exists():
        raise ValueError("Refuse to merge with an existing restored task")
    archive = Path(receipt["archive"])
    if archive.stat().st_size != receipt["archive_bytes"] or sha(archive) != receipt["archive_sha256"]:
        raise ValueError("Archived task changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            safe_member(member)
        stream.extractall(target.parent, members=members)
    index = load(target / "artifact_index.json")
    if index["original_work_root"] != str(target):
        raise ValueError("Inner archive origin differs")
    for entry in index["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid indexed path")
        path = target / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Restored artifact mismatch: {entry['path']}")
    return dict(verified=True, files=len(index["files"]), original_work_root=str(target),
                archive_sha256=receipt["archive_sha256"])
