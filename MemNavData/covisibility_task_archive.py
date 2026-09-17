"""Lossless per-task storage; no metric recomputation or evidence rewriting.

For full re-verification, extract into a new temporary directory and bind the
extracted `task` directory at `original_work_root` inside the frozen container.
All original absolute artifact paths then resolve without modifying a receipt.
"""
import argparse
import json
from pathlib import Path
import tarfile

from repaired_covisibility_eval import dump, load, require, sha


def archive(out, durable, exit_code):
    require(out.name == "task", "archive root must be the explicit task directory")
    require(not (durable / "archive_receipt.json").exists(), "refuse to overwrite archived task")
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_symlink():
            require(path.resolve().is_relative_to(out.resolve()), "archive contains an external symlink")
        elif path.is_file():
            files.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha(path)})
    dump(out / "artifact_index.json", {"original_work_root": str(out), "files": files,
        "note": "Index excludes itself; archived unchanged receipts use the recorded original root."})
    destination = durable / "artifacts.tar.gz"
    require(not destination.exists(), "refuse to overwrite raw archive")
    temporary = durable / "artifacts.tar.gz.partial"
    with tarfile.open(temporary, "x:gz", compresslevel=1) as tar:
        tar.add(out, arcname="task")
    # Read back every member and its hash before advertising a successful task.
    with tarfile.open(temporary, "r:gz") as tar:
        index = json.load(tar.extractfile("task/artifact_index.json"))
        import hashlib
        for entry in index["files"]:
            with tar.extractfile("task/" + entry["path"]) as stream:
                h = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    size += len(chunk)
                    h.update(chunk)
            require(h.hexdigest() == entry["sha256"] and size == entry["bytes"], "archive readback mismatch")
    temporary.rename(destination)
    verified_path = out / "independent_verification.json"
    verified = load(verified_path) if verified_path.exists() else None
    complete = exit_code == 0 and verified is not None and verified.get("verified") is True
    receipt = {"completed": complete, "exit_code": exit_code, "archive": str(destination),
        "archive_sha256": sha(destination), "archive_bytes": destination.stat().st_size,
        "original_work_root": str(out), "archived_file_count": len(files)+1,
        "all_member_hashes_readback_verified": True, "verification": verified,
        "restore": "Extract to a new directory; bind extracted task at original_work_root in the frozen container; run verify --out original_work_root."}
    dump(durable / "archive_receipt.json", receipt)
    if complete:
        dump(durable / "summary.json", load(out / "summary.json"))
        dump(durable / "independent_verification.json", verified)
    print(json.dumps({k: v for k, v in receipt.items() if k != "verification"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--durable", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    archive(args.out.resolve(), args.durable.resolve(), args.exit_code)
