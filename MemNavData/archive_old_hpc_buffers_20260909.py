"""Move explicitly scoped old runtime buffers to recoverable cold archives.

Never remove results, online histories, datasets, checkpoints, source bundles,
or arbitrary directories. Source deletion follows full archive readback and an
unchanged-source check. All targets are resolved in a separate discovery step.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import time

ROOT = Path("/scratch/yz11502/Research/Nav-axis-uturn-results")
COLD = ROOT / "maintenance/cold_buffers_20260909/verified_archives"
FINAL_COLD = Path("/archive/yz11502/nav_runtime_buffers_20260909")
EXPERIMENTS = (
    "certified_relocalization_closed_loop_20260812", "final14_cec_learned_20260817",
    "hm3d_fullmono_lifelong_natural_b_expansion_execution_20260830",
    "hm3d_fullmono_lifelong_power_v3_20260826", "paper_certified_compass_20260814",
    "hm3d_heldout_val10_revisit_20260816", "hm3d_table3_causal_survey_expansion_20260831",
    "revisit_fresh_confirmation_20260811", "hm3d_table1_controller_portability_20260829",
    "mp3d_table1_controller_portability_20260829",
)
SKIP = {"rgb", "depth", "videos", "benchmarks", "construction", "source_generation", "source",
        "scenes", "datasets", "weights", "sealed_inputs", "evaluation", "logs", "preflight", "__pycache__"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_target(target):
    require(target.is_absolute() and not target.is_symlink() and target.resolve() == target, "unresolved target")
    require(target.is_relative_to(ROOT) and len(target.relative_to(ROOT).parts) >= 3, "target too broad")
    require(target.relative_to(ROOT).parts[0] in EXPERIMENTS, "target outside explicit old experiments")
    require(target.name in ("buffer", "buffers"), "not a runtime buffer")
    require(target.is_dir() and target.stat().st_uid == os.getuid(), "target absent or owned by another user")


def discover(output):
    require(not output.exists(), "do not overwrite cleanup plan")
    targets = []
    for name in EXPERIMENTS:
        pending = [(ROOT / name, 0)]
        while pending:
            path, level = pending.pop()
            if not path.exists():
                continue
            for item in os.scandir(path):
                if item.is_symlink() or not item.is_dir(follow_symlinks=False):
                    continue
                p = Path(item.path)
                if item.name in ("buffer", "buffers"):
                    validate_target(p)
                    targets.append(str(p))
                elif level < 5 and item.name not in SKIP:
                    pending.append((p, level+1))
    targets.sort()
    require(len(targets) == len(set(targets)), "duplicate target")
    plan = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "old runtime buffers only; archive/verify before removing scatter files; login-node cold transfer follows",
        "archive_root": str(COLD), "targets": targets,
        "counts_by_experiment": dict(Counter(Path(t).relative_to(ROOT).parts[0] for t in targets)),
        "protected": ["current 28 online A histories", "all datasets/checkpoints/conda/source bundles",
                      "evaluation logs, metrics, traces, construction and verifier results"],
        "code_sha256": digest(__file__)}
    write(output, plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "targets"}, indent=2))
    print(f"PLAN {output} SHA256 {digest(output)} TARGETS {len(targets)}")


def inventory(target):
    files, directories = [], []
    for folder, dirs, names in os.walk(target, followlinks=False):
        directories.append(Path(folder))
        for name in dirs + names:
            p = Path(folder) / name
            s = p.lstat()
            require(not stat.S_ISLNK(s.st_mode), f"refuse symlink in buffer: {p}")
            require(s.st_uid == os.getuid(), f"foreign owner: {p}")
            if stat.S_ISDIR(s.st_mode):
                continue
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, f"special/hardlinked file: {p}")
            require(s.st_mtime < time.time()-48*3600, f"recently modified file; not cold: {p}")
            files.append({"path": str(p.relative_to(target)), "bytes": s.st_size,
                          "mtime_ns": s.st_mtime_ns, "inode": s.st_ino})
    files.sort(key=lambda r: r["path"])
    return files, directories


class HashReader:
    def __init__(self, stream):
        self.stream, self.h = stream, hashlib.sha256()

    def read(self, size):
        value = self.stream.read(size)
        self.h.update(value)
        return value


def archive_one(plan_path, expected_sha, index, receipts):
    require(digest(plan_path) == expected_sha, "cleanup plan changed")
    plan = json.loads(plan_path.read_text())
    require(plan["code_sha256"] == digest(__file__), "cleanup code changed since discovery")
    require(plan["archive_root"] == str(COLD), "wrong archive location")
    target = Path(plan["targets"][index])
    validate_target(target)
    identifier = f"{index:03d}_" + hashlib.sha256(str(target).encode()).hexdigest()[:12]
    COLD.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    destination = COLD / (identifier + ".tar.gz")
    receipt_path = receipts / (identifier + ".json")
    require(not destination.exists() and not receipt_path.exists(), "already archived; inspect receipt before retry")
    before, directories = inventory(target)
    original_bytes = sum(r["bytes"] for r in before)
    require(original_bytes < 100*1024**3, "single buffer unexpectedly exceeds 100 GiB")
    temporary = destination.with_name(destination.name + ".partial")
    require(not temporary.exists(), "prior interrupted archive must be inspected")
    print(f"ARCHIVING {target}: {len(before)} files {original_bytes} bytes", flush=True)
    with tarfile.open(temporary, "x:gz", compresslevel=1) as tar:
        for folder in directories:
            info = tar.gettarinfo(str(folder), arcname=str(Path("buffer") / folder.relative_to(target)))
            tar.addfile(info)
        for row in before:
            path = target / row["path"]
            info = tar.gettarinfo(str(path), arcname="buffer/"+row["path"])
            with path.open("rb") as source:
                reader = HashReader(source)
                tar.addfile(info, reader)
            row["sha256"] = reader.h.hexdigest()
        contents = json.dumps({"original_target": str(target), "files": before}, separators=(",", ":")).encode()
        info = tarfile.TarInfo("archive_index.json")
        info.size = len(contents)
        tar.addfile(info, io.BytesIO(contents))
    # A sequential full readback checks every stored byte before any deletion.
    expected = {"buffer/"+r["path"]: r for r in before}
    seen = set()
    with tarfile.open(temporary, "r|gz") as tar:
        for member in tar:
            if member.name not in expected:
                continue
            require(member.isfile(), "archive file type changed")
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    h.update(chunk)
            require(member.size == expected[member.name]["bytes"]
                    and h.hexdigest() == expected[member.name]["sha256"], "archive verification failed")
            seen.add(member.name)
    require(seen == set(expected), "archive misses files")
    after, after_dirs = inventory(target)
    stripped = [{k: v for k, v in r.items() if k != "sha256"} for r in before]
    require(after == stripped and set(after_dirs) == set(directories), "source changed during archive; do not remove")
    temporary.rename(destination)
    receipt = {"status": "archive_verified_before_prune", "original_target": str(target),
        "archive": str(destination), "archive_sha256": digest(destination), "archive_bytes": destination.stat().st_size,
        "source_files": len(before), "source_directories": len(directories), "source_bytes": original_bytes,
        "all_files_readback_verified": True, "source_unchanged": True, "plan_sha256": expected_sha,
        "restore": f"mkdir -p '{target}'; tar -xzf '{destination}' -C '{target}' --strip-components=1 buffer",
        "deleted_files": 0, "freed_scratch_bytes": 0}
    write(receipt_path, receipt)
    # Exact validated directory, no glob or broad environment variable target.
    # The unchanged original is now recoverable from the checked cold archive.
    shutil.rmtree(target)
    require(not target.exists(), "scratch cleanup incomplete")
    receipt.update(status="archived_and_pruned", deleted_files=len(before), freed_scratch_bytes=original_bytes)
    write(receipt_path, receipt)
    marker = target.with_name(target.name + ".ARCHIVED_20260909.json")
    require(not marker.exists(), "archive pointer already exists")
    write(marker, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def migrate(receipts):
    """Login-node storage transfer; no model execution or navigation changes."""
    require(FINAL_COLD.parent.is_dir(), "cold storage is not mounted on this node")
    FINAL_COLD.mkdir(exist_ok=True)
    for receipt_path in sorted(receipts.glob("*.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt["status"] != "archived_and_pruned":
            continue
        source = Path(receipt["archive"])
        if source.parent == FINAL_COLD:
            continue
        require(source.parent == COLD and not source.is_symlink(), "unexpected archive source")
        destination = FINAL_COLD / source.name
        require(source.is_file(), "verified staging archive missing")
        temporary = destination.with_name(destination.name+".partial")
        if not destination.exists():
            require(not temporary.exists(), "interrupted cold transfer requires inspection")
            with source.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=8 << 20)
                dst.flush()
                os.fsync(dst.fileno())
            require(digest(temporary) == receipt["archive_sha256"], "cold-copy hash mismatch")
            temporary.rename(destination)
        require(digest(destination) == receipt["archive_sha256"], "cold archive changed")
        require(source.stat().st_size == receipt["archive_bytes"], "staging size changed")
        target = Path(receipt["original_target"])
        receipt.update(archive=str(destination), previous_scratch_archive=str(source), cold_transfer_verified=True,
            restore=f"mkdir -p '{target}'; tar -xzf '{destination}' -C '{target}' --strip-components=1 buffer")
        write(receipt_path, receipt)
        write(target.with_name(target.name+".ARCHIVED_20260909.json"), receipt)
        # Only the exact verified duplicate tar is removed here, never a directory.
        source.unlink()
        print(f"COLD_MOVED {destination.name} bytes={receipt['archive_bytes']} source_files={receipt['source_files']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("discover", "execute", "migrate"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args()
    require(os.environ.get("USER") == "yz11502", "wrong account")
    if args.mode == "discover":
        discover(args.plan)
    elif args.mode == "migrate":
        migrate(args.receipts)
    else:
        archive_one(args.plan, args.plan_sha, args.index, args.receipts)
