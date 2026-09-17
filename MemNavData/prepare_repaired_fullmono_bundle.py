"""Build a standalone source-only gate bundle; no checkpoints or experiment data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".toml"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=False)
    files = set((ROOT / "MemNavData").glob("*.py"))
    for relative in ("NavDP/baselines/navdp", "NavDP/baselines/memnav", "InternNav/internnav",
                     "InternNav/scripts/train/configs", "InternNav/src/diffusion-policy"):
        for path in (ROOT / relative).rglob("*"):
            if path.suffix in EXTENSIONS and not any(p.startswith(".") or p == "__pycache__"
                                                    for p in path.relative_to(ROOT).parts):
                files.add(path)
    for name in ("run_habitat_contract_preflight.sh", "slurm_port_pair.sh", "slurm_safe_submit.sh",
                 "run_repaired_hm3d_gate.sh", "slurm_repaired_hm3d_gate.sbatch", "repaired_hm3d_hpc_env.sh",
                 "REPAIRED_FULLMONO_LOCAL_PROTOCOL_20260908.md", "REPAIRED_HM3D_GATE_PROTOCOL_20260908.md",
                 "REPAIRED_HM3D_GATE_EXTENSION_PROTOCOL_20260908.md", "slurm_repaired_hm3d_gate_extension.sbatch",
                 "run_coverage_ablation_hpc.sh", "slurm_coverage_ablation_eval.sbatch",
                 "slurm_coverage_ablation_summary.sbatch", "COVERAGE_ABLATION_HPC_PROTOCOL_20260910.md",
                 "strict_graph_blind_20260806.json", "hm3d_fresh_fullmono_mixed_role_protocol_20260820.json"):
        files.add(ROOT / "MemNavData" / name)
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing physical source: {path}")
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(json.dumps({"staging": str(destination), "files": len(files),
                      "bytes": sum(p.stat().st_size for p in files)}))


def seal(destination):
    receipt = destination / "SOURCE_BUNDLE.sha256"
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    if receipt.exists() or any(p.is_symlink() or "__pycache__" in p.parts for p in files):
        raise ValueError("Already sealed bundle, symlink, or bytecode contamination")
    with receipt.open("x") as stream:
        for path in files:
            stream.write(f"{digest(path)}  {path.relative_to(destination).as_posix()}\n")
    archive = destination.parent / f"repaired_fullmono_{digest(receipt)[:16]}.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for path in files + [receipt]:
            tar.add(path, arcname=path.relative_to(destination), recursive=False)
    for path in files + [receipt]:
        path.chmod(path.stat().st_mode & ~0o222)
    for path in sorted(destination.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    print(json.dumps({"source_receipt_sha256": digest(receipt), "archive": str(archive),
                      "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "seal"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else seal)(args.destination.resolve())
