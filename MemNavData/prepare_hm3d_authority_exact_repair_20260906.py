#!/usr/bin/env python3
"""Package the two audited infrastructure fixes without touching old outputs.

Run on the verified yz11502 login with --inputs pointing to the uploaded
repair files. Preparation only: this script does not submit a Slurm job.
"""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile


ROOT = Path("/scratch/yz11502/Research")
BUNDLES = ROOT / "Nav-axis-uturn-source-bundles"
RESULTS = ROOT / "Nav-axis-uturn-results"
TASK = BUNDLES / "hm3d_table1_authority_spectrum_9bb6fc2fcc16303b"
SERVER = BUNDLES / "hm3d_table1_navdp_authority_transaction_718661db1733d5de"
BASE = BUNDLES / "final14_mono_factorial_5690569a4373f2d2"
CLOSURE = BUNDLES / "hm3d_table1_navdp_authority_transaction_repair_51ee9a4ca063c7f1"
OLD_RUN = RESULTS / "hm3d_table1_authority_spectrum_20260904/formal_9bb6fc2fcc16303b"
NEW_RUN = RESULTS / "hm3d_table1_authority_spectrum_repair_20260906"
CONSTRUCTION = RESULTS / ("hm3d_table1_fresh_query_reserve_20260829/"
                          "construction_20260828T212552Z_bb757914")
BENCH = CONSTRUCTION / "population/natural_direction"
PARENT = RESULTS / ("hm3d_fresh_fullmono_mixed_role_20260820/"
                    "formal_20260820T143609Z_e6dd44c6/sealed_inputs/parent_manifest.json")
RETRY = [2, 3, 6, 7, 10, 13, 18, 21]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle(root):
    for line in (root / "SOURCE_BUNDLE.sha256").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        assert sha(root / relative) == digest, relative


def publish(old, replacements, name):
    verify_bundle(old)
    stage = Path(tempfile.mkdtemp(prefix=name + ".staging.", dir=BUNDLES))
    shutil.copytree(old, stage, dirs_exist_ok=True)
    # Only the private copy becomes writable. Original frozen files stay intact.
    for directory, _, files in os.walk(stage):
        Path(directory).chmod(0o755)
        for filename in files:
            (Path(directory) / filename).chmod(0o644)
    changes = []
    for relative, source in replacements.items():
        target = stage / relative
        before = target.read_text() if target.exists() else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        changes.extend(difflib.unified_diff(
            before.splitlines(True), target.read_text().splitlines(True),
            fromfile=str(old / relative), tofile=relative))
    (stage / "infrastructure_repair.diff").write_text("".join(changes))
    (stage / "repair_parent.json").write_text(json.dumps({
        "parent": str(old), "parent_receipt_sha256": sha(old / "SOURCE_BUNDLE.sha256"),
        "replaced_or_added_files": sorted(replacements),
        "scientific_protocol_changed": False,
    }, indent=2) + "\n")
    receipt = stage / "SOURCE_BUNDLE.sha256"
    lines = [f"{sha(path)}  ./{path.relative_to(stage)}\n"
             for path in sorted(stage.rglob("*"))
             if path.is_file() and path != receipt]
    receipt.write_text("".join(lines))
    digest = sha(receipt)
    target = BUNDLES / (name + "_" + digest[:16])
    assert not target.exists(), target
    for directory, _, files in os.walk(stage):
        for filename in files:
            (Path(directory) / filename).chmod(0o444)
        Path(directory).chmod(0o555)
    stage.rename(target)
    return target, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("USER") == "yz11502"
    assert not NEW_RUN.exists(), "repair root already exists"
    manifest = json.loads((BENCH / "manifest.json").read_text())
    assert sha(BENCH / "manifest.json") == (
        "f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01")
    assert len(manifest["episodes"]) == 28
    missing, reuse = [], []
    for index, item in enumerate(manifest["episodes"]):
        label = f'{index:03d}_{item["scene"]}_{item["episode"]}'
        source = OLD_RUN / "formal/evaluation/natural_direction" / label
        completion = source / "completion.json"
        sidecar = source / "completion.json.sha256"
        if completion.is_file() and sidecar.is_file():
            digest = sha(completion)
            assert sidecar.read_text().split()[0] == digest
            reuse.append({"index": index, "source": str(source),
                          "completion_sha256": digest})
        else:
            missing.append(index)
    assert missing == RETRY, missing
    assert len(reuse) == 20
    server, server_sha = publish(SERVER, {
        "NavDP/baselines/navdp/navdp_server.py": args.inputs / "navdp_server.py",
    }, "hm3d_authority_stationary_cache_repair")
    names = ["slurm_hm3d_table1_authority_spectrum.sbatch",
             "test_hm3d_table1_authority_spectrum.py",
             "test_navdp_monocular_transaction.py", "slurm_safe_submit.sh",
             "HM3D_AUTHORITY_EXACT_REPAIR_PROTOCOL_20260906.md"]
    task, task_sha = publish(TASK, {
        "MemNavData/" + name: args.inputs / name for name in names
    }, "hm3d_authority_exact_repair")
    evaluation = NEW_RUN / "formal/evaluation/natural_direction"
    evaluation.mkdir(parents=True)
    for item in reuse:
        source = Path(item["source"])
        (evaluation / source.name).symlink_to(source, target_is_directory=True)
    values = {
        "TASK_ROOT": str(task), "TASK_RECEIPT": str(task / "SOURCE_BUNDLE.sha256"),
        "EXPECTED_TASK_RECEIPT_SHA": task_sha,
        "BASE_SOURCE_ROOT": str(BASE), "BASE_RECEIPT": str(BASE / "source_inputs.sha256"),
        "EXPECTED_BASE_RECEIPT_SHA": sha(BASE / "source_inputs.sha256"),
        "SERVER_SOURCE_ROOT": str(server),
        "SERVER_SOURCE_RECEIPT": str(server / "SOURCE_BUNDLE.sha256"),
        "EXPECTED_SERVER_SOURCE_RECEIPT_SHA": server_sha,
        "RUNTIME_CLOSURE_ROOT": str(CLOSURE),
        "RUNTIME_CLOSURE_RECEIPT": str(CLOSURE / "SOURCE_BUNDLE.sha256"),
        "EXPECTED_RUNTIME_CLOSURE_RECEIPT_SHA": sha(CLOSURE / "SOURCE_BUNDLE.sha256"),
        "BENCH_ROOT": str(BENCH), "PARENT_MANIFEST": str(PARENT),
        "FORMAL_RUN_ROOT": str(NEW_RUN),
        "CONSTRUCTION_VERIFICATION": str(CONSTRUCTION / "hm3d_table1_fresh_query_verification.json"),
        "EXPECTED_CONSTRUCTION_VERIFICATION_SHA": sha(
            CONSTRUCTION / "hm3d_table1_fresh_query_verification.json"),
        "EXPECTED_MANIFEST_SHA": sha(BENCH / "manifest.json"),
    }
    receipt = {"retry_indices": RETRY, "reused_histories": reuse,
               "old_run": str(OLD_RUN), "new_run": str(NEW_RUN),
               "configuration": values, "jobs_submitted": False,
               "selection_uses_policy_outcomes": False}
    (NEW_RUN / "prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (NEW_RUN / "exports.sh").write_text("\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in values.items()) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
