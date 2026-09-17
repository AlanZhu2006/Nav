"""Check all frozen Table-I inputs before requesting GPU evaluation."""
import argparse
import os
from pathlib import Path
import subprocess

from table1_repaired_eval import ROOT, CONTROLLERS, command, dump, history, load, sha
from run_repaired_fullmono_local import execution_environment, HAB_PY


def audit(plan_path, out):
    out.mkdir(parents=True, exist_ok=False)
    plan = load(plan_path)
    assert not plan["local"] and len(plan["cells"]) == 210 and plan["total_rollouts"] == 840
    assert plan["frozen_protocol_sha256"] == sha(ROOT / "MemNavData/TABLE1_REPAIRED_THREE_CONTROLLER_PROTOCOL_20260910.md")
    checked, evidence = {}, []

    def check(path, expected):
        path = Path(path)
        if path not in checked:
            checked[path] = sha(path)
        assert checked[path] == expected, str(path)

    for cell in (c for c in plan["cells"] if c["controller"] == "navdp"):
        payload, frozen, folder = history(cell)
        receipt, source = frozen["receipt"], frozen["source"]
        check(receipt["source_asset"], receipt["source_asset_sha256"])
        check(Path(receipt["source_episode"]) / "data/chunk-000/episode_000000.parquet", receipt["source_parquet_sha256"])
        images = sorted((source / "rgb").glob("*.jpg"))
        assert len(images) == len(receipt["rgb_frame_hashes"]) == len(frozen["trace"]["poses"])
        for image, digest in zip(images, receipt["rgb_frame_hashes"]):
            check(image, digest)
        for pair in payload["pairs"]:
            for query in pair["queries"]:
                check(folder / query["goal_rgb"], query["goal_rgb_sha256"])
        evidence.append(dict(dataset=cell["dataset"], scene=cell["scene"], episode=cell["episode"],
                             images=len(images)))
    env = execution_environment()
    gates = [0, 28, 56, 84, 126, 168]
    for i in gates:
        cell = plan["cells"][i]
        assert cell["history_index"] == 0
        for role in ("novel", "revisit"):
            for arm in ("native", "cec"):
                cmd = command(cell, role, arm, out / "cli_no_output", 21120, 21121) + ["--contract_dry_run"]
                with (out / f"cli_{i}_{role}_{arm}.log").open("x") as log:
                    subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Exact remote interpreter, actual weights, HTTP route and mask/seed semantics.
    payload, frozen, folder = history(plan["cells"][0])
    goal = folder / payload["pairs"][0]["queries"][0]["goal_rgb"]
    for controller in ("vint", "nomad"):
        model_env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/InternNav/src/diffusion-policy")
        cmd = [os.environ["TABLE1_IMAGE_PY"], str(ROOT / "MemNavData/audit_image_controller_cpu.py"),
               "--controller", controller, "--checkpoint", str(Path(os.environ["TABLE1_CHECKPOINTS"]) / f"{controller}.pth"),
               "--history", str(frozen["source"] / "rgb"), "--goal", str(goal), "--out", str(out / controller)]
        with (out / f"{controller}_cpu.log").open("x") as log:
            subprocess.run(cmd, cwd=ROOT, env=model_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert load(out / controller / "verification.json")["verified"]
    dump(out / "verification.json", dict(verified=True, plan_sha256=sha(plan_path),
         runtime_sha256=sha(ROOT / "SOURCE_BUNDLE.sha256"), histories=evidence,
         unique_checked_files=len(checked), cli_combinations=24, gpu_gate_indices=gates,
         image_model_checks={c:load(out / c / "verification.json") for c in ("vint", "nomad")}))
    print("VERIFIED all 70 histories, 24 CLI contracts and both RGB checkpoints", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.plan.resolve(), args.out.resolve())
