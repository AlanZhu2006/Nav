"""Exact source-carrier and new role-free CLI checks, before model loading."""
import argparse
import os
from pathlib import Path
import subprocess

from MemNavData.table2_mixed_hpc import ROOT, validate_plan, base_source, load, dump, sha, runtime_provenance, protocol_for, formal
from MemNavData.table2_mixed_local import query_command, runtime_spec
from MemNavData.run_repaired_fullmono_local import execution_environment
from MemNavData.table2_sampling_profiles import verify_query_direction


def audit(plan_path, out):
    plan = validate_plan(load(plan_path))
    assert plan["protocol_sha256"] == sha(protocol_for(plan))
    provenance = runtime_provenance(plan)
    out.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    Image.new("RGB", (8, 8), (128,128,128)).save(out/"dummy_cli_only.jpg")
    checked, specs = {}, []
    if formal(plan):
        payload = Path(plan["payload_root"])
        assert sha(payload/"PAYLOAD.sha256") == plan["payload_receipt_sha256"]
        subprocess.run(["sha256sum", "-c", "--quiet", "PAYLOAD.sha256"], cwd=payload, check=True)
    for s in plan["sources"]:
        for path, digest in s["source_files"].items():
            if path not in checked:
                checked[path] = sha(path)
            assert checked[path] == digest, path
        spec = base_source(s, s["seed_rank"])
        assert spec["seed"] == s["seed"] and spec["prefix_root"] is None
        specs.append(spec)
        if formal(plan) and s["a_selection"]:
            chosen = s["a_selection"]
            assert sha(chosen["query"]) == chosen["query_sha256"]
            q = load(chosen["query"])
            verify_query_direction(q, plan["schema"])
            assert sha(q["goal_rgb"]) == q["goal_rgb_sha256"]
    commands = []
    # Dummy image is ONLY for --contract_dry_run; never enters any navigation.
    for stage in range(3):
        query = dict(specs[0], stage_number=stage, floor_position=[2.,0.,0.], yaw_rad=0., geodesic_m=2.,
                     goal_rgb=str((out/"dummy_cli_only.jpg").resolve()), goal_rgb_sha256=sha(out/"dummy_cli_only.jpg"))
        qpath = out/f"runtime_stage{stage}.json"
        dump(qpath, runtime_spec(query))
        for arm in (("native",) if stage == 0 else ("native", "cec")):
            cmd = query_command(plan["sources"][0], query, out/"no_navigation_output", 21120, 21121, arm)
            cmd.append("--contract_dry_run")
            env = dict(execution_environment(), TABLE2_QUERY=str(qpath))
            with (out/f"cli_{stage}_{arm}.log").open("x") as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            commands.append(dict(stage=stage, arm=arm, command=cmd))
    assert not (out/"no_navigation_output").exists()
    dump(out/"verification.json", dict(verified=True, plan_sha256=sha(plan_path),
         **provenance, sources=len(specs), checked_files=len(checked),
         commands=commands, gpu_inference_performed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    audit(a.plan.resolve(), a.out.resolve())
