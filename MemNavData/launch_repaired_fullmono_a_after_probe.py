"""Submit the preflighted first scene only after the visual construction gate.

This is a continuation of the frozen plan, not an outcome-based retry.  A
failed or incomplete construction probe submits nothing and keeps its output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROBE_SHA = "b9297569ff0805d7b79c31b66dec184f83dfc40b29284ee3ada7a83258441063"
COLLECTOR_SHA = "3e90959723d4b66b1e18f5340ca0e7b978aaa8fb5445e3a348441c260da835ec"
PLAN_SHA = "ba1a12273bf9eeef5024166c4bfa28037ee6076c695d2543959a7e169adafba7"
EXPECTED_SCENES = ("rJhMRvNn4DS", "jgPBycuV1Jq")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_gate(summary, payloads):
    require(summary["completed"] and summary["navigation_rollouts"] == 0
            and summary["query_outcomes_read"] is False, "Not a completed construction-only probe")
    require(summary["scene_count"] == summary["pair_count"] == 2, "Both declared pairs are required")
    require(tuple(p["scene"] for p in payloads) == EXPECTED_SCENES, "Probe identity/order changed")
    for payload in payloads:
        require(payload["episode"] == "episode_0000", "Probe task changed")
        require(payload["pair_constructible"] and payload["navigation_rollouts"] == 0
                and payload["query_outcomes_read"] is False, "Pair unavailable or query outcomes read")
        require(payload["all_historical_rgb_rerender_hashes_match"], "Factual RGB binding failed")
        chosen = payload["selected_novel_direction"]
        available = payload["novel_by_direction"]
        require(chosen in available and available[chosen]["constructible"], "No chosen legal Novel")
        queries = payload["queries"]
        revisit = [q for q in queries if q["analysis_role"] == "revisit"]
        novel = [q for q in queries if q["query_id"] == available[chosen]["query_id"]]
        require(len(revisit) == len(novel) == 1, "Expected one Revisit and one selected Novel")
        require(novel[0]["analysis_role"] == "novel" and novel[0]["max_online_a_covis"] < .10,
                "Novel support bound changed")
        require(.55 <= revisit[0]["max_online_a_covis"] <= .90, "Revisit support bound changed")
        require(all(2. <= q["geodesic_from_a_end_m"] <= 9. for q in revisit + novel),
                "Query distance bound changed")


def verify_probe(root, addon):
    require(sha(addon / "SOURCE_BUNDLE.sha256") == PROBE_SHA, "Probe addon changed")
    subprocess.run(["sha256sum", "-c", "--quiet", "SOURCE_BUNDLE.sha256"], cwd=addon, check=True)
    summary = read(root / "summary.json")
    require(summary["protocol_sha256"] == sha(addon / "REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_PROTOCOL_20260909.md"),
            "Construction protocol binding changed")
    payloads = []
    for row in summary["reports"]:
        path = Path(row["path"])
        require(path.resolve().is_relative_to(root.resolve()), "Construction escaped its declared root")
        require(sha(path) == row["sha256"], "Construction receipt changed")
        payload = read(path)
        for q in payload["queries"]:
            for key in ("goal_rgb", "goal_depth"):
                asset = (path.parent / q[key]).resolve()
                require(asset.is_relative_to(path.parent.resolve()), "Goal asset path escaped its receipt")
                require(sha(asset) == q[key + "_sha256"], "Goal asset changed")
        payloads.append(payload)
    validate_gate(summary, payloads)
    return {"summary": str(root / "summary.json"), "summary_sha256": sha(root / "summary.json"),
            "scene_count": 2, "pair_count": 2, "query_rollouts": 0}


def launch(args):
    args.out.mkdir(parents=True, exist_ok=False)
    gate = verify_probe(args.probe, args.probe_addon)
    require(sha(args.collector / "SOURCE_BUNDLE.sha256") == COLLECTOR_SHA, "Collector changed")
    subprocess.run(["sha256sum", "-c", "--quiet", "SOURCE_BUNDLE.sha256"], cwd=args.collector, check=True)
    plan = args.collector / "core_source_plan.json"
    require(sha(plan) == PLAN_SHA, "Source plan changed")
    first = read(plan)["sources"][0]
    require(first["scene_rank"] == 0 and first["scene"] == EXPECTED_SCENES[0], "First scene changed")
    env = dict(os.environ, CORE_ADDON=str(args.collector), CORE_PLAN=str(plan), CORE_PLAN_SHA=PLAN_SHA,
               EXPECTED_CORE_ADDON_SHA=COLLECTOR_SHA, CORE_RUN=str(args.run_root))
    template = args.collector / "slurm_repaired_fullmono_a_collection.sbatch"
    command = ["bash", "-c", 'source "$1"; shift; safe_sbatch --lint-fatal "$@"', "safe-submit",
               str(args.collector / "slurm_safe_submit.sh"), "--parsable", "--partition=a100_tandon",
               "--array=0", "--export=ALL", str(template)]
    with (args.out / "request.json").open("x") as f:
        json.dump({"gate": gate, "command": command, "scope": "first-scene four-A execution gate",
                   "source_plan_sha256": PLAN_SHA, "query_job_submitted": False}, f, indent=2)
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    (args.out / "sbatch.stdout").write_text(completed.stdout)
    (args.out / "sbatch.stderr").write_text(completed.stderr)
    completed.check_returncode()
    job = completed.stdout.strip().split(";")[0]
    require(job.isdigit(), "Inspect saved sbatch output before any further submission")
    result = {"submitted": True, "job_id": int(job), "array": "0", "source_count": 4,
              "run_root": str(args.run_root), "source_plan_sha256": PLAN_SHA,
              "collector_receipt_sha256": COLLECTOR_SHA, "construction_gate": gate,
              "formal_expansion_submitted": False, "query_job_submitted": False}
    with (args.out / "submission.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "probe-addon", "collector", "run-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    launch(p.parse_args())
