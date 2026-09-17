"""Five paired NavDP controls on metadata-selected original paper histories.

The paper geometry and strict certificate are fixed in every arm. Intervention
hooks reuse the independently checked local pilot; model code is unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MemNavData")]
from MemNavData.gem_bearing_attribution import dump, load, sha

HERE = Path(__file__).resolve()
ARMS = ["native", "fixed_half_turn", "gem_no_turn", "initial_turn_only", "full_gem"]


def check_plan(path):
    plan = load(path)
    assert plan["schema"] in ("gem_reviewer_paper_controls_v1", "gem_reviewer_paper_controls_v2")
    assert plan["arms"] == ARMS
    if plan["schema"] == "gem_reviewer_paper_controls_v1":
        assert len(plan["cells"]) == 13 and plan["total_rollouts"] == 65
    else:
        assert len(plan["cells"]) == 57 and plan["total_rollouts"] == 285
        completed = set(plan["completed_parent_paper_indices"])
        additional = {c["paper_index"] for c in plan["cells"]}
        assert len(completed) == 13 and len(additional) == 57
        population = set(plan["population_paper_indices"])
        assert len(population) == 70 and not completed & additional
        assert completed | additional == population
        assert plan["dense_window"] == 32 and plan["success_radius_m"] == 1.0
        assert plan["max_steps"] == 600 and plan["exec_horizon"] == 8
    assert [c["index"] for c in plan["cells"]] == list(range(len(plan["cells"])))
    identities = {(c["dataset"], c["scene"], c["episode"]) for c in plan["cells"]}
    assert len(identities) == len(plan["cells"])
    assert plan["memory_mode"] == "legacy" and plan["authority_policy"] == "strict_certificate"
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    return plan


def evaluate():
    arm = os.environ["GEM_PAPER_ARM"]
    if arm in ("full_gem", "initial_turn_only"):
        os.environ["GEM_BEARING_ARM"] = arm
        from MemNavData.gem_bearing_attribution import evaluate as execute
    else:
        assert arm in ARMS
        os.environ["GEM_REVIEWER_ARM"] = arm
        from MemNavData.gem_reviewer_attribution import evaluate as execute
    execute()


def query_command(cell, arm, target, mem, nav):
    from MemNavData.table1_repaired_eval import command
    cmd = command(cell, "revisit", "native" if arm in ARMS[:2] else "cec", target, mem, nav)
    cmd[2] = str(HERE)
    # Existing evaluator defaults are precisely the Table I strict certificate.
    return cmd


def run(args):
    from MemNavData.run_gem_memory_navigation import servers
    from MemNavData.run_repaired_fullmono_local import execution_environment, run_child
    from MemNavData.run_gem_support_navigation import gpu_binding
    from MemNavData.table1_repaired_eval import history
    plan = check_plan(args.plan)
    cell = plan["cells"][args.index]
    assert cell["index"] == args.index
    payload, frozen, folder = history(cell)
    query = next(q for p in payload["pairs"] for q in p["queries"] if q["analysis_role"] == "revisit")
    assert query["goal_rgb_sha256"] == cell["goal_rgb_sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "logs").mkdir()
    inputs = [folder / "role_pairs.json", folder / query["goal_rgb"],
              frozen["source"] / "receipt.json", frozen["source"] / "online_a_trace.json"]
    shift = args.index % len(ARMS)
    order = ARMS[shift:] + ARMS[:shift]
    dump(args.out / "manifest.json", dict(cell=cell, arms_in_order=order,
         plan=str(args.plan), plan_sha256=sha(args.plan),
         source_sha256=plan["source_sha256"], input_sha256={str(p): sha(p) for p in inputs}))
    env = dict(execution_environment(), TABLE1_PLAN=str(args.plan), TABLE1_INDEX=str(args.index))
    summary = dict(completed=False, cell=cell, records=[], plan_sha256=sha(args.plan))
    dump(args.out / "summary.json", summary)
    start = time.monotonic()
    try:
        for arm in order:
            run_child(query_command(cell, arm, args.out / "dry" / arm, args.mem_port, args.nav_port)
                      + ["--contract_dry_run"], args.out / "logs" / (arm + "_cli.log"),
                      environment=dict(env, GEM_PAPER_ARM=arm))
        with servers(args.out, "legacy", args.mem_port, args.nav_port, dense_window=32):
            pose_log = args.out / "lingbot_pose_readout.jsonl"
            for arm in order:
                target = args.out / "evaluation" / arm
                dump(args.out / "progress.json", dict(arm=arm, completed=len(summary["records"]),
                     total=5, updated_unix=time.time()))
                offset = pose_log.stat().st_size if pose_log.exists() else 0
                print("START", args.index, cell["scene"], arm, flush=True)
                seconds = run_child(query_command(cell, arm, target, args.mem_port, args.nav_port),
                     args.out / "logs" / (arm + ".log"), environment=dict(env, GEM_PAPER_ARM=arm))
                with pose_log.open() as stream:
                    stream.seek(offset)
                    dump(target / "lingbot_frame_poses.json", [json.loads(l) for l in stream if l.strip()])
                terminal = load(target / "terminal_measurements.json")
                assert len(terminal) == 1
                row = dict(terminal[0], directory=str(target), scene=cell["scene"], episode=cell["episode"],
                           role="revisit", arm=arm, wall_seconds=seconds)
                summary["records"].append(row)
                dump(args.out / "summary.json", summary)
                print("DONE", args.index, arm, row["reached"], row["steps"], flush=True)
            dump(args.out / "paired_gpu_binding.json", gpu_binding(args.out))
        check_plan(args.plan)
        summary["completed"] = True
        summary["elapsed_seconds"] = time.monotonic() - start
        dump(args.out / "summary.json", summary)
        run_child([os.environ["REPAIRED_HAB_PY"], "-u", str(HERE), "verify",
                   "--plan", str(args.plan), "--index", str(args.index), "--out", str(args.out)],
                  args.out / "logs" / "independent_verify.log", environment=env)
        assert load(args.out / "independent_verification.json")["verified"]
        dump(args.out / "completion.json", dict(completed=True, index=args.index,
             elapsed_seconds=time.monotonic()-start, summary_sha256=sha(args.out / "summary.json"),
             independent_verification_sha256=sha(args.out / "independent_verification.json"),
             plan_sha256=sha(args.plan), gpu=load(args.out / "paired_gpu_binding.json")))
    except BaseException as error:
        dump(args.out / "failure.json", dict(type=type(error).__name__, error=str(error),
             elapsed_seconds=time.monotonic()-start))
        raise


def verify(args):
    import numpy as np
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_gem_reviewer_attribution import fixed_scan
    from MemNavData.verify_gem_bearing_attribution import non_timing
    from MemNavData.verify_habitat_minimal_repair import read_rows
    plan = check_plan(args.plan)
    summary, manifest = load(args.out / "summary.json"), load(args.out / "manifest.json")
    assert summary["completed"] and len(summary["records"]) == 5
    assert summary["plan_sha256"] == manifest["plan_sha256"] == sha(args.plan)
    assert summary["cell"] == manifest["cell"] == plan["cells"][args.index]
    for name, digest in manifest["input_sha256"].items():
        assert sha(name) == digest, name
    indexed, checks = {}, []
    for row in summary["records"]:
        arm, folder = row["arm"], Path(row["directory"])
        audit, evidence, plans, actions = fixed_scan(row) if arm == "fixed_half_turn" else verify_rollout(row)
        bearing_arm = arm in ("full_gem", "initial_turn_only")
        decisions = read_rows(folder / ("bearing_decisions.jsonl" if bearing_arm else "intervention_decisions.jsonl"))
        boundary, http = read_rows(folder / "memory_http_boundary.jsonl"), read_rows(folder / "navdp_http_receipts.jsonl")
        query_plans = load(folder / "query_plans.json")["query_leg"]
        assert len(decisions) == len(plans)
        for d, p in zip(decisions, plans):
            assert d["goal_sha256"] == summary["cell"]["goal_rgb_sha256"]
            assert d["image_sha256"] == p["receipt"]["execution_input_audit"]["image_jpeg_sha256"]
            assert d["diffusion_seed_requested"] == d["diffusion_seed_returned"]
        forbidden = {"executed_translation_m", "executed_yaw_rad", "executed_forward_m", "executed_left_m",
                     "executor_local_se2_source", "executor_local_se2_contract"}
        assert not any(forbidden & set(r["sent_fields"]) for r in boundary)
        if arm in ARMS[:2]:
            assert all(not p["receipt"]["revisit_adapter_takeover"] and p["receipt"]["memory_controller_pointgoal"] is None for p in plans)
            assert not any(r["path"] in ("/retrieval_probe_step", "/certified_relocalize") for r in boundary)
            assert all(r["path"] in ("/imagegoal_step", "/memory_replay_step") for r in http if r["query_active"])
        if arm == "gem_no_turn":
            assert audit["heading"]["turn_actions"] == 0
        intervention = load(folder / "bearing_intervention.json") if bearing_arm else None
        if arm == "initial_turn_only":
            cutoff = intervention["cutoff_action"]
            assert cutoff is not None and intervention["phase"] == "native"
            assert intervention["completed_turns"] == int(intervention["initial_rearward"])
            later = [p for p in plans if p["next_action_index"] >= cutoff]
            assert later and all(not p["receipt"]["revisit_adapter_takeover"] and p["receipt"]["memory_controller_pointgoal"] is None for p in later)
            assert all(a["action_kind"] != "heading" for a in actions[cutoff:])
            calls = [r for r in http if r["query_active"] and r["next_action_index"] >= cutoff]
            assert calls and all(r["path"] == "/imagegoal_step" for r in calls)
            calls = [r for r in boundary if r["query_active"] and r["next_action_index"] >= cutoff]
            # The first decision at cutoff zero is allowed to reject initial recall.
            if cutoff:
                assert not any(r["path"] in ("/certified_relocalize", "/retrieval_probe_step") for r in calls)
        elif arm == "full_gem":
            assert not any(d["native_after_cutoff"] for d in decisions)
        assert arm not in indexed
        indexed[arm] = dict(row=row, plans=plans, actions=actions, intervention=intervention,
                            query_plans=query_plans, poses=load(folder / "lingbot_frame_poses.json"))
        checks.append(audit)
    assert set(indexed) == set(ARMS)
    native, scan = indexed["native"], indexed["fixed_half_turn"]
    for item in indexed.values():
        for field in ("first_query_rgb_sha256", "goal_xz_evaluator_only", "geodesic_m"):
            assert item["row"][field] == native["row"][field]
        assert item["plans"][0]["position"] == native["plans"][0]["position"]
        assert item["plans"][0]["yaw"] == native["plans"][0]["yaw"]
    for field in ("selected_trajectory", "all_trajectory", "all_values"):
        np.testing.assert_array_equal(native["plans"][0][field], scan["plans"][0][field])
    full, initial = indexed["full_gem"], indexed["initial_turn_only"]
    for other in (initial, indexed["gem_no_turn"]):
        for field in ("certified_relocalization_accepted", "certified_relocalization_certificate",
                      "certified_relocalization_pnp", "router_selected_anchor"):
            assert non_timing(full["query_plans"][0][field]) == non_timing(other["query_plans"][0][field]), field
    cutoff = initial["intervention"]["cutoff_action"]
    if initial["intervention"]["initial_rearward"]:
        assert full["intervention"]["cutoff_action"] == cutoff
        assert full["actions"][:cutoff] == initial["actions"][:cutoff]
        first = [next(p for p in v["plans"] if p["next_action_index"] == cutoff) for v in (full, initial)]
        for field in ("position", "yaw"):
            assert first[0][field] == first[1][field]
        for field in ("memory_frame_idx", "diffusion_seed"):
            assert first[0]["receipt"][field] == first[1]["receipt"][field]
        k = first[0]["receipt"]["memory_frame_idx"]
        for a, b in zip(full["poses"][:k+1], initial["poses"][:k+1]):
            for field in ("frame_idx", "image_sha256", "camera_pose9", "pose_count", "motion_receipt_recorded"):
                assert a[field] == b[field], (field, a["frame_idx"])
        for field in ("input_tensor_sha256", "output_tensor_sha256", "contract"):
            assert first[0]["receipt"]["monocular_depth_receipt"]["navdp_depth_raster"][field] == first[1]["receipt"]["monocular_depth_receipt"]["navdp_depth_raster"][field]
    dump(args.out / "independent_verification.json", dict(verified=True, checks=checks,
         initial_sample_identical=True, first_localization_identical=True,
         initial_guided_prefix_identical=True, plan_sha256=sha(args.plan),
         summary_sha256=sha(args.out / "summary.json"), verifier_sha256=sha(HERE)))
    print("VERIFIED", args.index, "five paired arms", flush=True)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "eval":
        evaluate()
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--index", type=int, required=True)
        p.add_argument("--mem-port", type=int, default=21930)
        p.add_argument("--nav-port", type=int, default=21931)
        args = p.parse_args()
        args.out, args.plan = args.out.resolve(), args.plan.resolve()
        if action == "run":
            run(args)
        elif action == "verify":
            verify(args)
        else:
            p.error("Expected eval, run, or verify")
