"""Paired full-history/recent-decision archive access on frozen paper queries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MemNavData")]
from MemNavData.gem_bearing_attribution import dump, load, sha
from MemNavData.gem_retrieval_domain import ARMS, eligible_indices

HERE = Path(__file__).resolve()


def check_plan(path):
    plan = load(path)
    assert plan["schema"] == "gem_retrieval_domain_v1"
    assert plan["arms"] == list(ARMS) and len(plan["cells"]) == 13
    assert plan["total_rollouts"] == 26
    assert plan["memory_mode"] == "legacy" and plan["dense_window"] == 32
    assert plan["authority_policy"] == "strict_certificate"
    assert plan["max_steps"] == 600 and plan["success_radius_m"] == 1.0
    assert [c["index"] for c in plan["cells"]] == list(range(13))
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    return plan


def command(cell, target, mem, nav):
    from MemNavData.table1_repaired_eval import command as original
    cmd = original(cell, "revisit", "cec", target, mem, nav)
    cmd[2] = str(HERE)
    return cmd


def run(args):
    from MemNavData.run_gem_memory_navigation import servers
    from MemNavData.run_gem_support_navigation import gpu_binding
    from MemNavData.run_repaired_fullmono_local import execution_environment, run_child
    from MemNavData.table1_repaired_eval import history
    plan = check_plan(args.plan)
    cell = plan["cells"][args.index]
    payload, frozen, folder = history(cell)
    query = next(q for p in payload["pairs"] for q in p["queries"] if q["analysis_role"] == "revisit")
    assert query["goal_rgb_sha256"] == cell["goal_rgb_sha256"]
    decision_steps = [int(p["step"]) for p in frozen["trace"]["plans"]]
    assert decision_steps[-7:] == cell["recent_indices"]
    assert len(frozen["trace"]["poses"]) == cell["history_frames"]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "logs").mkdir()
    inputs = [folder / "role_pairs.json", folder / query["goal_rgb"],
              frozen["source"] / "receipt.json", frozen["source"] / "online_a_trace.json"]
    order = list(ARMS if args.index % 2 == 0 else reversed(ARMS))
    dump(args.out / "manifest.json", dict(cell=cell, arms_in_order=order,
         plan_sha256=sha(args.plan), input_sha256={str(p): sha(p) for p in inputs},
         source_sha256=plan["source_sha256"]))
    env = dict(execution_environment(), TABLE1_PLAN=str(args.plan), TABLE1_INDEX=str(args.index))
    goal_key = hashlib.md5((folder / query["goal_rgb"]).read_bytes()).hexdigest()
    summary = dict(completed=False, cell=cell, records=[], plan_sha256=sha(args.plan))
    dump(args.out / "summary.json", summary)
    start = time.monotonic()
    try:
        for arm in order:
            run_child(command(cell, args.out / "dry" / arm, args.mem_port, args.nav_port)
                      + ["--contract_dry_run"], args.out / "logs" / (arm + "_cli.log"), environment=env)
        with servers(args.out, "legacy", args.mem_port, args.nav_port, dense_window=32,
                     memory_server_entrypoint=ROOT / "MemNavData/gem_retrieval_domain_server.py"):
            for arm in order:
                config = dict(arm=arm, history_frames=cell["history_frames"],
                              recent_indices=cell["recent_indices"], goal_key=goal_key)
                dump(args.out / "active_domain.json", config)
                target = args.out / "evaluation" / arm
                paths = [args.out / "lingbot_pose_readout.jsonl", args.out / "retrieval_domain_audit.jsonl"]
                offsets = [p.stat().st_size if p.exists() else 0 for p in paths]
                dump(args.out / "progress.json", dict(arm=arm, completed=len(summary["records"]), total=2))
                print("START", args.index, cell["scene"], arm, flush=True)
                elapsed = run_child(command(cell, target, args.mem_port, args.nav_port),
                                    args.out / "logs" / (arm + ".log"), environment=env)
                for path, offset, filename in zip(paths, offsets, ["lingbot_frame_poses.json", "domain_audit.json"]):
                    with path.open() as stream:
                        stream.seek(offset)
                        dump(target / filename, [json.loads(line) for line in stream if line.strip()])
                dump(target / "domain_config.json", config)
                terminal = load(target / "terminal_measurements.json")
                assert len(terminal) == 1
                row = dict(terminal[0], directory=str(target), scene=cell["scene"],
                           episode=cell["episode"], role="revisit", arm=arm, wall_seconds=elapsed)
                summary["records"].append(row)
                dump(args.out / "summary.json", summary)
                print("DONE", args.index, arm, row["reached"], row["steps"], flush=True)
            dump(args.out / "paired_gpu_binding.json", gpu_binding(args.out))
        check_plan(args.plan)
        summary.update(completed=True, elapsed_seconds=time.monotonic() - start)
        dump(args.out / "summary.json", summary)
        run_child([os.environ["REPAIRED_HAB_PY"], "-u", str(HERE), "verify", "--plan", str(args.plan),
                   "--index", str(args.index), "--out", str(args.out)],
                  args.out / "logs/independent_verify.log", environment=env)
        assert load(args.out / "independent_verification.json")["verified"]
        dump(args.out / "completion.json", dict(completed=True, index=args.index,
             elapsed_seconds=time.monotonic() - start, plan_sha256=sha(args.plan),
             summary_sha256=sha(args.out / "summary.json"),
             independent_verification_sha256=sha(args.out / "independent_verification.json"),
             gpu=load(args.out / "paired_gpu_binding.json")))
    except BaseException as error:
        dump(args.out / "failure.json", dict(type=type(error).__name__, error=str(error),
             elapsed_seconds=time.monotonic() - start))
        raise


def verify(args):
    import math
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    from MemNavData.certified_relocalization_runtime import CERTIFIED_CANDIDATE_TOP_K, CERTIFIED_CANDIDATE_MIN_GAP
    plan = check_plan(args.plan)
    cell = plan["cells"][args.index]
    manifest, summary = load(args.out / "manifest.json"), load(args.out / "summary.json")
    assert summary["completed"] and len(summary["records"]) == 2
    assert summary["cell"] == manifest["cell"] == cell
    assert summary["plan_sha256"] == manifest["plan_sha256"] == sha(args.plan)
    for path, expected in manifest["input_sha256"].items():
        assert sha(path) == expected, path
    indexed, checks = {}, []
    for row in summary["records"]:
        arm, folder = row["arm"], Path(row["directory"])
        assert arm in ARMS and arm not in indexed
        audit, evidence, plans, actions = verify_rollout(row)
        config = load(folder / "domain_config.json")
        events = load(folder / "domain_audit.json")
        shortlist = [e for e in events if e["event"] == "shortlist"]
        reads = [e for e in events if e["event"] == "sparse_read"]
        assert len(shortlist) == len(reads) == len(plans) > 0
        first = shortlist[0]
        assert not first["cached"] and all(e["cached"] for e in shortlist[1:])
        assert config["arm"] == arm and config["recent_indices"] == cell["recent_indices"]
        assert first["frame_index"] == cell["history_frames"]
        permitted = eligible_indices(config, first["frame_index"], first["candidate_ceiling"])
        assert first["eligible_indices"] == permitted
        ranked = sorted((i for i in permitted if math.isfinite(first["scores"][i])),
                        key=lambda i: (-first["scores"][i], i))
        chosen = []
        for index in ranked:
            if all(abs(index - old) >= CERTIFIED_CANDIDATE_MIN_GAP for old in chosen):
                chosen.append(index)
                if len(chosen) == CERTIFIED_CANDIDATE_TOP_K:
                    break
        expected = [{"anchor": i, "score": first["scores"][i]} for i in chosen]
        assert first["candidates"] == expected
        for event in shortlist + reads:
            assert event["arm"] == arm and event["candidates"] == expected
            assert event["domain_fingerprint"] == first["domain_fingerprint"]
            assert event["goal_key"] == config["goal_key"]
        assert all(e["selected_anchor"] is None or e["selected_anchor"] in chosen for e in reads)
        query = load(folder / "query_plans.json")
        assert query["original_goal_sha256"] == cell["goal_rgb_sha256"]
        if arm == "recent_seven":
            assert query["replay"]["decision_steps"][-7:] == permitted
        assert query["replay"]["online_frames"] == cell["history_frames"]
        assert [p["frame_idx"] for p in query["replay"]["memory_trace"]] == list(range(cell["history_frames"]))
        for p, read in zip(query["query_leg"], reads):
            assert p["router_candidate_order_dino"] == chosen
            assert all(i in chosen for i in p["router_candidate_order_used"])
            assert p["router_selected_anchor"] == read["selected_anchor"]
            assert p["certified_relocalization_accepted"] == read["accepted"]
            assert p["certified_relocalization_reason"] != "certificate_endpoint_failure"
        forbidden = {"executed_translation_m", "executed_yaw_rad", "executed_forward_m", "executed_left_m",
                     "executor_local_se2_source", "executor_local_se2_contract"}
        boundary = read_rows(folder / "memory_http_boundary.jsonl")
        assert not any(forbidden & set(e["sent_fields"]) for e in boundary)
        checks.append(audit)
        indexed[arm] = dict(row=row, audit=audit, plans=plans, query=query,
                            first=first, poses=load(folder / "lingbot_frame_poses.json"))
    full, recent = indexed["full_history"], indexed["recent_seven"]
    assert full["first"]["scores"] == recent["first"]["scores"]
    h = cell["history_frames"]
    assert len(full["poses"]) > h and len(recent["poses"]) > h
    for a, b in zip(full["poses"][:h+1], recent["poses"][:h+1]):
        for field in ("frame_idx", "image_sha256", "camera_pose9", "pose_count", "motion_receipt_recorded"):
            assert a[field] == b[field], (field, a["frame_idx"])
    for field in ("first_query_rgb_sha256", "goal_xz_evaluator_only", "geodesic_m"):
        assert full["row"][field] == recent["row"][field]
    for field in ("position", "yaw"):
        assert full["plans"][0][field] == recent["plans"][0][field]
    for field in ("input_tensor_sha256", "output_tensor_sha256", "contract"):
        assert full["plans"][0]["receipt"]["monocular_depth_receipt"]["navdp_depth_raster"][field] == recent["plans"][0]["receipt"]["monocular_depth_receipt"]["navdp_depth_raster"][field]
    reference = cell["reference_full"]
    for key in ("reached", "steps", "actual_path_len_m", "spl"):
        assert full["row"][key] == reference[key], ("original full reference", key)
    assert full["audit"]["heading"]["turn_actions"] == reference["turn_actions"]
    dump(args.out / "independent_verification.json", dict(verified=True, checks=checks,
         archive_indices_verified=True, shortlist_recomputed=True, cached_domain_unchanged=True,
         full_causal_writer_prefix_identical=True, initial_depth_identical=True,
         full_reference_reproduced=True, plan_sha256=sha(args.plan), summary_sha256=sha(args.out / "summary.json")))
    print("VERIFIED", args.index, "paired archive domains", flush=True)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "eval":
        from MemNavData.run_gem_memory_navigation import evaluate
        evaluate()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--index", type=int, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--mem-port", type=int, default=21930)
        parser.add_argument("--nav-port", type=int, default=21931)
        args = parser.parse_args()
        args.plan, args.out = args.plan.resolve(), args.out.resolve()
        if action == "run":
            run(args)
        elif action == "verify":
            verify(args)
        else:
            parser.error("Expected run, verify, or eval")
