"""Local two-arm attribution of continued bearing after initial alignment.

All hooks are private to the evaluator process. The GEM writer, sparse
estimator, NavDP, observation FIFO and bounded executor are unchanged.
"""
from dataclasses import replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MemNavData")]
HERE = Path(__file__).resolve()
ARMS = ("full_gem", "initial_turn_only")
SCENES = ("gxdoqLR6rwA", "pLe4wQe7qrG", "yqstnuAEVhm", "mJXqzFtmKg4")


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


class InitialTurnIntervention:
    """Only the first decision can authorize the experimental initial turn."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError("Unknown frozen arm")
        self.arm = arm
        self.decisions = 0
        self.initial_rearward = None
        self.phase = "initial"
        self.cutoff_action = None
        self.completed_turns = 0

    @property
    def native(self):
        return self.arm == "initial_turn_only" and self.phase == "native"

    def adapt(self, decision):
        self.decisions += 1
        if self.decisions != 1:
            if self.native:
                raise RuntimeError("Sparse control was called after its cutoff")
            return decision
        point = decision.controller_pointgoal
        self.initial_rearward = bool(decision.takeover and point is not None and point[0] < 0)
        self.phase = "turning" if self.initial_rearward else "continuous"
        if self.arm == "initial_turn_only" and not self.initial_rearward:
            self.phase, self.cutoff_action = "native", 0
            # A frontward cue is not permitted one extra mixed-policy segment.
            # The already-appended frame goes directly to the native endpoint.
            return replace(decision, takeover=False,
                           reason="experimental_initial_turn_absent",
                           controller_contract="native_imagegoal",
                           controller_pointgoal=None, controller_distance_m=None)
        return decision

    def completed(self, last_turn_action):
        self.completed_turns += 1
        if self.completed_turns == 1:
            if self.initial_rearward is not True:
                raise RuntimeError("Initial alignment was not authorized by the first cue")
            self.cutoff_action = int(last_turn_action) + 1
            self.phase = "native" if self.arm == "initial_turn_only" else "continuous"
        elif self.arm == "initial_turn_only":
            raise RuntimeError("The initial-only arm executed another guided turn")


def evaluate():
    import eval_2leg_habitat as base
    from MemNavData import navdp_front_goal_adapter as front
    from MemNavData.run_habitat_minimal_repair_local import evaluate as run_evaluator
    from MemNavData.table1_repaired_eval import evaluate_query

    out = Path(base.args.out)
    state = InitialTurnIntervention(os.environ["GEM_BEARING_ARM"])
    original_front = front.FrontGoalAdapter
    original_adapt, original_plan = base.adapt_revisit_pointgoal, base.srv_plan
    original_post = base.requests.post

    class ObservedFrontGoalAdapter(original_front):
        def observe(self, yaw, action_index):
            super().observe(yaw, action_index)
            if not self.active:
                state.completed(action_index)

    def adapted(*args, **kwargs):
        return state.adapt(original_adapt(*args, **kwargs))

    def planned(*args, **kwargs):
        native = state.native
        if native:
            kwargs["policy_backend"] = "navdp"
        result = original_plan(*args, **kwargs)
        if result.get("certified_relocalization_reason") == "certificate_endpoint_failure":
            raise RuntimeError("The sparse estimator failed; no fallback completes this trial")
        native = native or state.native
        if native:
            if result.get("memory_controller_pointgoal") is not None:
                raise RuntimeError("A PointGoal survived the experimental cutoff")
            if result.get("revisit_adapter_takeover") is True:
                raise RuntimeError("Memory retained control after the cutoff")
            result["revisit_adapter_takeover"] = False
            result["bearing_ablation_native_after_initial_turn"] = True
        append(out / "bearing_decisions.jsonl", {
            "decision_index": len(decision_rows), "native_after_cutoff": native,
            "goal_sha256": hashlib.sha256(args[1]).hexdigest(),
            "image_sha256": hashlib.sha256(args[0]).hexdigest(),
            "diffusion_seed_requested": kwargs.get("diffusion_seed"),
            "diffusion_seed_returned": result.get("diffusion_seed"),
            "memory_frame_idx": result.get("memory_frame_idx"),
            "takeover": result.get("revisit_adapter_takeover"),
            "pointgoal": result.get("memory_controller_pointgoal"),
            "bearing": result.get("memory_bearing_unit"),
            "phase": state.phase,
        })
        decision_rows.append(native)
        return result

    def observed_post(url, *args, **kwargs):
        result = original_post(url, *args, **kwargs)
        if url.endswith("/certified_relocalize"):
            result.raise_for_status()
            value = result.json()
            if value.get("pnp", {}).get("status") == "runtime_exception":
                raise RuntimeError(value["pnp"])
            if any(item.get("error") for item in value.get("ranked_candidates", [])):
                raise RuntimeError("Correspondence inference failed")
        return result

    decision_rows = []
    front.FrontGoalAdapter = ObservedFrontGoalAdapter
    base.adapt_revisit_pointgoal, base.srv_plan = adapted, planned
    base.runtime_executor_motion_form = lambda *args, **kwargs: {}
    base.requests.post = observed_post
    try:
        run_evaluator(query_main=evaluate_query)
    finally:
        if out.exists():
            dump(out / "bearing_intervention.json", dict(vars(state),
                 native_decisions=sum(decision_rows), total_decisions=len(decision_rows)))


def freeze(out):
    from MemNavData.run_cec_stream_depth_closed_loop import BENCH, MEM_CKPT, NAV_CKPT, LINGBOT
    from MemNavData import table1_repaired_eval as existing
    if out.exists():
        raise FileExistsError(out)
    histories = load(BENCH / "manifest.json")["episodes"]
    if tuple(h["scene"] for h in histories) != SCENES:
        raise RuntimeError("The four fixed development histories changed")
    cells = []
    inputs = {str(BENCH / "manifest.json"): sha(BENCH / "manifest.json")}
    for index, h in enumerate(histories):
        pair_path = BENCH / h["scene"] / h["episode"] / "role_pairs.json"
        cell = dict(index=index, dataset="four_consumed_MP3D_histories", controller="navdp",
                    history_index=index, benchmark=str(BENCH), benchmark_sha256=inputs[str(BENCH / "manifest.json")],
                    scene=h["scene"], episode=h["episode"], role_pairs_sha256=sha(pair_path))
        payload, frozen, folder = existing.history(cell)
        goal = next(q for pair in payload["pairs"] for q in pair["queries"] if q["analysis_role"] == "revisit")
        cell.update(goal_rgb_sha256=goal["goal_rgb_sha256"], geodesic_m=goal["geodesic_from_a_end_m"],
                    arms_in_order=list(ARMS if index % 2 == 0 else reversed(ARMS)))
        cells.append(cell)
        paths = [pair_path, frozen["source"] / "receipt.json", frozen["source"] / "online_a_trace.json",
                 folder / goal["goal_rgb"], Path(frozen["receipt"]["source_asset"])]
        for path in paths:
            inputs[str(path)] = sha(path)
    paths = set((ROOT / "MemNavData").glob("*.py"))
    paths.update((ROOT / "NavDP/baselines/memnav").glob("*.py"))
    paths.update((ROOT / "NavDP/baselines/memnav/gem").glob("*.py"))
    paths.update((ROOT / "NavDP/baselines/navdp").glob("*.py"))
    protocol = ROOT / "MemNavData/GEM_BEARING_ATTRIBUTION_PROTOCOL_20260915.md"
    paths.add(protocol)
    source_hashes = {str(p): sha(p) for p in sorted(paths)}
    weights = {str(p): sha(p) for p in (MEM_CKPT, NAV_CKPT, LINGBOT / "weights/lingbot-map-long.pt")}
    out.mkdir(parents=True)
    for path in sorted(paths):
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    plan = dict(schema="gem_bearing_initial_turn_attribution_v1", cells=cells, arms=ARMS,
                max_steps=600, exec_horizon=8, success_radius_m=1., total_rollouts=8,
                history_source="Four previously consumed metric-NavDP A histories; query depth is monocular",
                scope="Within-history mechanism diagnostic; not fresh or paper SR confirmation",
                memory_mode="native_interval7", dense_window=64, geometry_storage="detector_support",
                kv_storage="reader_precision", authority_policy="certificate_without_coverage",
                prediction_fallback=False, radius_m=2.5, role_runtime_visible=False,
                input_sha256=inputs, source_sha256=source_hashes, weights_sha256=weights,
                created_unix=time.time())
    dump(out / "plan.json", plan)
    (out / "plan.sha256").write_text(sha(out / "plan.json") + "\n")
    print("FROZEN four histories / eight Revisit rollouts", flush=True)


def run(out, mem_port, nav_port):
    from MemNavData.run_gem_memory_navigation import servers
    from MemNavData.run_repaired_fullmono_local import execution_environment, run_child
    from MemNavData import table1_repaired_eval as existing
    plan = load(out / "plan.json")
    assert sha(out / "plan.json") == (out / "plan.sha256").read_text().strip()
    for path, digest in {**plan["input_sha256"], **plan["source_sha256"]}.items():
        if sha(path) != digest:
            raise RuntimeError("A frozen input or source changed: " + path)
    if (out / "summary.json").exists():
        raise FileExistsError("Do not implicitly rerun or resume consumed outcomes")
    (out / "logs").mkdir()
    summary = dict(completed=False, records=[], plan_sha256=sha(out / "plan.json"))
    dump(out / "summary.json", summary)
    env = dict(execution_environment(), TABLE1_PLAN=str(out / "plan.json"))
    try:
        with servers(out, plan["memory_mode"], mem_port, nav_port,
                     dense_window=plan["dense_window"], geometry_storage=plan["geometry_storage"],
                     kv_storage=plan["kv_storage"]):
            pose_log = out / "lingbot_pose_readout.jsonl"
            for cell in plan["cells"]:
                for arm in cell["arms_in_order"]:
                    target = out / "evaluation" / cell["scene"] / arm
                    command = existing.command(cell, "revisit", "cec", target, mem_port, nav_port)
                    command[2] = str(HERE)
                    command += ["--certified_authority_policy", plan["authority_policy"]]
                    dump(out / "progress.json", dict(scene=cell["scene"], arm=arm,
                         completed=len(summary["records"]), total=8, updated_unix=time.time(),
                         supervisor_pid=os.getpid()))
                    offset = pose_log.stat().st_size if pose_log.exists() else 0
                    print("START", cell["scene"], arm, flush=True)
                    seconds = run_child(command, out / "logs" / f"{cell['index']}_{arm}.log",
                                        environment=dict(env, TABLE1_INDEX=str(cell["index"]), GEM_BEARING_ARM=arm))
                    with pose_log.open() as stream:
                        stream.seek(offset)
                        dump(target / "lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
                    terminal = load(target / "terminal_measurements.json")
                    assert len(terminal) == 1
                    record = dict(terminal[0], directory=str(target), scene=cell["scene"],
                                  arm=arm, role="revisit", wall_seconds=seconds)
                    summary["records"].append(record)
                    dump(out / "summary.json", summary)
                    print("DONE", cell["scene"], arm, "reached", record["reached"],
                          "steps", record["steps"], flush=True)
        changed = [path for path, digest in plan["source_sha256"].items() if sha(path) != digest]
        if changed:
            raise RuntimeError("Frozen sources changed during evaluation: " + repr(changed))
        summary["completed"] = True
        dump(out / "summary.json", summary)
    except BaseException as error:
        dump(out / "failure.json", dict(type=type(error).__name__, error=str(error)))
        raise


def main():
    action = sys.argv.pop(1)
    if action == "eval":
        evaluate()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mem-port", type=int, default=21910)
    parser.add_argument("--nav-port", type=int, default=21911)
    args = parser.parse_args()
    out = args.out.resolve()
    if action == "freeze":
        freeze(out)
    elif action == "run":
        run(out, args.mem_port, args.nav_port)
    else:
        parser.error("Expected freeze, run, or eval")


if __name__ == "__main__":
    main()
