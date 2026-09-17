"""Frozen local controls: unconditional half-turn and GEM without explicit turns.

This is an experimental evaluator, not a change to the deployed memory module.
The half-turn uses neither a role label nor any image/pose of the goal.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MemNavData")]
from MemNavData.gem_bearing_attribution import append, dump, load, sha
from MemNavData.navdp_front_goal_adapter import FrontGoalAdapter
from MemNavData.bounded_pursuit import PursuitCommand

HERE = Path(__file__).resolve()
PARENT = ROOT / ".diagnostics/gem_bearing_attribution_20260915/initial_turn_001"


class FixedHalfTurn(FrontGoalAdapter):
    """A positive 180 degree scan once, on every query, independent of content."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.considered = False
        self.remaining = math.pi

    def consider(self, response, yaw, action_index):
        if self.considered:
            return False
        if not math.isfinite(yaw) or action_index != 0:
            raise ValueError("The fixed scan must start at query action zero")
        self.considered = self.active = True
        self.target_yaw = float(yaw + math.pi)
        self.events.append(dict(event="fixed_initial_half_turn", action_index=0,
                                yaw_before_rad=float(yaw), requested_turn_rad=math.pi,
                                target_yaw_rad=self.target_yaw, activated=True,
                                goal_or_memory_information_used=False))
        return True

    def command(self, position, yaw):
        if not self.active:
            raise RuntimeError("The fixed scan is not active")
        import numpy as np
        turn = min(self.max_turn_rad, self.remaining)
        return PursuitCommand(np.asarray(position, dtype=float).copy(), float(yaw + turn),
                              0., 0., 0., -1, "fixed_initial_half_turn")

    def observe(self, yaw, action_index):
        if not self.active:
            raise RuntimeError("Unexpected scan observation")
        self.actions += 1
        self.remaining = max(0., self.remaining - self.max_turn_rad)
        if self.remaining < 1e-8:
            self.active = False
            self.events.append(dict(event="heading_complete", action_index=int(action_index),
                                    yaw_after_rad=float(yaw), residual_rad=self.remaining,
                                    fresh_replan_required=True))


class NoExplicitTurn(FrontGoalAdapter):
    def consider(self, response, yaw, action_index):
        return False


def evaluate():
    import eval_2leg_habitat as base
    from MemNavData import navdp_front_goal_adapter as front
    from MemNavData.run_habitat_minimal_repair_local import evaluate as execute
    from MemNavData.table1_repaired_eval import evaluate_query
    arm = os.environ["GEM_REVIEWER_ARM"]
    if arm not in ("native", "fixed_half_turn", "gem_no_turn"):
        raise ValueError(arm)
    front.FrontGoalAdapter = FixedHalfTurn if arm == "fixed_half_turn" else NoExplicitTurn
    original_plan, original_post = base.srv_plan, base.requests.post
    decisions = []
    out = Path(base.args.out)

    def planned(*args, **kwargs):
        if arm != "gem_no_turn":
            kwargs["policy_backend"] = "navdp"
        result = original_plan(*args, **kwargs)
        if result.get("certified_relocalization_reason") == "certificate_endpoint_failure":
            raise RuntimeError("Sparse estimator runtime failure")
        if arm != "gem_no_turn":
            if result.get("revisit_adapter_takeover") or result.get("memory_controller_pointgoal") is not None:
                raise RuntimeError("Memory authority entered a memory-free control")
        row = dict(decision_index=len(decisions), arm=arm,
                   image_sha256=hashlib.sha256(args[0]).hexdigest(),
                   goal_sha256=hashlib.sha256(args[1]).hexdigest(),
                   diffusion_seed_requested=kwargs.get("diffusion_seed"),
                   diffusion_seed_returned=result.get("diffusion_seed"),
                   memory_frame_idx=result.get("memory_frame_idx"),
                   takeover=result.get("revisit_adapter_takeover"),
                   pointgoal=result.get("memory_controller_pointgoal"))
        append(out / "intervention_decisions.jsonl", row)
        decisions.append(row)
        return result

    def observed_post(url, *args, **kwargs):
        if arm != "gem_no_turn" and url.endswith(("/certified_relocalize", "/retrieval_probe_step", "/navdp_step_ip_mixgoal")):
            raise RuntimeError("A memory-free arm requested a historical readout")
        result = original_post(url, *args, **kwargs)
        if url.endswith("/certified_relocalize"):
            result.raise_for_status()
            value = result.json()
            if value.get("pnp", {}).get("status") == "runtime_exception" or any(
                    item.get("error") for item in value.get("ranked_candidates", [])):
                raise RuntimeError("Correspondence or PnP runtime failure")
        return result

    base.srv_plan, base.requests.post = planned, observed_post
    base.runtime_executor_motion_form = lambda *args, **kwargs: {}
    execute(query_main=evaluate_query)
    dump(out / "intervention.json", dict(arm=arm, decisions=len(decisions),
         memory_readout_enabled=arm == "gem_no_turn", fixed_scan_degrees=180 if arm == "fixed_half_turn" else 0))


def freeze(out):
    from MemNavData import table1_repaired_eval as existing
    if out.exists():
        raise FileExistsError(out)
    parent = load(PARENT / "plan.json")
    assert load(PARENT / "independent_verification.json")["verified"]
    for path, digest in parent["source_sha256"].items():
        if sha(path) != digest:
            raise RuntimeError("A previous comparison source changed: " + path)
    plan = dict(parent)
    plan.update(schema="gem_reviewer_attribution_v1", created_unix=time.time(), total_rollouts=20,
                parent_results=str(PARENT), parent_summary_sha256=sha(PARENT / "summary.json"),
                primary_comparison="Fixed positive 180-degree initial scan versus native, separately on Revisit and Novel",
                secondary_comparison="GEM without any explicit turn versus the previously verified full GEM",
                scope="Four consumed development histories / eight fixed goals; diagnostic, not a new population estimate")
    plan["input_sha256"] = dict(parent["input_sha256"])
    plan["runs"] = []
    for cell in plan["cells"]:
        payload, _, folder = existing.history(cell)
        for role in ("revisit", "novel"):
            goal = next(q for p in payload["pairs"] for q in p["queries"] if q["analysis_role"] == role)
            goalpath = folder / goal["goal_rgb"]
            plan["input_sha256"][str(goalpath)] = sha(goalpath)
            arms = ["native", "fixed_half_turn", "gem_no_turn"] if role == "revisit" else ["native", "fixed_half_turn"]
            if cell["index"] % 2:
                arms.reverse()
            for arm in arms:
                plan["runs"].append(dict(cell_index=cell["index"], scene=cell["scene"], role=role,
                                         arm=arm, goal_sha256=goal["goal_rgb_sha256"]))
    plan["source_sha256"] = dict(parent["source_sha256"], **{str(HERE): sha(HERE)})
    protocol = ROOT / "MemNavData/GEM_REVIEWER_P0_PROTOCOL_20260915.md"
    plan["source_sha256"][str(protocol)] = sha(protocol)
    out.mkdir(parents=True)
    for path in plan["source_sha256"]:
        saved = out / "source_snapshot" / Path(path).relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(Path(path).read_bytes())
    dump(out / "plan.json", plan)
    (out / "plan.sha256").write_text(sha(out / "plan.json") + "\n")
    print("FROZEN 20 additional rollouts; previous eight retained", flush=True)


def run(out, mem_port, nav_port):
    from MemNavData.run_gem_memory_navigation import servers
    from MemNavData.run_repaired_fullmono_local import execution_environment, run_child
    from MemNavData import table1_repaired_eval as existing
    plan = load(out / "plan.json")
    assert sha(out / "plan.json") == (out / "plan.sha256").read_text().strip()
    for path, digest in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        if sha(path) != digest:
            raise RuntimeError("Frozen input/source changed: " + path)
    if (out / "summary.json").exists():
        raise FileExistsError("No implicit rerun of consumed results")
    (out / "logs").mkdir()
    summary = dict(completed=False, records=[], plan_sha256=sha(out / "plan.json"))
    dump(out / "summary.json", summary)
    env = dict(execution_environment(), TABLE1_PLAN=str(out / "plan.json"))
    try:
        with servers(out, plan["memory_mode"], mem_port, nav_port, dense_window=64,
                     geometry_storage=plan["geometry_storage"], kv_storage=plan["kv_storage"]):
            pose_log = out / "lingbot_pose_readout.jsonl"
            for spec in plan["runs"]:
                cell = plan["cells"][spec["cell_index"]]
                arm, role = spec["arm"], spec["role"]
                target = out / "evaluation" / cell["scene"] / role / arm
                command = existing.command(cell, role, "cec" if arm == "gem_no_turn" else "native", target, mem_port, nav_port)
                command[2] = str(HERE)
                if arm == "gem_no_turn":
                    command += ["--certified_authority_policy", plan["authority_policy"]]
                dump(out / "progress.json", dict(spec, completed=len(summary["records"]), total=20,
                     updated_unix=time.time(), supervisor_pid=os.getpid()))
                offset = pose_log.stat().st_size if pose_log.exists() else 0
                print("START", cell["scene"], role, arm, flush=True)
                seconds = run_child(command, out / "logs" / f"{cell['index']}_{role}_{arm}.log",
                    environment=dict(env, TABLE1_INDEX=str(cell["index"]), GEM_REVIEWER_ARM=arm))
                with pose_log.open() as stream:
                    stream.seek(offset)
                    import json
                    dump(target / "lingbot_frame_poses.json", [json.loads(line) for line in stream if line.strip()])
                terminal = load(target / "terminal_measurements.json")
                assert len(terminal) == 1
                row = dict(terminal[0], directory=str(target), scene=cell["scene"],
                           arm=arm, role=role, wall_seconds=seconds)
                summary["records"].append(row)
                dump(out / "summary.json", summary)
                print("DONE", cell["scene"], role, arm, "reached", row["reached"], "steps", row["steps"], flush=True)
        for path, digest in plan["source_sha256"].items():
            if sha(path) != digest:
                raise RuntimeError("Source changed during evaluation: " + path)
        summary["completed"] = True
        dump(out / "summary.json", summary)
    except BaseException as error:
        dump(out / "failure.json", dict(type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "eval":
        evaluate()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--mem-port", type=int, default=21920)
        parser.add_argument("--nav-port", type=int, default=21921)
        args = parser.parse_args()
        if action == "freeze":
            freeze(args.out.resolve())
        elif action == "run":
            run(args.out.resolve(), args.mem_port, args.nav_port)
        else:
            parser.error("Expected freeze, run, or eval")
