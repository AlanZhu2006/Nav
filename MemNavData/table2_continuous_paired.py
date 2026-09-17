"""Local common-goal coordinator; each arm keeps its own running servers/world.

Only the evaluator exchanges stage requests and target images. Policies never
see another arm, role labels, GT support, or another arm's observations.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

from MemNavData.table2_mixed_local import load, dump, sha, ROOT
from MemNavData.table2_continuous_local import SCHEMA, DECLARATION, choose, NoConstructibleGoal
from MemNavData.table2_sampling_profiles import FORWARD_SCHEMA
from MemNavData.table2_common_goals import choose_common

HERE = Path(__file__).resolve()
ARMS = ("native", "cec")


def publish(path, data):
    """Publish a complete IPC record, never a half-written JSON document."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    dump(temporary, data)
    os.link(temporary, path)  # Atomic, refuses to replace an existing receipt.
    temporary.unlink()


def await_record(path, root, processes=(), timeout=1800):
    deadline = time.monotonic() + timeout
    while not path.exists():
        abort = root / "abort.json"
        if abort.exists():
            raise RuntimeError(f"Coordinator aborted: {load(abort)}")
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"Worker {process.pid} exited before {path.name}: {process.returncode}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"No stage receipt: {path}")
        time.sleep(.2)
    return load(path)


def worker():
    from MemNavData.table2_continuous_local import evaluate_chain
    from MemNavData.run_habitat_minimal_repair_local import evaluate
    manifest = load(os.environ["TABLE2_CONTINUOUS_MANIFEST"])
    root, arm = Path(manifest["common_root"]), manifest["arm"]

    def goal(index, stage, prefix, original, position, yaw):
        publish(root / f"request_{stage}_{arm}.json", dict(
            stage=stage, arm=arm, position=position.tolist(), yaw=yaw,
            prefix=None if prefix is None else str(prefix)))
        permit = await_record(root / f"permit_{stage}_{arm}.json", root)
        if permit["status"] == "construction_empty":
            raise NoConstructibleGoal("No common successor under the unchanged per-arm task rules")
        if sha(permit["query"]) != permit["query_sha256"]:
            raise ValueError("Issued query changed")
        query = load(permit["query"])
        if original is not None and query["goal_rgb_sha256"] != original["goal_rgb_sha256"]:
            raise ValueError("The common coordinator changed the preselected A")
        return query

    def finished(index, stage, continuity, measurement):
        publish(root / f"done_{stage}_{arm}.json", dict(
            stage=stage, arm=arm, continuity=continuity, measurement=measurement))

    evaluate("role_pair", query_main=lambda: evaluate_chain(goal_provider=goal, after_leg=finished))


def trim_servers(ports, out, key):
    import requests
    result = {}
    for port in ports:
        response = requests.post(f"http://127.0.0.1:{port}/private_cuda_trim", timeout=90)
        response.raise_for_status()
        receipt = response.json()
        if not receipt["live_tensors_preserved"]:
            raise RuntimeError("Allocator trim changed live tensor allocation")
        result[str(port)] = receipt
    dump(out / "allocator" / f"{key}.json", result)


def run(args):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, hab_env, run_child, private_servers, evaluator_command, execution_environment,
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    frozen = None
    if getattr(args, "population", None) is not None:
        from MemNavData.table2_continuous_population import task_at
        if args.task_index is None or args.source_file is not None or args.sequence is not None or args.source_index is not None:
            raise ValueError("A frozen task cannot also override its source or sequence")
        frozen = task_at(args.population, args.task_index)
        source, a_query = frozen["source"], frozen["a_query"]
        args.source_index, args.sequence = frozen["source_index"], frozen["sequence"]
        dump(out / "frozen_task.json", frozen)
    else:
        if args.source_index is None or args.sequence is None:
            raise ValueError("Local smoke needs an explicit source index and sequence")
        source_file = getattr(args, "source_file", None)
        source = (load(source_file) if source_file is not None else
                  deepcopy(load(DECLARATION)["sources"][args.source_index]["initial_state"]))
        source["schema"] = FORWARD_SCHEMA
        dump(out / "source.json", source)
        run_child([HAB_PY, "-u", "-m", "MemNavData.table2_balanced_sampling", "construct",
                   "--source", str(out / "source.json"), "--stage", "A", "--out", str(out / "A_menu")],
                  out / "logs/construct_A.log", environment=hab_env())
        a_query = choose(load(out / "A_menu/construction.json"), "novel", "continuous_local/A/novel")
    dump(out / "source.json", source)
    dump(out / "A_query.json", a_query)
    commands, environments, ports_by_arm = {}, {}, {}
    for index, arm in enumerate(ARMS):
        directory = out / arm
        (directory / "logs").mkdir(parents=True)
        ports_by_arm[arm] = (args.port + 2*index, args.port + 2*index+1)
        manifest = dict(schema=SCHEMA, formal_population=frozen is not None, paired_navigation_comparison=True,
            common_online_goals=True, common_root=str(out), source=source, arm=arm,
            sequence=args.sequence, a_query=str(out / "A_query.json"), a_query_sha256=sha(out / "A_query.json"),
            runtime_profile="bounded_standard/rgb_v1/source_rgb/heading_on", max_steps_per_leg=600,
            execution_horizon=8, success_radius_m=1., declared_source_index=args.source_index,
            successor_selection="same online goal on all surviving own histories",
            code_sha256={str(p): sha(p) for p in (HERE, HERE.with_name("table2_common_goals.py"),
                HERE.with_name("table2_continuous_local.py"), HERE.with_name("private_gpu_cache_server.py"))})
        if frozen is not None:
            manifest.update(population_sha256=frozen["population_sha256"], task_index=frozen["task_index"])
        dump(directory / "manifest.json", manifest)
        command = evaluator_command(source, directory / "evaluation", *ports_by_arm[arm],
                                    arm=arm, role="novel", benchmark=out)
        command[2:4] = [str(HERE), "worker"]
        at = command.index("--role_pair_query_role")
        del command[at:at+2]
        command[command.index("--leg1_mode")+1] = "policy"
        env = dict(execution_environment(), TABLE2_CONTINUOUS_MANIFEST=str(directory / "manifest.json"))
        run_child(command + ["--contract_dry_run"], directory / "logs/cli_preflight.log", environment=env)
        commands[arm], environments[arm] = command, env
    dump(out / "protocol.json", dict(schema="table2_continuous_common_local_20260911_v1",
        formal_population=frozen is not None, sequence=args.sequence, declared_source_index=args.source_index,
        source_scene=source["scene"], runtime_role_hidden=True, common_goal_images=True,
        independent_live_model_processes=True, same_local_gpu=True,
        allocator_trim="unused blocks only; no reset, offload or replay"))
    if args.prepare_only:
        print(f"PREPARED {out}; no models or navigation started", flush=True)
        return

    processes, handles, live = {}, [], list(ARMS)
    all_ports, stages, arm_stacks = [], [], {}
    try:
        with ExitStack() as stack:
            for arm in ARMS:
                arm_stacks[arm] = stack.enter_context(ExitStack())
                arm_stacks[arm].enter_context(private_servers(out / arm, *ports_by_arm[arm], memory_control=True))
                all_ports.extend(ports_by_arm[arm])
                trim_servers(all_ports, out, f"loaded_{arm}")
            for arm in ARMS:
                handle = (out / arm / "logs/evaluation.log").open("x")
                handles.append(handle)
                processes[arm] = subprocess.Popen(commands[arm], cwd=ROOT, env=environments[arm],
                    stdout=handle, stderr=subprocess.STDOUT)
            for index, stage in enumerate("ABC"):
                if not live:
                    break
                requests = {arm: await_record(out / f"request_{stage}_{arm}.json", out,
                                             [processes[arm]]) for arm in live}
                trim_servers(all_ports, out, f"before_{stage}")
                if stage == "A":
                    queries = {arm: str(out / "A_query.json") for arm in live}
                else:
                    role = "novel" if args.sequence[index] == "N" else "revisit"
                    request = out / f"construct_{stage}.json"
                    dump(request, dict(source=source, stage=stage, role=role,
                                       prefixes={a: requests[a]["prefix"] for a in live}))
                    run_child([HAB_PY, "-u", "-m", "MemNavData.table2_common_goals",
                               "--request", str(request), "--out", str(out / f"common_{stage}")],
                              out / "logs" / f"construct_{stage}.log", environment=hab_env())
                    menu = load(out / f"common_{stage}/construction.json")
                    counts = (None if frozen is None else
                              {b: i for i, b in enumerate(frozen["successor_distance_priority"])})
                    chosen = choose_common(menu, f"continuous_common/{stage}/{role}", counts)
                    if frozen is not None:
                        dump(out / f"distance_selection_{stage}.json", dict(
                            priority=frozen["successor_distance_priority"],
                            available=sorted({c["distance_band"] for c in menu["candidates"]}),
                            selected=None if chosen is None else chosen["distance_band"],
                            eligibility_relaxed=False))
                    queries = chosen["queries"] if chosen else {}
                if not queries:
                    for arm in live:
                        publish(out / f"permit_{stage}_{arm}.json", dict(status="construction_empty"))
                    empty = dict(stage=stage, live_arms=list(live), status="construction_empty")
                    stages.append(empty)
                    dump(out / f"stage_{stage}.json", empty)
                    break
                digests = {load(q)["goal_rgb_sha256"] for q in queries.values()}
                if len(digests) != 1:
                    raise ValueError("Surviving arms were issued different images")
                entry = dict(stage=stage, live_arms=list(live), goal_sha256=next(iter(digests)),
                             queries=queries, outcomes={})
                # Alternate which arm executes first; waiting produces no observations.
                parity = index + (0 if frozen is None else frozen["task_index"])
                order = list(ARMS if parity % 2 == 0 else reversed(ARMS))
                entry["execution_order"] = [a for a in order if a in live]
                for arm in order:
                    if arm not in live:
                        continue
                    publish(out / f"permit_{stage}_{arm}.json", dict(
                        status="run", query=queries[arm], query_sha256=sha(queries[arm])))
                    print(f"RUN {stage} {arm}: same goal {entry['goal_sha256'][:12]}", flush=True)
                    done = await_record(out / f"done_{stage}_{arm}.json", out, [processes[arm]])
                    entry["outcomes"][arm] = done["measurement"]
                    if not done["measurement"]["reached"]:
                        # Its episode is over: keep all evidence, but do not
                        # reserve GPU state for legs that will never execute.
                        if processes[arm].wait(timeout=120) != 0:
                            raise RuntimeError(f"{arm} failed to archive its ended episode")
                        arm_stacks[arm].close()
                        all_ports = [p for p in all_ports if p not in ports_by_arm[arm]]
                    trim_servers(all_ports, out, f"after_{stage}_{arm}")
                    print(f"DONE {stage} {arm}: reached={done['measurement']['reached']}", flush=True)
                stages.append(entry)
                dump(out / f"stage_{stage}.json", entry)
                live = [a for a in live if entry["outcomes"][a]["reached"]]
            for arm, process in processes.items():
                if process.wait(timeout=120) != 0:
                    raise RuntimeError(f"{arm} worker did not close cleanly")
            result = dict(formal_population=frozen is not None, protocol="common online tasks, own continuous state",
                task_index=None if frozen is None else frozen["task_index"],
                population_sha256=None if frozen is None else frozen["population_sha256"],
                source_id=f"{source['scene']}/{source['episode']}", sequence=args.sequence,
                stages=stages, arms={a: load(out / a / "evaluation/chain_summary.json") for a in ARMS})
            dump(out / "pair_summary.json", result)
            print(f"COMPLETE common-goal local pair: {out}", flush=True)
    except BaseException as exc:
        if not (out / "abort.json").exists():
            publish(out / "abort.json", dict(error=repr(exc), navigation_failure_imputed=False))
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        del sys.argv[1]
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--sequence", choices=("NNN", "NNR", "NRN", "NRR"))
        parser.add_argument("--source-index", type=int)
        parser.add_argument("--source-file", type=Path, help="Explicit portable initial state; no expert history")
        parser.add_argument("--population", type=Path, help="Frozen continuous population; no A resampling")
        parser.add_argument("--task-index", type=int)
        parser.add_argument("--port", type=int, default=22081)
        parser.add_argument("--prepare-only", action="store_true")
        run(parser.parse_args())
