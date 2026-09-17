"""Read-only, archive-based Table-II SR/SPL and population verification.

This postprocessor does not import the rollout, constructor, metric helpers or
the original summarizer. It needs Python + NumPy, not Habitat or policy GPUs.
Running it never changes the sealed experiment or its query population.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def same_number(actual, expected, *, tolerance=1e-9):
    if (not math.isfinite(float(actual)) or not math.isfinite(float(expected))
            or abs(float(actual) - float(expected)) > tolerance):
        raise ValueError(f"Numerical mismatch: {actual} != {expected}")


def vector_equal(a, b):
    if len(a) != len(b):
        raise ValueError("Vector dimension mismatch")
    for x, y in zip(a, b):
        same_number(x, y)


def measure(terminal, trace, actions):
    """Integrate actual action displacements, including the final action."""
    count = int(terminal["steps"])
    if len(actions) != count or len(trace["poses"]) != count:
        raise ValueError("Missing action or observation")
    if [p["step"] for p in trace["poses"]] != list(range(count)):
        raise ValueError("Observation sequence is not contiguous")
    endpoint = terminal["end_position"]
    vector_equal(endpoint, trace["end_position"])
    points = [[p[k] for k in "xyz"] for p in trace["poses"]] + [endpoint]
    for i, action in enumerate(actions):
        if action["action_index"] != i or action["selected_mode"] != "bounded_standard":
            raise ValueError("Wrong action index or controller profile")
        vector_equal(points[i], action["position_before"])
        vector_equal(points[i+1], action["actual_position"])
    length = math.fsum(math.hypot(b[0]-a[0], b[2]-a[2]) for a,b in zip(points,points[1:]))
    goal = terminal["goal_xz_evaluator_only"]
    distance = math.hypot(endpoint[0]-goal[0],endpoint[2]-goal[1])
    # This is the frozen evaluator's position-based criterion, not a newly
    # introduced geodesic/visual STOP criterion.
    reached = int(distance < 1.)
    if reached != terminal["reached"] or reached != trace["reached"]:
        raise ValueError("Terminal success label disagrees with saved position")
    shortest = float(terminal["geodesic_m"])
    if not math.isfinite(shortest) or shortest <= 0:
        raise ValueError("Invalid initial shortest-path distance")
    spl = reached * shortest / max(shortest,length)
    same_number(length,terminal["actual_path_len_m"],tolerance=1e-8)
    same_number(distance,terminal["final_goal_dist_m"])
    same_number(spl,terminal["spl"])
    return dict(reached=reached,spl=spl,actual_path_len_m=length,steps=count,
                final_goal_dist_m=distance)


def read_task(folder, plan_sha):
    """Stream one gzip once; never unpack over an existing runtime directory."""
    summary = load(folder/"summary.json")
    receipt = load(folder/"archive_receipt.json")
    verifier = load(folder/"independent_verification.json")
    if not (summary["completed"] and receipt["completed"] and verifier["verified"]
            and receipt["all_member_hashes_readback_verified"]):
        raise ValueError(f"Incomplete task: {folder}")
    if summary["plan_sha256"] != plan_sha or digest(folder/"summary.json") != verifier["summary_sha256"]:
        raise ValueError("Task is not bound to this plan and verifier")
    if receipt["verification"] != verifier or digest(receipt["archive"]) != receipt["archive_sha256"]:
        raise ValueError("Archive or verification receipt changed")
    root = PurePosixPath(receipt["original_work_root"])
    needed = {"task/summary.json"}
    directories = {}
    filenames = ("terminal_measurements.json","actual_trace.json","executor_actions.jsonl",
                 "full_plan_outputs.jsonl","query_receipt.json","prefix_replay.json")
    for row in summary["records"]:
        prefix = "task/" + str(PurePosixPath(row["directory"]).relative_to(root))
        directories[row["arm"]] = prefix
        needed.update(prefix+"/"+name for name in filenames)
    b_prefix = summary.get("c_source",{}).get("prefix") if summary["stage"] == "B" else None
    if b_prefix:
        prefix_trace = "task/" + str(PurePosixPath(b_prefix).relative_to(root)) + "/online_a_trace.json"
        needed.add(prefix_trace)
    raw = {}
    with tarfile.open(receipt["archive"],"r|gz") as archive:
        for member in archive:
            if member.name not in needed:
                continue
            if not member.isfile():
                raise ValueError("Required measurement is not a regular archived file")
            if member.name in raw:
                raise ValueError("Duplicate measurement in archive")
            stream = archive.extractfile(member)
            raw[member.name] = stream.read()
    if set(raw) != needed:
        raise ValueError(f"Missing archived measurements: {needed-set(raw)}")
    if json.loads(raw["task/summary.json"]) != summary:
        raise ValueError("Small summary differs from its raw archive")
    measured, traces, runtime_queries, plan_arrays, action_arrays = {}, {}, {}, {}, {}
    for row in summary["records"]:
        arm, prefix = row["arm"], directories[row["arm"]]
        terminal, = json.loads(raw[prefix+"/terminal_measurements.json"])
        trace = json.loads(raw[prefix+"/actual_trace.json"])
        actions = [json.loads(line) for line in raw[prefix+"/executor_actions.jsonl"].splitlines() if line]
        plans = [json.loads(line) for line in raw[prefix+"/full_plan_outputs.jsonl"].splitlines() if line]
        result = measure(terminal,trace,actions)
        for key,value in terminal.items():
            if row.get(key) != value:
                raise ValueError(f"Summary changed terminal field {key}")
        query = json.loads(raw[prefix+"/query_receipt.json"])["query"]
        if {"analysis_role","covis_curve","direction_stratum","cell"}.intersection(query):
            raise ValueError("Analysis-only role/support field entered the runtime query")
        measured[arm], traces[arm], runtime_queries[arm] = result, trace, query
        plan_arrays[arm], action_arrays[arm] = plans, actions
    if summary["stage"] != "A" and measured:
        if set(measured) != {"native","cec"} or runtime_queries["native"] != runtime_queries["cec"]:
            raise ValueError("Pair did not execute the same runtime query")
        takeover = any(p["receipt"]["revisit_adapter_takeover"] is True for p in plan_arrays["cec"])
        pair, = verifier["pairs"]
        if pair["cec_takeover"] != takeover or pair["no_takeover_exact_native"] != (not takeover):
            raise ValueError("Takeover label disagrees with original plan receipts")
        if not takeover:
            if len(plan_arrays["native"]) != len(plan_arrays["cec"]) or len(action_arrays["native"]) != len(action_arrays["cec"]):
                raise ValueError("No-takeover arm changed the request/action count")
            for x,y in zip(plan_arrays["native"],plan_arrays["cec"]):
                for key in ("selected_trajectory","all_trajectory","all_values","position","yaw"):
                    if x[key] != y[key]:
                        raise ValueError("No-takeover arm changed native planning")
            for x,y in zip(action_arrays["native"],action_arrays["cec"]):
                for key in ("actual_position","actual_yaw","action_kind"):
                    if x[key] != y[key]:
                        raise ValueError("No-takeover arm changed native execution")
    return dict(summary=summary,measured=measured,traces=traces,runtime_queries=runtime_queries,
                prefix_trace=json.loads(raw[prefix_trace]) if b_prefix else None,
                prefix_trace_sha256=hashlib.sha256(raw[prefix_trace]).hexdigest() if b_prefix else None)


def independent_statistics(rows):
    if not rows:
        return None
    scenes = sorted({r["scene"] for r in rows})
    delta = [r["cec"]["reached"]-r["native"]["reached"] for r in rows]
    wins,losses = delta.count(1),delta.count(-1)
    discordant = wins+losses
    p = min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    draws = np.random.default_rng(20260910).integers(0,len(scenes),(20000,len(scenes)))
    per_scene = [([d for r,d in zip(rows,delta) if r["scene"]==s]) for s in scenes]
    totals = np.asarray([sum(ds) for ds in per_scene])
    sizes = np.asarray([len(ds) for ds in per_scene])
    bootstrap = np.sum(totals[draws],axis=1)/np.sum(sizes[draws],axis=1)
    return dict(queries=len(rows),scenes=len(scenes),gain=wins,loss=losses,
        risk_difference=sum(delta)/len(delta),exact_mcnemar_p=p,
        scene_cluster_bootstrap_ci95=np.percentile(bootstrap,[2.5,97.5]).tolist(),
        arms={a:dict(successes=sum(r[a]["reached"] for r in rows),
            sr=sum(r[a]["reached"] for r in rows)/len(rows),
            spl=math.fsum(r[a]["spl"] for r in rows)/len(rows)) for a in ("native","cec")})


def compare_statistics(measured, reported):
    if measured is None or reported is None:
        if measured != reported:
            raise ValueError("Empty/nonempty population mismatch")
        return
    for key in ("queries","scenes","gain","loss","risk_difference","exact_mcnemar_p"):
        same_number(measured[key],reported[key])
    vector_equal(measured["scene_cluster_bootstrap_ci95"],reported["scene_cluster_bootstrap_ci95"])
    for arm in ("native","cec"):
        for key in ("successes","sr","spl"):
            same_number(measured["arms"][arm][key],reported["arms"][arm][key])


def verify(plan_path, run, summary_path, out):
    plan, reported, cpop = load(plan_path),load(summary_path),load(run/"c_population.json")
    plan_sha = digest(plan_path)
    if not reported["completed"] or reported["plan_sha256"] != plan_sha or cpop["plan_sha256"] != plan_sha:
        raise ValueError("Incomplete or mixed-plan final summary")
    n = len(plan["sources"])
    expected = {"A":n,"B":2*n,"C":2*len(cpop["selected"])}
    if expected["C"] != cpop["c_query_tasks"]:
        raise ValueError("C task denominator changed")
    task_data, a_rows, pairs = {},[],[]
    for stage,count in expected.items():
        for index in range(count):
            print(f"VERIFY {stage}/{index:03d}",flush=True)
            data = read_task(run/stage/f"task_{index:03d}",plan_sha)
            task = data["summary"]
            if task["stage"] != stage or task["index"] != index:
                raise ValueError("Task identity mismatch")
            source_index = index if stage=="A" else index//2 if stage=="B" else cpop["selected"][index//2]["source_index"]
            source = plan["sources"][source_index]
            if task["source_id"] != source["source_id"]:
                raise ValueError("Task belongs to another source")
            task_data[stage,index] = data
            if stage == "A":
                a_rows.extend(data["measured"].values())
            elif data["measured"]:
                role = ("novel","revisit")[index%2]
                if any(r["role"] != role for r in task["records"]):
                    raise ValueError("Role mislabelled in paired results")
                pairs.append(dict(scene=source["scene"],stage=stage,role=role,
                    phase=task["records"][0]["phase"],**data["measured"]))
    candidates = []
    for index in range(2*n):
        b = task_data["B",index]
        candidate = b["summary"]["c_source"]
        if (candidate["b_role"] != ("novel","revisit")[index%2]
                or candidate["b_task_index"] != index or candidate["source_index"] != index//2):
            raise ValueError("B reference role or parent identity changed")
        actual_reached = bool(b["measured"].get("native",{}).get("reached",False))
        if candidate["collector"] != "native" or candidate["b_reached"] != actual_reached:
            raise ValueError("C eligibility does not use the actual native B outcome")
        if actual_reached:
            a_trace = task_data["A",index//2]["traces"]["native"]
            b_trace = b["traces"]["native"]
            merged = b["prefix_trace"]
            poses = a_trace["poses"] + b_trace["poses"]
            if len(poses) != len(merged["poses"]):
                raise ValueError("Merged history length differs from actual A+B")
            for expected_pose,actual in zip(poses,merged["poses"]):
                for key in ("x","y","z","yaw","jpg_sha256"):
                    if expected_pose[key] != actual[key]:
                        raise ValueError("A+B history has mixed or changed physical observations")
            vector_equal(merged["end_position"],b_trace["end_position"])
            vector_equal([b_trace["poses"][0][k] for k in "xyz"],a_trace["end_position"])
        if actual_reached and candidate["both_c_constructed"]:
            candidates.append(candidate)
    groups = {role:[c for c in candidates if c["b_role"]==role] for role in ("novel","revisit")}
    size = min(map(len,groups.values()))
    expected_c = [c["b_task_index"] for pair in zip(groups["novel"][:size],groups["revisit"][:size]) for c in pair]
    if [c["b_task_index"] for c in cpop["selected"]] != expected_c:
        raise ValueError("C is not the frozen source-order 50/50 reference mix")
    for index in range(expected["C"]):
        selected = cpop["selected"][index//2]
        parent = task_data["B",selected["b_task_index"]]
        current = task_data["C",index]
        for arm in ("native","cec"):
            query = current["runtime_queries"][arm]
            if query["prefix_trace_sha256"] != parent["prefix_trace_sha256"]:
                raise ValueError("C did not replay its selected actual native-B branch")
            vector_equal(query["start_position"],parent["traces"]["native"]["end_position"])
            same_number(query["start_yaw"],parent["traces"]["native"]["end_yaw"])
    measured_groups = {stage:{} for stage in ("B","C")}
    for stage in measured_groups:
        for role in ("novel","revisit"):
            rows = [r for r in pairs if r["stage"]==stage and r["role"]==role]
            measured_groups[stage][role] = independent_statistics(rows)
            compare_statistics(measured_groups[stage][role],reported["groups"][stage][role])
    for prior in ("novel","revisit"):
        for role in ("novel","revisit"):
            rows=[r for r in pairs if r["stage"]=="C" and r["role"]==role and r["phase"]=="C_after_"+prior]
            compare_statistics(independent_statistics(rows),reported["C_by_native_B_role"][prior][role])
    if len(a_rows) != reported["A"]["constructed"] or sum(r["reached"] for r in a_rows) != reported["A"]["successes"]:
        raise ValueError("A denominator or SR count changed")
    if a_rows:
        same_number(math.fsum(r["spl"] for r in a_rows)/len(a_rows),reported["A"]["mean_spl"])
    result = dict(verified=True,plan_sha256=plan_sha,summary_sha256=digest(summary_path),
        verifier_sha256=digest(__file__),expected_tasks=expected,verified_tasks=len(task_data),
        A=reported["A"],groups=measured_groups,C_reference_prefixes=len(expected_c),
        C_native_B_source_counts={"novel":size,"revisit":size},
        method="Raw archived terminal coordinates + action integration; independent paired statistics; actual A/B prefix comparison")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.plan,args.run,args.summary,args.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ("groups",)},indent=2))
