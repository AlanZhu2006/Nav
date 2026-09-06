#!/usr/bin/env python3
"""Measurement-only replay of the five Table-III depth conditions.

The original frozen rollout and server sources are imported from FROZEN_BASE.
The callback observes completed rollouts; it returns the original leg object
unchanged and never modifies requests, actions, or the online path counter.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys


ARMS = ("mono_native", "zero_native", "mono_cec", "metric_native", "metric_cec")
MANIFEST_SHA = "7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")


def rotated_arm_order(index):
    offset = int(index) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def install_profile(runner):
    from MemNavData.final14_zero_depth import audit_zero_depth_plans
    original_audit = runner.audit_depth_plans
    runner.ARMS = ARMS
    runner.rotated_arm_order = rotated_arm_order
    runner.SCHEMA = "final14_table3_exact_spl_replay_20260907"
    for name, zero_value in (("DEPTH_SOURCE", "zero"),
                             ("HYBRID_ROUTE", "native_sidecar"),
                             ("REVISIT_ADAPTER", "legacy_metric"),
                             ("EVALUATOR_ARM", "native_sidecar")):
        mapping = {arm: value for arm, value in getattr(runner, name).items()
                   if arm in ARMS}
        mapping["zero_native"] = zero_value
        setattr(runner, name, mapping)
    runner.audit_depth_plans = lambda arm, plans: (
        audit_zero_depth_plans(plans) if arm == "zero_native"
        else original_audit(arm, plans))


def run_history():
    from MemNavData import run_final14_mono_factorial_episode as runner
    frozen = Path(os.environ["FROZEN_BASE"]).resolve()
    require(Path(runner.__file__).resolve().is_relative_to(frozen),
            "runner must come from the frozen Final14 bundle")
    install_profile(runner)
    original_run = runner.run_command

    def measured_command(command, log_path):
        require(command[2].endswith("/eval_shared_online_role_pairs.py"),
                "unexpected evaluator command")
        replacement = command[:2] + [str(Path(__file__).resolve()), "eval"] + command[3:]
        return original_run(replacement, log_path)

    runner.run_command = measured_command
    runner.main()


def measurement(leg, goal_xz, geodesic):
    from MemNavData.executed_path_metrics import planar_path_length, spl
    endpoint = [float(x) for x in leg["end_pos"]]
    actual = planar_path_length(leg["rollout_trace"], steps=int(leg["steps"]),
                                end_position=endpoint)
    require(actual is not None, "terminal position is mandatory")
    return {
        "measurement_schema": "post_rollout_actual_planar_path_v1",
        "end_position": endpoint,
        "end_yaw_rad": float(leg["end_psi"]),
        "goal_xz_evaluator_only": [float(x) for x in goal_xz],
        "steps": int(leg["steps"]), "reached": int(bool(leg["reached"])),
        "geodesic_m": float(geodesic),
        "final_goal_dist_m": float(leg["final_goal_dist_m"]),
        "actual_path_len_m": actual,
        "commanded_path_len_m": float(leg["path_len"]),
        "spl": spl(int(bool(leg["reached"])), float(geodesic), actual),
    }


def evaluate():
    import eval_2leg_habitat as base
    frozen = Path(os.environ["FROZEN_BASE"]).resolve()
    require(Path(base.__file__).resolve() == frozen / "MemNavData/eval_2leg_habitat.py",
            "rollout source is not the frozen evaluator")
    original_leg = base.run_policy_leg
    records = []
    output = Path(base.args.out)

    def observed_leg(*args, **kwargs):
        leg = original_leg(*args, **kwargs)
        records.append(measurement(leg, args[5], args[6]))
        dump(output / "terminal_measurements.json", records)
        return leg

    base.run_policy_leg = observed_leg
    runpy.run_path(str(frozen / "MemNavData/eval_shared_online_role_pairs.py"),
                   run_name="__main__")
    metric_path = output / "metric.csv"
    with metric_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(records) == 2, "expected exactly two measured queries")
    for row, record in zip(rows, records):
        require(int(row["steps"]) == record["steps"] and
                int(row["reached"]) == record["reached"], "rollout order changed")
        require(abs(float(row["final_goal_dist_m"]) - record["final_goal_dist_m"]) < 1e-9,
                "terminal distance changed")
        row["commanded_path_len_m"] = row["path_len_m"]
        row["path_len_m"] = record["actual_path_len_m"]
        row["spl"] = record["spl"]
        for axis, value in zip("xyz", record["end_position"]):
            row[f"end_{axis}_m"] = value
        row["end_yaw_rad"] = record["end_yaw_rad"]
        path = output / f"{row['episode']}_{row['query_id']}_plans.json"
        payload = json.loads(path.read_text())
        require("query_result" not in payload, "terminal receipt already exists")
        payload["query_result"] = record
        dump(path, payload)
    # Keep the unmodified serializer output beside the measured CSV.
    metric_path.rename(output / "metric_commanded.csv")
    with metric_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_measurement(row, payload):
    """Independent endpoint/path calculation; does not import path helpers."""
    terminal = payload["query_result"]
    count = int(row["steps"])
    trace = [p for p in payload["rollout_traces"]["query"] if int(p["step"]) <= count]
    require([p["step"] for p in trace] in (list(range(count)), list(range(count + 1))),
            "missing or duplicate pre-action positions")
    points = [(float(p["x"]), float(p["z"])) for p in trace]
    end = terminal["end_position"]
    require(len(end) == 3 and all(math.isfinite(v) for v in end), "invalid endpoint")
    if len(trace) == count:
        points.append((end[0], end[2]))
    elif points:
        require(math.hypot(points[-1][0] - end[0], points[-1][1] - end[2]) < 1e-6,
                "last observation disagrees with terminal position")
    require(all(math.isfinite(v) for p in points for v in p), "nonfinite path")
    distance = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    goal = terminal["goal_xz_evaluator_only"]
    final_distance = math.hypot(end[0] - goal[0], end[2] - goal[1])
    success = int(final_distance < 1.0)
    require(success == int(row["reached"]) == terminal["reached"], "success mismatch")
    require(abs(final_distance - float(row["final_goal_dist_m"])) < 1e-6,
            "final distance mismatch")
    for axis, value in zip("xyz", end):
        require(abs(float(row[f"end_{axis}_m"]) - value) < 1e-9, "CSV endpoint mismatch")
    geodesic = float(row["geodesic_m"])
    require(math.isfinite(geodesic) and geodesic > 0, "invalid initial geodesic")
    value = success * geodesic / max(geodesic, distance)
    require(abs(distance - float(row["path_len_m"])) < 1e-7, "executed path mismatch")
    require(abs(value - float(row["spl"])) < 1e-9, "SPL mismatch")
    require(abs(distance - terminal["actual_path_len_m"]) < 1e-7, "receipt path mismatch")
    return value


def exact_p(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)


def audit_and_summarize():
    import argparse
    from MemNavData import run_final14_mono_factorial_episode as runner
    install_profile(runner)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--history-index", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.bench_root) / "manifest.json"
    require(sha256(manifest_path) == MANIFEST_SHA, "manifest changed")
    population = json.loads(manifest_path.read_text())["episodes"]
    require(len(population) == 21, "population changed")
    indices = list(range(21)) if args.history_index is None else [args.history_index]
    records, receipts = [], {}
    for index in indices:
        item = population[index]
        label = f"{index:03}_{item['scene']}_{item['episode']}"
        root = Path(args.run_root) / "evaluation/natural_direction" / label
        completion_path = root / "completion.json"
        completion = json.loads(completion_path.read_text())
        require(sha256(completion_path) == (root / "completion.json.sha256").read_text().split()[0],
                "completion SHA mismatch")
        require(completion["benchmark_manifest_sha256"] == MANIFEST_SHA and
                completion["prefix_equality"] and not completion["smoke"] and
                completion["max_steps"] == 600 and not completion["fresh_confirmation"] and
                completion["arms"] == list(ARMS) and
                completion["arm_order"] == list(rotated_arm_order(index)), "pairing contract mismatch")
        receipts[label] = sha256(completion_path)
        all_rows, all_payloads = {}, {}
        for arm in ARMS:
            with (root / arm / "metric.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == 2 and {r["analysis_role"] for r in rows} == {"novel", "revisit"},
                    "role population changed")
            payloads = runner.load_payloads(root / arm, item["episode"], rows)
            all_rows[arm] = {r["analysis_role"]: r for r in rows}
            all_payloads[arm] = payloads
            for row in rows:
                role = row["analysis_role"]
                payload = payloads[role]
                require(payload["analysis_role_not_forwarded"], "role was forwarded")
                runner.audit_depth_plans(arm, payload["query_leg"])
                value = independent_measurement(row, payload)
                require(int(row["reached"]) == completion["outcomes"][arm][role], "outcome mismatch")
                records.append({"history": index, "scene": item["scene"], "arm": arm,
                                "role": role, "success": int(row["reached"]), "spl": value,
                                "path_m": float(row["path_len_m"]), "steps": int(row["steps"])})
        for arm in ARMS:
            runner.compare_shared_replay(all_rows["mono_native"], all_payloads["mono_native"],
                                         all_rows[arm], all_payloads[arm], arm)
        for cec, native in (("mono_cec", "mono_native"), ("metric_cec", "metric_native")):
            for role in ("novel", "revisit"):
                runner.audit_fully_rejected_fallback(arm=cec, role=role,
                    cec_row=all_rows[cec][role], cec_payload=all_payloads[cec][role],
                    native_payload=all_payloads[native][role])
    require(len(records) == len(indices) * 10, "incomplete paired population")
    summary, contrasts = {}, {}
    for role in ("all", "novel", "revisit"):
        selected = [r for r in records if role == "all" or r["role"] == role]
        summary[role] = {}
        for arm in ARMS:
            rows = [r for r in selected if r["arm"] == arm]
            summary[role][arm] = {"n": len(rows), "successes": sum(r["success"] for r in rows),
                "sr": sum(r["success"] for r in rows) / len(rows),
                "spl": math.fsum(r["spl"] for r in rows) / len(rows)}
        contrasts[role] = {}
        for a, b in (("mono_cec", "mono_native"), ("metric_cec", "metric_native"),
                     ("mono_native", "zero_native"), ("mono_native", "metric_native"),
                     ("mono_cec", "metric_cec")):
            left = {(r["history"], r["role"]): r for r in selected if r["arm"] == a}
            right = {(r["history"], r["role"]): r for r in selected if r["arm"] == b}
            wins = sum(left[k]["success"] > right[k]["success"] for k in left)
            losses = sum(left[k]["success"] < right[k]["success"] for k in left)
            contrasts[role][f"{a}_vs_{b}"] = {"wins": wins, "losses": losses,
                "risk_difference": (wins - losses) / len(left), "exact_mcnemar_p": exact_p(wins, losses)}
    result = {"verified": True, "scope": "consumed_final14_measurement_replay_not_fresh_confirmation",
              "manifest_sha256": MANIFEST_SHA, "histories": len(indices),
              "scenes": len({population[i]["scene"] for i in indices}), "rollouts": len(records),
              "complete_population": len(indices) == 21, "path_metric": "actual_planar_displacement_with_terminal",
              "summary": summary, "contrasts": contrasts, "completion_sha256": receipts, "records": records}
    require(not Path(args.out).exists(), "refuse to overwrite an existing verification")
    dump(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "completion_sha256")}, indent=2))


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    {"run": run_history, "eval": evaluate, "audit": audit_and_summarize}[mode]()
