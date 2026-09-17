"""Bind newly collected mono-A to the existing repaired three-arm evaluator.

The rollout, controller, model servers and terminal-SPL verifier are reused
without editing the frozen covisibility experiment. This adapter only changes
the source population and verifies that its A used the same runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

import repaired_covisibility_eval as runtime
from seal_repaired_fullmono_queries import SCHEMA, SCOPE

original_load_task = runtime.load_task


def load_task(population_path, index):
    task, payload, query, frozen = original_load_task(population_path, index)
    population = runtime.load(population_path)
    require, sha, read = runtime.require, runtime.sha, runtime.load
    for field in ("runtime_source_receipt", "evaluation_source_receipt"):
        receipt = population[field]
        require(sha(receipt["path"]) == receipt["sha256"], "Sealed source receipt changed")
    require(sha(os.environ["REPAIRED_SOURCE_RECEIPT"]) == population["runtime_source_receipt"]["sha256"],
            "Query runtime differs from sealed actual A")
    require(sha(Path(__file__).with_name("SOURCE_BUNDLE.sha256"))
            == population["evaluation_source_receipt"]["sha256"], "Query adapter differs from sealed source")
    require(sha(population["source_plan"]) == population["source_plan_sha256"], "Source plan changed")
    plan = read(population["source_plan"])
    source = next(s for s in plan["sources"] if s["source_index"] == task["history_index"])
    require(all(payload["source"][k] == source[k] for k in ("source_index", "scene", "episode", "seed")),
            "Query identity differs from the actual A task")
    require(task["query_id"] in payload["selected_query_ids"], "Unselected diagnostic query")
    binding = next(b for b in population["source_scenes"]
                   if task["history_index"] in b["source_indices"])
    root = Path(binding["root"])
    require(str(root) == task["scene_construction_root"], "Scene construction binding changed")
    for filename, key in (("summary.json", "summary_sha256"),
                          ("independent_verification.json", "verification_sha256"),
                          ("a_input_receipt.json", "a_input_sha256")):
        require(sha(root / filename) == binding[key], "Scene receipt changed")
    inputs = read(root / "a_input_receipt.json")
    collection = Path(inputs["collection"])
    require(sha(collection / "manifest.json") == inputs["manifest_sha256"]
            and sha(collection / "independent_verification.json") == inputs["verification_sha256"],
            "Actual A verification changed")
    manifest = read(collection / "manifest.json")
    require(manifest["runtime_source_receipt"]["sha256"] == population["runtime_source_receipt"]["sha256"]
            == plan["base_runtime_sha256"], "Actual A used an old execution version")
    require(manifest["source_plan_sha256"] == population["source_plan_sha256"], "Actual A used another plan")
    verified = next(v for v in read(collection / "independent_verification.json")["goal_a"]
                    if v["source_index"] == task["history_index"])
    require(verified["reached"] and verified["trace_sha256"] == payload["online_a_trace_sha256"],
            "History is not the independently verified actual A")
    return task, payload, query, frozen


def bind_runtime(population_sha):
    runtime.require(len(population_sha) == 64 and all(c in "0123456789abcdef" for c in population_sha),
                    "An explicit frozen population SHA is required")
    runtime.SCHEMA, runtime.SCOPE = SCHEMA, SCOPE
    runtime.POPULATION_SHA = population_sha
    runtime.HERE = Path(__file__).resolve()
    runtime.load_task = load_task


def main():
    bind_runtime(os.environ.get("GEM_POPULATION_SHA", ""))
    # The sealed covisibility dispatcher loads its callback-enabled instrumenter
    # adjacent to HERE. The older base bundle does not expose query_main.
    runtime.main()


if __name__ == "__main__":
    main()
