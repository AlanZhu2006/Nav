"""Read-only independent checks of issued targets and each live episode chain."""
import argparse
from pathlib import Path

from MemNavData.table2_mixed_local import load, dump
from MemNavData.verify_table2_continuous_local import verify


def check(root):
    results = {}
    for arm in ("native", "cec"):
        results[arm] = verify(root / arm, root / arm / "independent_verification.json")
    issued, alive = [], {"native", "cec"}
    for stage in "ABC":
        record = root / f"stage_{stage}.json"
        if not record.exists():
            break
        row = load(record)
        assert set(row["live_arms"]) == alive
        if row.get("status") == "construction_empty":
            for arm in alive:
                assert load(root / f"permit_{stage}_{arm}.json")["status"] == "construction_empty"
                assert results[arm]["status"] == "task_construction_blocked"
            break
        queries = {a: load(q) for a, q in row["queries"].items()}
        assert set(queries) == alive
        assert {q["goal_rgb_sha256"] for q in queries.values()} == {row["goal_sha256"]}
        assert len({(tuple(q["floor_position"]), q["yaw_rad"]) for q in queries.values()}) == 1
        for arm in alive:
            assert queries[arm] == load(root / arm / "evaluation" / f"selected_{stage}.json")
        issued.append(dict(stage=stage, arms=sorted(alive), goal_sha256=row["goal_sha256"]))
        alive = {a for a in alive if row["outcomes"][a]["reached"]}
    # If A has no GEM intervention, different service instances and allocator
    # trimming must not create a hidden A difference in this local test.
    a = {arm: load(root / arm / "evaluation/leg_A/actual_trace.json") for arm in results}
    takeover = any(p.get("revisit_adapter_takeover") for p in a["cec"]["plans"])
    fields = ("x", "y", "z", "yaw", "jpg_sha256")
    prefix_equal = ([tuple(p[k] for k in fields) for p in a["native"]["poses"]] ==
                    [tuple(p[k] for k in fields) for p in a["cec"]["poses"]])
    if not takeover:
        assert prefix_equal, "Identical no-memory A diverged across the paired live workers"
    formal = load(root / "protocol.json")["formal_population"]
    assert all(r["formal_population"] is formal for r in results.values())
    result = dict(verified=True, formal_population=formal, issued_common_goals=issued,
                  per_arm=results, no_takeover_A_equal=prefix_equal if not takeover else None)
    dump(root / "independent_pair_verification.json", result)
    # The existing lossless HPC archiver expects these two canonical names.
    dump(root / "independent_verification.json", result)
    dump(root / "summary.json", load(root / "pair_summary.json"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(check(args.run.resolve()))
