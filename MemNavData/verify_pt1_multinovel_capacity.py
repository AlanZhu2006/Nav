"""Recount offline PT1 capacity results independently of the sampling code."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    inventory = json.loads((root / "trajectory_inventory.json").read_text())
    plan = json.loads((root / "visual_all76_plan.json").read_text())
    visual = json.loads((root / "visual_all76/summary.json").read_text())
    spatial = json.loads((root / "spatial/summary.json").read_text())
    assert visual["plan_sha256"] == sha(root / "visual_all76_plan.json")
    assert plan["inventory_sha256"] == sha(root / "trajectory_inventory.json")
    assert plan["spatial_sha256"] == sha(root / "spatial/summary.json")
    histories = {r["source_id"]: r for r in inventory["histories"]}
    assert set(histories) == {r["source_id"] for r in plan["sources"]}
    observations = {(r["source_id"], r["stage"]): r for r in visual["results"]}
    assert len(observations) == len(histories) * 3 == len(visual["results"])
    for h in histories.values():
        assert sha(h["parquet"]) == h["parquet_sha256"]
        assert sha(h["metadata"]) == h["metadata_sha256"]
        frame = pd.read_parquet(h["parquet"], columns=["action"])
        raw = np.array([np.asarray(v.tolist(), dtype=float).reshape(4, 4) for v in frame.action])
        steps = np.linalg.norm(np.diff(raw[:, :3, 3], axis=0), axis=1)
        assert abs(steps.sum() - h["recorded_path_m"]) < 1e-6
        a, b = h["switches"]
        assert np.allclose(h["leg_path_m"], [steps[:a-1].sum(), steps[a-1:b-1].sum(), steps[b-1:].sum()], atol=1e-6)
    for source in plan["sources"]:
        for stage in source["stages"]:
            name = stage["state"]["stage"]
            result = observations[source["source_id"], name]
            assert result["history_frames"] == stage["state"]["history_frames"]
            assert result["candidates"] == len(stage["candidates"]) == len(result["records"])
            assert [r["proposal_index"] for r in result["records"]] == [r["proposal_index"] for r in stage["candidates"]]
            accepted = []
            for r in result["records"]:
                curve = np.asarray(r["covis_curve"])
                assert len(curve) == result["history_frames"] == r["history_frames_checked"]
                assert np.isfinite(curve).all() and np.all((0 <= curve) & (curve <= 1))
                maximum = float(curve.max()) if len(curve) else 0.
                assert maximum == r["max_history_covis"]
                good = r["surface_points"] > 0 and maximum < .10
                assert bool(good) == r["unsupported_by_this_expert_prefix"]
                assert Path(r["goal_rgb"]).is_file()
                if good:
                    accepted.append(r)
            assert result["unsupported"] == len(accepted)
            assert result["unsupported_by_direction"] == dict(Counter(r["direction"] for r in accepted))
    stages = {}
    for name in ("A_start", "B_after_A", "C_after_AB"):
        rows = [r for r in visual["results"] if r["stage"] == name]
        records = [r for row in rows for r in row["records"]]
        accepted = [r for r in records if r["unsupported_by_this_expert_prefix"]]
        stages[name] = dict(
            histories=len(rows), candidate_images=len(records), unsupported_images=len(accepted),
            empty_surface_images=sum(r["surface_points"] == 0 for r in records),
            histories_with_unsupported=sum(r["unsupported"] > 0 for r in rows),
            histories_with_front_unsupported=sum(r["unsupported_by_direction"].get("front", 0) > 0 for r in rows),
            histories_with_all_three_unsupported_directions=sum(len(r["unsupported_by_direction"]) == 3 for r in rows),
            unsupported_by_direction=dict(Counter(r["direction"] for r in accepted)),
            unsupported_by_distance_bin=dict(Counter(r["distance_bin"] for r in accepted)),
            unsupported_and_current_covis_below_0_1=sum(r["current_view_covis"] < .10 for r in accepted),
        )
    capacities = {r["source_id"]: r for s in spatial["scene_results"] for r in s.get("source_results", [])}
    feasible = []
    for source in histories:
        b, c = observations[source, "B_after_A"], observations[source, "C_after_AB"]
        if b["unsupported"] and c["unsupported"] and capacities[source]["expert_AB_in_2_to_9m"]:
            feasible.append(dict(source_id=source,
                expert_AB_geodesic_m=capacities[source]["expert_AB_terminal_geodesic_m"],
                B_unsupported=b["unsupported"], C_unsupported=c["unsupported"],
                B_directions=b["unsupported_by_direction"], C_directions=c["unsupported_by_direction"]))
    return dict(verified=True, history_count=len(histories), scene_count=inventory["scene_count"],
        unit="offline candidate images and expert-prefix constructibility; NOT navigation SR",
        actual_online_history=False, navigation_rollouts=0, formal_population_created=False,
        stages=stages, spatially_screened_visual_followup_sources=feasible,
        followup_warning="These are diagnostic candidates, not frozen actual-online evaluation queries",
        inputs={str(root / f): sha(root / f) for f in (
            "trajectory_inventory.json", "spatial/summary.json", "visual_all76_plan.json", "visual_all76/summary.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("inputs", "spatially_screened_visual_followup_sources")}, indent=2))
    print('followup_sources:',len(result['spatially_screened_visual_followup_sources']))
