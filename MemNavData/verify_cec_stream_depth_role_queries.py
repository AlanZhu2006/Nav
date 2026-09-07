#!/usr/bin/env python3
"""CPU recount of frozen shortlist evidence and paired role-query certificates."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.benchmark_cec_stream_depth_reuse import sha
from MemNavData.certified_relocalization_runtime import (
    CERTIFIED_EPIPOLAR_THRESHOLD_PX, certificate_decision,
    fundamental_can_reach_certificate, fundamental_support, rank_candidates,
    scale_free_relative_xy)
from MemNavData.lingbot_pnp_localization import (
    SiftPnPConfig, correspondence_pnp_localize, jsonable_pnp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed"] and summary["navigation_SR"] is None
    assert summary["manifest_sha256"] == sha(root / "manifest.json")
    records = []
    for case, result in zip(manifest["cases"], summary["cases"], strict=True):
        assert case["id"] == result["case"]
        assert result["same_stream_and_scale"] and result["query_state_preserved"]
        directory = root / case["id"]
        poses = np.load(directory / "online_poses.npy", allow_pickle=False)
        for q, pair in zip(case["queries"], result["queries"], strict=True):
            assert q["id"] == pair["query_id"]
            assert q["goal_sha256"] == pair["goal_sha256"] == sha(q["goal_path"])
            assert pair["shortlist_equals_production_plan"]
            query_dir = directory / q["id"]
            evidence, matches = [], {}
            for rank, candidate in enumerate(pair["candidates"], 1):
                anchor = candidate["anchor"]
                path = query_dir / f"matches_{anchor}.npz"
                assert sha(path) == pair["matches_file_sha256"][str(anchor)]
                with np.load(path, allow_pickle=False) as saved:
                    matched = {k: saved[k] for k in saved.files}
                matches[anchor] = matched
                support = fundamental_support(
                    matched["reference_raw_points"], matched["query_raw_points"], matched["scores"],
                    tuple(matched["reference_raw_hw"]), tuple(matched["query_raw_hw"]),
                    threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX)
                evidence.append({**support, "anchor": anchor,
                                 "dino_cosine": candidate["score"], "dino_rank": rank, "error": None})
            ranked = rank_candidates(evidence)
            selected = ranked[0]
            anchor = selected["anchor"]
            possible, precheck = fundamental_can_reach_certificate(selected)
            depth = None
            if possible:
                path = directory / pair["depth_file"]
                assert sha(path) == pair["depth_sha256"]
                with np.load(path, allow_pickle=False) as saved:
                    depth = {k: saved[k] for k in saved.files}
            for arm, prefix in [("canonical", "canonical"), ("route_sparse", "sparse")]:
                saved = pair["arms"][arm]["receipt"]
                assert saved["selected_anchor"] == anchor
                assert saved["ranked_candidates"] == ranked
                if possible:
                    matched = matches[anchor]
                    pnp = jsonable_pnp(correspondence_pnp_localize(
                        matched["reference_points"], matched["query_points"],
                        depth[f"{prefix}_depth"], depth[f"{prefix}_conf"], poses[anchor],
                        config=SiftPnPConfig(), match_scores=matched["scores"],
                        epipolar_threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX))
                    assert pnp == saved["pnp"], (case["id"], q["id"], arm, "PnP differs")
                    certificate = certificate_decision(pnp)
                    assert certificate == saved["certificate"]
                    assert certificate["accepted"] == saved["accepted"]
                    if saved["accepted"]:
                        bearing = scale_free_relative_xy(poses[-1], pnp["pose9"])
                        assert np.allclose(bearing, saved["aux_pose"], atol=1e-7, rtol=1e-7)
                else:
                    assert not saved["accepted"] and saved["reason"] == precheck
                records.append({"history": case["id"], "query": q["id"], "role": q["analysis_role"],
                                "arm": arm, "accepted": saved["accepted"],
                                "PnP_exercised": possible, "verified": True})
    counts = {role: {"queries": sum(r["role"] == role and r["arm"] == "canonical" for r in records),
                    **{arm: sum(r["role"] == role and r["arm"] == arm and r["accepted"] for r in records)
                       for arm in ["canonical", "route_sparse"]}}
              for role in ["novel", "revisit"]}
    assert counts == summary["counts"]
    receipt = {"verified": True, "arm_records": len(records), "counts": counts,
               "scope": "separate CPU driver, same frozen matching-support/PnP implementation; no SR",
               "summary_sha256": sha(root / "summary.json"), "records": records}
    with (root / "independent_verification.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"verified": True, "arm_records": len(records), "counts": counts}))


if __name__ == "__main__":
    main()
