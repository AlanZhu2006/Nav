#!/usr/bin/env python3
"""Recompute paired PnP/certificate receipts from saved arrays, on CPU.

Independent data/driver verification with the SAME frozen PnP implementation;
not an independent reimplementation of PnP and not a navigation evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.certified_relocalization_runtime import (
    CERTIFIED_EPIPOLAR_THRESHOLD_PX, certificate_decision,
    fundamental_can_reach_certificate, fundamental_support, scale_free_relative_xy)
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
    assert summary["manifest_sha256"] == hashlib.sha256(
        (root / "manifest.json").read_bytes()).hexdigest()
    verification = []
    for case, result in zip(manifest["cases"], summary["cases"], strict=True):
        folder = root / case["id"]
        arrays = np.load(folder / "depths_and_poses.npz", allow_pickle=False)
        assert result["stream_and_scale_identical"] and result["query_state_preserved"]
        for pair in result["pairs"]:
            match = np.load(folder / f"matches_{pair['goal_index']}.npz", allow_pickle=False)
            digest = hashlib.sha256()
            for key in sorted(match.files):
                value = np.asarray(match[key])
                digest.update(key.encode())
                digest.update(str(value.shape).encode())
                digest.update(str(value.dtype).encode())
                digest.update(value.tobytes())
            assert digest.hexdigest() == pair["correspondences_sha256"]
            support = fundamental_support(
                match["reference_raw_points"], match["query_raw_points"], match["scores"],
                tuple(match["reference_raw_hw"]), tuple(match["query_raw_hw"]),
                threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX)
            possible, precheck = fundamental_can_reach_certificate(support)
            for arm, prefix in [("canonical", "canonical"), ("route_sparse", "sparse")]:
                saved = pair["arms"][arm]["receipt"]
                if possible:
                    pnp = jsonable_pnp(correspondence_pnp_localize(
                        match["reference_points"], match["query_points"],
                        arrays[f"{prefix}_depth"], arrays[f"{prefix}_conf"],
                        arrays["poses"][case["anchor"]], config=SiftPnPConfig(),
                        match_scores=match["scores"],
                        epipolar_threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX))
                    certificate = certificate_decision(pnp)
                    assert certificate == saved["certificate"], (case["id"], arm, certificate)
                    assert pnp == saved["pnp"], (case["id"], arm, "PnP differs")
                    if saved["accepted"]:
                        bearing = scale_free_relative_xy(
                            arrays["poses"][case["current"]], pnp["pose9"])
                        assert np.allclose(bearing, saved["aux_pose"], atol=1e-7, rtol=1e-7)
                else:
                    assert not saved["accepted"] and saved["reason"] == precheck
                verification.append({"case": case["id"], "goal": pair["goal_index"],
                                     "arm": arm, "accepted": saved["accepted"],
                                     "verified": True})
    receipt = {"verified": True, "arm_records": len(verification),
               "scope": "CPU recomputation from immutable matches/depth/poses using the same PnP implementation",
               "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
               "records": verification}
    target = root / "independent_verification.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"verified": True, "arm_records": len(verification), "path": str(target)}))


if __name__ == "__main__":
    main()
