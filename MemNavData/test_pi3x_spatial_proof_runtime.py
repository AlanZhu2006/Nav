import hashlib
import json

import numpy as np
import torch

from MemNavData.pi3x_spatial_proof_runtime import Pi3XSpatialProofEnsemble
from MemNavData.train_pi3x_spatial_reliability_crossfit_oof import (
    Pi3XSpatialReliabilityHead,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path, thresholds):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    members = []
    for member, threshold in enumerate(thresholds):
        model = Pi3XSpatialReliabilityHead(8, model_dim=16, layers=1, heads=4)
        for parameter in model.parameters():
            parameter.data.zero_()
        path = checkpoints / f"member_{member}.pt"
        torch.save({
            "schema_version": 1,
            "model_name": "pi3x_spatial_reliability_head_v1",
            "member": member,
            "fit_scenes": ["fit"],
            "calibration_scenes": ["calibration"],
            "threshold": threshold,
            "model_config": {
                "descriptor_dim": 8,
                "model_dim": 16,
                "layers": 1,
                "heads": 4,
            },
            "state_dict": model.state_dict(),
        }, path)
        members.append({
            "member": member,
            "threshold": threshold,
            "checkpoint": f"checkpoints/{path.name}",
            "checkpoint_sha256": _sha(path),
        })
    manifest = {
        "schema_version": 1,
        "model": {
            "descriptor_dim": 8,
            "model_dim": 16,
            "layers": 1,
            "heads": 4,
        },
        "authorization": {
            "consensus_numerator": 2,
            "consensus_denominator": len(thresholds),
        },
        "members": members,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _inputs():
    candidates, views, height, width = 2, 3, 3, 4
    return {
        "overlaps": [0.1, 0.9],
        "bearings_forward_left": [[1.0, 0.0], [3.0, 4.0]],
        "descriptors": np.zeros((candidates, views, 8), dtype=np.float32),
        "roles": np.zeros((candidates, views), dtype=np.int64),
        "relative_age": np.zeros((candidates, views), dtype=np.float32),
        "valid": np.ones((candidates, views), dtype=bool),
        "world_points_in_current": np.zeros(
            (candidates, views, height, width, 3), dtype=np.float32
        ),
        "local_points": np.zeros(
            (candidates, views, height, width, 3), dtype=np.float32
        ),
        "confidence": np.ones(
            (candidates, views, height, width, 1), dtype=np.float32
        ),
        "poses_in_current": np.zeros((candidates, views, 3, 4), dtype=np.float32),
    }


def test_runtime_selects_overlap_and_accepts(tmp_path):
    runtime = Pi3XSpatialProofEnsemble(_bundle(tmp_path, [0.4, 0.4]))
    decision = runtime.decide(**_inputs())
    assert decision.status == "accepted"
    assert decision.selected_candidate == 1
    assert decision.member_votes == 2
    assert np.allclose(decision.scale_free_bearing_forward_left, [0.6, 0.8])


def test_runtime_rejects_and_fails_closed(tmp_path):
    runtime = Pi3XSpatialProofEnsemble(_bundle(tmp_path, [0.6, 0.6]))
    decision = runtime.decide(**_inputs())
    assert decision.status == "abstain"
    assert decision.scale_free_bearing_forward_left is None
    inputs = _inputs()
    inputs["overlaps"] = [float("nan"), 0.9]
    decision = runtime.decide(**inputs)
    assert decision.status == "error"
    assert decision.scale_free_bearing_forward_left is None
