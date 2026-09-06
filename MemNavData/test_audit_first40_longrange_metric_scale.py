import json
from pathlib import Path

import pytest

from MemNavData.audit_first40_longrange_metric_scale import (
    audit,
    extract_row,
)


def _plan(*, raw=(0.6, 0.8), scale=3.0, gt=3.0, sensor_depth=False):
    raw_norm = (raw[0] ** 2 + raw[1] ** 2) ** 0.5
    return {
        "analysis_role_not_forwarded": True,
        "arm": "certified",
        "query_leg": [
            {
                "step": 0,
                "frame_idx": 100,
                "certified_graph_target_anchor": 50,
                "certified_relocalization_accepted": True,
                "revisit_adapter_takeover": True,
                "memory_unbounded_pointgoal": list(raw),
                "memory_unbounded_pointgoal_norm": raw_norm,
                "memory_unbounded_pointgoal_units": "lingbot_raw_direction_only",
                "memory_unbounded_pointgoal_distance_m": None,
                "memory_controller_pointgoal_distance_m": 2.5,
                "memory_pointgoal_fixed_radius_m": 2.5,
                "metric_depth_sensor_consumed": sensor_depth,
                "evaluation_gt_goal_distance_m": gt,
                "monocular_depth_receipt": {
                    "metric_depth_sensor_consumed": sensor_depth,
                    "scale_active": True,
                    "scale_receipt": {
                        "schema": "mdtec_first40_scale_receipt_v1_20260819",
                        "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
                        "scale_prefix_frames": 40,
                        "whole_episode_ground_cache_consumed": False,
                        "scale_valid": True,
                        "scale_hat": scale,
                        "valid_frame_ratio": 1.0,
                        "relative_floor_iqr": 0.1,
                        "scale_clamped": False,
                    },
                },
                "certified_relocalization_pnp": {
                    "inliers": 20,
                    "inlier_ratio": 0.8,
                    "reprojection_rmse_px": 1.0,
                    "query_inlier_coverage": 0.2,
                    "reference_inlier_coverage": 0.2,
                    "world_inlier_planarity": 0.1,
                    "world_inlier_spread_raw": 1.0,
                },
            }
        ],
    }


def _write_plan(root: Path, population: str, payload: dict) -> Path:
    path = root / population / "mono_cec" / "episode_0000_pair_00_revisit_plans.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    return path


def test_extract_row_reconstructs_metric_distance(tmp_path):
    path = _write_plan(
        tmp_path,
        "000_sceneA_episode_0000",
        _plan(raw=(0.6, 0.8), scale=3.0, gt=3.0),
    )
    row = extract_row(path)
    assert row is not None
    assert row.predicted_distance_m == pytest.approx(3.0)
    assert row.gt_planar_distance_m == pytest.approx(3.0)
    assert row.absolute_relative_error == pytest.approx(0.0)
    assert row.scene == "sceneA"
    assert row.history_gap_frames == 50


def test_audit_uses_one_first_handoff_per_query(tmp_path):
    _write_plan(
        tmp_path,
        "000_sceneA_episode_0000",
        _plan(raw=(1.0, 0.0), scale=3.0, gt=3.0),
    )
    _write_plan(
        tmp_path,
        "001_sceneB_episode_0000",
        _plan(raw=(1.0, 0.0), scale=2.0, gt=2.5),
    )
    rows, summary = audit(tmp_path, bootstrap_samples=50, bootstrap_seed=7)
    assert len(rows) == 2
    assert summary["scene_count"] == 2
    assert summary["metric_error"]["within_20_percent"] == 2
    assert summary["paired_metric_vs_fixed"]["metric_closer"] == 1
    assert summary["paired_metric_vs_fixed"]["fixed_closer"] == 1
    assert summary["paired_metric_vs_fixed"]["ties"] == 0


def test_extract_row_rejects_simulator_depth(tmp_path):
    path = _write_plan(
        tmp_path,
        "000_sceneA_episode_0000",
        _plan(sensor_depth=True),
    )
    with pytest.raises(ValueError, match="simulator metric depth"):
        extract_row(path)
