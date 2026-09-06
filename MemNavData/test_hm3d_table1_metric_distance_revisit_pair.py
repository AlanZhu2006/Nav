import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ARMS = ("fixed_2p5m", "full_metric")


def test_rejected_fallback_projection_ignores_diagnostics_only():
    from MemNavData.run_hm3d_table1_metric_distance_revisit_pair import (
        rejected_fallback_control_projection,
    )

    shared = {
        "revisit_adapter_takeover": False,
        "revisit_adapter_reason": "certificate_reject",
        "selected_trajectory_sha256": "same-native-trajectory",
        "diffusion_seed": 17,
        "monocular_depth_receipt": {
            "first40_scale": 1.25,
            "first40_scale_freeze_ms": 10.0,
        },
    }
    fixed = [{
        **shared,
        "revisit_adapter_mode": "verified_bearing_v1",
        "memory_pointgoal_fixed_radius_m": 2.5,
        "certified_relocalization_ms": 2.0,
    }]
    metric = [{
        **shared,
        "revisit_adapter_mode": "verified_metric_v1",
        "full_metric_adapter_schema_version": 1,
        "certified_relocalization_ms": 3.0,
        "monocular_depth_receipt": {
            **shared["monocular_depth_receipt"],
            "first40_scale_freeze_ms": 12.0,
        },
    }]
    assert rejected_fallback_control_projection(fixed) == (
        rejected_fallback_control_projection(metric)
    )

    metric[0]["selected_trajectory_sha256"] = "different-trajectory"
    assert rejected_fallback_control_projection(fixed) != (
        rejected_fallback_control_projection(metric)
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_population(tmp_path: Path, *, cap_hit: bool) -> tuple[Path, Path, str]:
    benchmark = tmp_path / "benchmark"
    episodes = [{
        "scene": f"scene_{index % 21:02d}",
        "episode": f"episode_{index:04d}",
    } for index in range(28)]
    write_json(benchmark / "manifest.json", {"episodes": episodes})
    manifest_sha = digest(benchmark / "manifest.json")
    run_root = tmp_path / "run"
    for index, item in enumerate(episodes):
        root = (run_root / "evaluation" / "revisit_metric_distance"
                / f"{index:03d}_{item['scene']}_{item['episode']}")
        # Both fixtures have a +2 or better paired metric gain.  The second
        # fixture proves that a native-support cap hit blocks promotion.
        fixed = int(index < (24 if cap_hit else 25))
        metric = int(index < 27)
        completion = {
            "schema_version": (
                "hm3d_table1_metric_distance_revisit_pair_v1_20260901"
            ),
            "benchmark_manifest_sha256": manifest_sha,
            "history_index": index,
            "scene": item["scene"],
            "episode": item["episode"],
            "arms": list(ARMS),
            "same_server_pair": True,
            "runtime_role_visibility": "none",
            "analysis_query_role": "revisit",
            "prefix_equality": True,
            "outcomes": {"fixed_2p5m": fixed, "full_metric": metric},
            "geodesic_m": {arm: 4.0 for arm in ARMS},
            "path_len_m": {"fixed_2p5m": 5.0, "full_metric": 4.0},
            "steps": {"fixed_2p5m": 50, "full_metric": 40},
            "first_takeover": {"differs_from_fixed_2p5m": True},
            "metric_controller_distances_m": [3.0, 2.0],
            "metric_native_support_cap_hits": int(cap_hit and index == 0),
        }
        write_json(root / "completion.json", completion)
        (root / "completion.json.sha256").write_text(
            digest(root / "completion.json") + "  completion.json\n"
        )
        for arm, reached in completion["outcomes"].items():
            arm_root = root / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            with (arm_root / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "analysis_role", "reached", "final_goal_dist_m",
                    "runtime_failure_plans",
                ])
                writer.writeheader()
                writer.writerow({
                    "analysis_role": "revisit",
                    "reached": reached,
                    "final_goal_dist_m": 0.8 if reached else 1.5,
                    "runtime_failure_plans": 0,
                })
            plans = []
            for distance in (3.0, 2.0):
                if arm == "fixed_2p5m":
                    plans.append({
                        "revisit_adapter_takeover": True,
                        "memory_controller_pointgoal_distance_m": 2.5,
                    })
                else:
                    plans.append({
                        "revisit_adapter_takeover": True,
                        "memory_metric_scale_m_per_raw": 1.0,
                        "memory_unbounded_pointgoal_norm": distance,
                        "memory_unbounded_pointgoal_distance_m": distance,
                        "memory_controller_pointgoal_distance_m": distance,
                    })
            write_json(arm_root / f"{item['episode']}_plans.json", {
                "analysis_role_not_forwarded": True,
                "query_leg": plans,
            })
    return run_root, benchmark, manifest_sha


def aggregate(tmp_path: Path, *, cap_hit: bool) -> dict:
    run_root, benchmark, manifest_sha = make_population(
        tmp_path, cap_hit=cap_hit,
    )
    output = tmp_path / "summary.json"
    script = Path(__file__).with_name(
        "aggregate_hm3d_table1_metric_distance_revisit_pair.py"
    )
    subprocess.run([
        sys.executable, str(script),
        "--run-root", str(run_root),
        "--benchmark-root", str(benchmark),
        "--expected-manifest-sha256", manifest_sha,
        "--out", str(output),
        "--bootstrap-samples", "1000",
    ], check=True, capture_output=True, text=True)
    return json.loads(output.read_text())


def test_metric_distance_aggregate_promotes_clean_net_gain(tmp_path: Path):
    summary = aggregate(tmp_path, cap_hit=False)
    assert summary["histories"] == 28
    assert summary["scene_clusters"] == 21
    assert summary["success"] == {"fixed_2p5m": 25, "full_metric": 27}
    assert summary["paired"]["gain"] == 2
    assert summary["paired"]["loss"] == 0
    assert summary["treatment_audit"]["native_10m_support_cap_hits"] == 0
    assert summary["preregistered_decision"] == (
        "promote_to_independent_revisit_only_replication"
    )

    from MemNavData.independent_verify_hm3d_table1_metric_distance_revisit_pair import (  # noqa: E501
        verify,
    )
    run_root = tmp_path / "run"
    benchmark = tmp_path / "benchmark"
    checked = verify(
        run_root, benchmark, digest(benchmark / "manifest.json"),
        tmp_path / "summary.json",
    )
    assert checked["verified"] is True
    assert checked["raw_metric_rows"] == 56
    assert checked["paired_gain"] == 2


def test_metric_distance_aggregate_blocks_cap_activation(tmp_path: Path):
    summary = aggregate(tmp_path, cap_hit=True)
    assert summary["paired"]["net"] == 3
    assert summary["treatment_audit"]["native_10m_support_cap_hits"] == 1
    assert summary["preregistered_decision"] == (
        "stop_keep_fixed_2p5m_native_support_saturation"
    )
