import csv
import hashlib
import json
import shutil
from pathlib import Path

import independent_verify_paper_role_pair_eval as verifier
import summarize_paper_role_pair_eval as summarizer


ARMS = summarizer.LEARNED_AMENDMENT_ARMS
MODEL_SHA = "6" * 64
PROOF_SHA = "1" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _plan(*, role: str, arm: str, accepted: bool = False) -> dict:
    base = {
        "step": 0,
        "requested_diffusion_seed": 17,
        "diffusion_seed": 17,
        "server_selected_idx": 0,
        "trajectory_candidate_count": 16,
        "selected_trajectory_sha256": "a" * 64,
    }
    if arm == "learned_pi3x_spatial":
        base.update({
            "learned_pi3x_relocalization_ok": True,
            "learned_pi3x_relocalization_accepted": accepted,
            "learned_pi3x_initial_candidate_selection_cached": False,
            "learned_pi3x_relocalization_ms": 10.0,
            "learned_pi3x_peak_gpu_memory_allocated_bytes": 1024,
            # A catastrophic value is intentional: formal summarization must
            # report a failed L3 method gate, not abort as infrastructure.
            "learned_pi3x_evaluation_gt_bearing_error_deg": (
                120.0 if accepted else None
            ),
        })
    if arm == "certified":
        base.update({
            "certified_relocalization_ms": 2.0,
            "certified_relocalization_uncached_ms": 2.0,
            "certified_relocalization_cached": False,
            "certified_relocalization_accepted": role == "revisit",
            "certified_relocalization_reason": (
                "certificate_accepted" if role == "revisit" else
                "certificate_rejected"
            ),
        })
    if arm == "geometry_fixed":
        base["router_verification_total_ms"] = 1.0
    return base


def _make_fixture(root: Path) -> None:
    benchmark = root / "benchmarks"
    benchmark.mkdir(parents=True)
    (benchmark / "SEALED").write_text("sealed\n")
    receipt = benchmark / "BENCHMARK_FILES.sha256"
    receipt.write_text("fixture\n")
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    (benchmark / "BENCHMARK_FILES.sha256.sha256").write_text(
        f"{receipt_hash}  BENCHMARK_FILES.sha256\n"
    )
    _write_json(benchmark / "population_receipt.json", {
        "policy_outcomes_read": False,
        "role_pair_constructible_histories": 1,
        "role_pair_scene_count": 1,
        "target_histories": 1,
        "target_scenes": 1,
    })
    for protocol in summarizer.PROTOCOLS:
        _write_json(benchmark / protocol / "manifest.json", {
            "episodes": [{"scene": "scene", "episode": "episode"}]
        })
        episode_root = (
            root / "evaluation" / protocol / "000_scene_episode"
        )
        _write_json(episode_root / "episode_contract.json", {
            "arms": list(ARMS),
            "learned_pi3x": {
                "model_sha256": MODEL_SHA,
                "proof_manifest_sha256": PROOF_SHA,
            },
        })
        completion = {
            "prefix_equality": True,
            "runtime_role_visibility": "none",
            "wall_time_seconds": {arm: 1.0 for arm in ARMS},
            "learned_pi3x": {
                "model_sha256": MODEL_SHA,
                "proof_manifest_sha256": PROOF_SHA,
            },
        }
        _write_json(episode_root / "completion.json", completion)
        completion_hash = hashlib.sha256(
            (episode_root / "completion.json").read_bytes()
        ).hexdigest()
        (episode_root / "completion.json.sha256").write_text(
            f"{completion_hash}  completion.json\n"
        )
        for arm in ARMS:
            arm_root = episode_root / arm
            arm_root.mkdir(parents=True)
            fields = [
                "analysis_role", "scene", "episode", "pair_id",
                "query_id", "seed", "shared_A_frames",
                "shared_A_decision_frames", "geodesic_m", "reached",
                "path_len_m", "steps", "final_goal_dist_m",
                "termination_reason", "router_active_plans",
                "certificate_accept_plans", "adapter_takeover_plans",
                "runtime_failure_plans", "learned_pi3x_accept_plans",
                "learned_pi3x_initial_inference_plans",
            ]
            rows = []
            for role, query in (("novel", "qn"), ("revisit", "qr")):
                learned_accept = (
                    arm == "learned_pi3x_spatial" and role == "revisit"
                )
                reached = int(
                    role == "novel"
                    or arm in ("certified", "learned_pi3x_spatial")
                )
                rows.append({
                    "analysis_role": role,
                    "scene": "scene",
                    "episode": "episode",
                    "pair_id": "pair",
                    "query_id": query,
                    "seed": "0",
                    "shared_A_frames": "50",
                    "shared_A_decision_frames": "7",
                    "geodesic_m": "3.0",
                    "reached": str(reached),
                    "path_len_m": "3.5",
                    "steps": "10",
                    "final_goal_dist_m": "0.5" if reached else "2.0",
                    "termination_reason": "success" if reached else "budget",
                    "router_active_plans": str(int(learned_accept)),
                    "certificate_accept_plans": str(int(
                        arm == "certified" and role == "revisit"
                    )),
                    "adapter_takeover_plans": str(int(
                        (arm == "certified" and role == "revisit")
                        or learned_accept
                    )),
                    "runtime_failure_plans": "0",
                    "learned_pi3x_accept_plans": str(int(learned_accept)),
                    "learned_pi3x_initial_inference_plans": (
                        "1" if arm == "learned_pi3x_spatial" else "0"
                    ),
                })
                plan = _plan(
                    role=role, arm=arm, accepted=learned_accept
                )
                rollout = ["native", role]
                if role == "revisit" and learned_accept:
                    rollout = ["learned", role]
                _write_json(arm_root / f"episode_{query}_plans.json", {
                    "analysis_role_not_forwarded": True,
                    "query_leg": [plan],
                    "rollout_traces": {"query": rollout},
                })
            with (arm_root / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def _one_cluster_interval_fraction(records, left, right):
    difference = sum(
        int(row["outcomes"][right]) - int(row["outcomes"][left])
        for row in records
    ) / len(records)
    return [difference, difference]


def _one_cluster_interval_percent(records, left, right):
    return [
        100.0 * value
        for value in _one_cluster_interval_fraction(records, left, right)
    ]


def _duplicate_natural_history(root: Path) -> None:
    protocol = "natural_direction"
    source = root / "evaluation" / protocol / "000_scene_episode"
    destination = root / "evaluation" / protocol / "001_scene2_episode2"
    shutil.copytree(source, destination)
    for arm in ARMS:
        metric_path = destination / arm / "metric.csv"
        with metric_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for row in rows:
            row["scene"] = "scene2"
            row["episode"] = "episode2"
            old_plan = destination / arm / f"episode_{row['query_id']}_plans.json"
            new_plan = destination / arm / f"episode2_{row['query_id']}_plans.json"
            shutil.copy2(old_plan, new_plan)
        with metric_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    manifest_path = root / "benchmarks" / protocol / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"].append({"scene": "scene2", "episode": "episode2"})
    _write_json(manifest_path, manifest)


def test_learned_summary_reports_negative_safety_gate(tmp_path, monkeypatch):
    _make_fixture(tmp_path)
    monkeypatch.setattr(
        summarizer, "cluster_interval", _one_cluster_interval_fraction
    )
    monkeypatch.setattr(
        verifier, "cluster_interval", _one_cluster_interval_percent
    )
    result = summarizer.summarize(
        tmp_path, include_learned_pi3x=True
    )
    assert result["arms"] == list(ARMS)
    safety = result["protocols"]["natural_direction"][
        "learned_pi3x_safety"
    ]
    assert safety["fully_abstained_exact_native_queries"] == 1
    assert safety["accepted_bearing_errors_over_90_deg"] == 1
    assert result["learned_pi3x_qualification"][
        "L3_novel_safety_and_exact_fallback"
    ]["pass"] is False
    latency = result["protocols"]["natural_direction"]["latency"]
    assert latency["certificate_uncached_ms"] == {
        "n": 2,
        "mean": 2.0,
        "median": 2.0,
        "p95": 2.0,
        "maximum": 2.0,
    }
    assert latency["certificate_cached_ms"]["n"] == 0
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, result)
    verification = verifier.verify(
        tmp_path,
        summary_path,
        include_learned_pi3x=True,
    )
    assert verification["verified"] is True
    assert verification["learned_gate_checks"] == {
        "L1": True,
        "L2": True,
        "L3": False,
    }


def test_final14_population_layout_keeps_hard_support_separate(
    tmp_path, monkeypatch
):
    _make_fixture(tmp_path)
    (tmp_path / "benchmarks/support_controlled").rename(
        tmp_path / "benchmarks/hard_support"
    )
    (tmp_path / "evaluation/support_controlled").rename(
        tmp_path / "evaluation/hard_support"
    )
    _duplicate_natural_history(tmp_path)
    _write_json(tmp_path / "benchmarks/population_receipt.json", {
        "schema_version": "final14_role_pair_population_v1_20260817",
        "policy_outcomes_read": False,
        "role_pair_constructible_histories": 1,
        "role_pair_scene_count": 1,
        "populations": {
            "natural_standard": {
                "histories": 2, "scenes": 2,
                "target_histories": 2, "target_scenes": 2,
                "target_met": True,
                "underpowered_if_target_not_met": False,
            },
            "hard_support": {
                "histories": 1, "scenes": 1,
                "target_histories": 1, "target_scenes": 1,
                "target_met": True,
                "underpowered_if_target_not_met": False,
            },
        },
    })
    monkeypatch.setattr(
        summarizer, "cluster_interval", _one_cluster_interval_fraction
    )
    monkeypatch.setattr(
        verifier, "cluster_interval", _one_cluster_interval_percent
    )
    result = summarizer.summarize(
        tmp_path, include_learned_pi3x=True
    )
    assert result["population"]["layout"] == (
        "final14_standard_natural_plus_hard_subset"
    )
    assert set(result["protocols"]) == {"natural_direction", "hard_support"}
    assert result["protocols"]["hard_support"]["analysis_roles"] == [
        "revisit"
    ]
    assert result["protocols"]["hard_support"][
        "duplicated_novel_is_instrumentation_only"
    ] is True
    assert result["population"]["all_protocol_targets_met"] is True
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, result)
    verification = verifier.verify(
        tmp_path,
        summary_path,
        include_learned_pi3x=True,
    )
    assert verification["verified"] is True
    assert verification["analysis_histories_by_protocol"] == {
        "natural_direction": 2,
        "hard_support": 1,
    }
