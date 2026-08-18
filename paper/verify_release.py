#!/usr/bin/env python3
"""Verify sealed paper summaries and the frozen CEC source contract."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{relative} must contain one JSON object")
    return value


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def literal_constants(relative: str) -> dict[str, Any]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                constants[target.id] = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                pass
    return constants


def verify_final14() -> None:
    summary_path = "paper/results/final14/paper_role_pair_summary.json"
    verification = load_json(
        "paper/results/final14/paper_role_pair_independent_verification.json")
    summary = load_json(summary_path)
    assert verification["verified"] is True
    assert sha256(summary_path) == verification["summary_sha256"]
    assert summary["runtime_role_visibility"] == "none"
    natural = summary["protocols"]["natural_direction"]
    metrics = natural["metrics"]
    expected = {
        "native": (7, 4, 11),
        "raw_fixed_bearing": (2, 19, 21),
        "geometry_fixed": (9, 18, 27),
        "learned_pi3x_spatial": (8, 19, 27),
        "certified": (8, 20, 28),
    }
    for arm, (novel, revisit, total) in expected.items():
        assert metrics[arm]["novel"]["successes"] == novel
        assert metrics[arm]["revisit"]["successes"] == revisit
        assert metrics[arm]["all"]["successes"] == total
    contrast = natural["contrasts"]["certified_minus_raw_fixed_bearing"]["all"]
    assert (contrast["gains"], contrast["losses"]) == (8, 1)
    assert contrast["exact_mcnemar_two_sided_p"] == 0.0390625
    qualification = summary["learned_pi3x_qualification"]
    assert qualification["L1_useful_revisit_control"]["pass"] is True
    assert qualification["L2_noninferior_to_cec"]["pass"] is False
    assert qualification["L3_novel_safety_and_exact_fallback"]["pass"] is False
    assert qualification["eligible_for_primary_method_promotion"] is False


def verify_hm3d() -> None:
    summary_path = "paper/results/hm3d/hm3d_heldout_val10_revisit_summary.json"
    verification = load_json(
        "paper/results/hm3d/"
        "hm3d_heldout_val10_revisit_independent_verification.json")
    assert verification["verified"] is True
    assert sha256(summary_path) == verification["report_sha256"]
    expected = {
        "native": (7, 7),
        "geometry_router": (17, 17),
        "raw_fixed_oracle_role": (18, 18),
        "certified_relocalization": (19, 19),
    }
    for arm, (conditional, joint) in expected.items():
        record = verification["arms"][arm]
        assert record["goal_b_successes_given_a"] == conditional
        assert record["joint_successes"] == joint
    contrast = verification["contrasts"]["certified_minus_native"]["conditional_b"]
    assert (contrast["right_only_gain"], contrast["left_only_loss"]) == (12, 0)
    assert contrast["mcnemar_exact_two_sided_p"] == 0.00048828125


def verify_three_leg() -> None:
    report_path = "paper/results/three_leg/report.json"
    verification = load_json(
        "paper/results/three_leg/independent_verification.json")
    assert sha256(report_path) == verification["report_sha256"]
    assert verification["episodes"] == 19
    assert verification["arm_successes"]["native"] == 5
    assert verification["arm_successes"]["certified"] == 16
    contrast = verification["contrasts"]["certified_minus_native"]
    assert (contrast["gains"], contrast["losses"]) == (11, 0)
    assert contrast["p"] == 0.0009765625


def verify_source_contract() -> None:
    config = load_json("paper/configs/cec_v1.json")
    proof = literal_constants("MemNavData/certified_relocalization_runtime.py")
    adapter = literal_constants("MemNavData/revisit_bearing_adapter.py")
    assert proof["CERTIFIED_CANDIDATE_TOP_K"] == config["proposal"]["top_k"]
    assert proof["CERTIFIED_CANDIDATE_MIN_GAP"] == config["proposal"][
        "minimum_temporal_gap_frames"]
    assert proof["CERTIFIED_MINIMUM_ANCHOR"] == config["proposal"][
        "minimum_anchor_frame"]
    assert proof["CERTIFIED_EPIPOLAR_THRESHOLD_PX"] == config[
        "correspondence"]["epipolar_threshold_px"]
    assert proof["CERTIFICATE_MIN_INLIERS"] == config["certificate"][
        "minimum_pnp_inliers"]
    assert proof["CERTIFICATE_MIN_QUERY_COVERAGE"] == config["certificate"][
        "minimum_query_hull_coverage"]
    assert proof["CERTIFICATE_MIN_REFERENCE_COVERAGE"] == config[
        "certificate"]["minimum_reference_hull_coverage"]
    assert proof["CERTIFICATE_MAX_REPROJECTION_RMSE_PX"] == config[
        "certificate"]["maximum_reprojection_rmse_px"]
    assert adapter["VERIFIED_BEARING_RADIUS_M"] == config[
        "controller_interface"]["fixed_residual_radius_m"]


def verify_mainline_manifest() -> None:
    manifest = load_json("paper/mainline_manifest.json")
    categories = (
        "primary_runtime",
        "closed_loop_contract",
        "final14_population_and_analysis",
        "hm3d_analysis",
        "secondary_learned_route",
    )
    listed: list[str] = []
    for category in categories:
        paths = manifest[category]
        assert isinstance(paths, list) and paths, category
        for relative in paths:
            assert isinstance(relative, str) and (ROOT / relative).is_file(), relative
            listed.append(relative)
    assert len(listed) == len(set(listed)), "mainline manifest contains duplicates"
    assert manifest["primary_method"] == "certified_episodic_compass"
    assert "pi3x_learned_proof" in manifest["disabled_in_primary"]


def main() -> None:
    verify_final14()
    verify_hm3d()
    verify_three_leg()
    verify_source_contract()
    verify_mainline_manifest()
    print("CEC paper release verified: Final14, HM3D, three-leg, and source contract")


if __name__ == "__main__":
    main()
