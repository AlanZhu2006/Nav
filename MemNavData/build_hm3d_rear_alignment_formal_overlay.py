#!/usr/bin/env python3
"""Build the rear-alignment diagnostic on the sealed route-tangent runtime.

The working tree contains later long-range attribution branches.  This builder
deliberately starts from the independently verified 2026-09-03 route-tangent
source bundle, then applies only the proof-bound initial-yaw mechanism.  The
result therefore changes the treatment and its receipts without changing the
visual route estimator whose parent outcomes selected the consumed population.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import stat


FORMAL_BUNDLE_NAME = "hm3d_longrange_route_tangent_751efdcbb436c99b"

OVERLAY_FILES = (
    "MemNavData/hm3d_longrange_rear_alignment_protocol_20260904.json",
    "MemNavData/hm3d_longrange_rear_alignment.py",
    "MemNavData/run_hm3d_longrange_rear_alignment_history.py",
    "MemNavData/analyze_hm3d_longrange_rear_alignment.py",
    "MemNavData/independent_verify_hm3d_longrange_rear_alignment.py",
    "MemNavData/slurm_hm3d_longrange_rear_alignment.sbatch",
    "MemNavData/slurm_hm3d_longrange_rear_alignment_result.sbatch",
    "MemNavData/submit_hm3d_longrange_rear_alignment_remote.sh",
    "MemNavData/prepare_hm3d_longrange_rear_alignment_formal_overlay_bundle.sh",
    "MemNavData/build_hm3d_rear_alignment_formal_overlay.py",
    "MemNavData/route_alignment_contract.py",
    "MemNavData/cec_bearing_alignment.py",
    "MemNavData/test_hm3d_longrange_rear_alignment.py",
    "MemNavData/test_analyze_hm3d_longrange_rear_alignment.py",
    "MemNavData/test_cec_bearing_alignment.py",
    "MemNavData/test_policy_agent_route_alignment_minimal.py",
    "MemNavData/test_slurm_port_pair.sh",
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def extract_method(text: str, name: str, next_name: str) -> str:
    start = text.index(f"    @torch.no_grad()\n    def {name}")
    end = text.index(f"\n    @torch.no_grad()\n    def {next_name}", start)
    return text[start:end]


def minimal_policy(formal: str, current: str) -> str:
    """Splice only proof-bound route alignment into the formal policy."""

    method = extract_method(
        current,
        "_certified_monocular_route_tangent_direction",
        "certified_path_field_reanchor",
    )
    later_diagnostics = '''            "local_tangent_route_motion_model": (
                self.certified_route_motion_model),
            "local_tangent_historical_motion_model": (
                certified_route_edge_motion_model(
                    self.certified_route_motion_model,
                    "historical_adjacent_sample")),
            "local_tangent_live_query_motion_model": (
                certified_route_edge_motion_model(
                    self.certified_route_motion_model,
                    "live_query_adjacent_sample")),
            "local_tangent_query_motion_sampling": (
                "per_action_dense"
                if self.certified_route_motion_model ==
                "direct_pnp_dense_query"
                else "fixed_stride_sparse"
            ),
            "local_tangent_route_depth_cache_stride": int(
                self.certified_route_depth_cache_stride),
            "local_tangent_unfiltered_pnp_shadow_enabled": bool(
                self.certified_route_motion_unfiltered_shadow),
'''
    method = replace_once(
        method, later_diagnostics, "", "remove post-formal route-motion branch")
    method = replace_once(
        method,
        '''                        cached_depth_frames=(set(
                            self._certified_route_depth_cached_anchors).union(
                                self._certified_route_live_depth_cache)),
''',
        '''                        cached_depth_frames=(
                            self._certified_route_depth_cached_anchors),
''',
        "restore formal sparse query schedule",
    )
    method = replace_once(
        method,
        '''                self._certified_route_live_depth_cache.clear()
''',
        "",
        "remove post-formal live-depth cache mutation",
    )
    for forbidden in (
        "certified_route_motion_model",
        "certified_route_edge_motion_model",
        "_certified_route_live_depth_cache",
        "direct_pnp_dense_query",
    ):
        if forbidden in method:
            raise RuntimeError(
                f"minimal route method retained post-formal symbol {forbidden}")

    formal_method = extract_method(
        formal,
        "_certified_monocular_route_tangent_direction",
        "certified_path_field_reanchor",
    )
    result = replace_once(
        formal, formal_method, method, "splice proof-bound route method")
    result = replace_once(
        result,
        "import hashlib\nimport json\nimport os\n",
        "import hashlib\nimport json\nimport math\nimport os\n",
        "add route-yaw math dependency")

    cached_old = '''                            goal_pose9=np.asarray(
                                cached["goal_pose9"], dtype=np.float64),
                        ))'''
    cached_new = '''                            goal_pose9=np.asarray(
                                cached["goal_pose9"], dtype=np.float64),
                            authority_proof={
                                "ok": result.get("ok"),
                                "accepted": result.get("accepted"),
                                "reason": result.get("reason"),
                                "selected_anchor": result.get(
                                    "selected_anchor"),
                                "selected_anchor_image_sha256": result.get(
                                    "selected_anchor_image_sha256"),
                                "certificate": result.get("certificate"),
                                "authority": result.get("authority"),
                                "pnp": result.get("pnp"),
                            },
                            goal_image_sha256=hashlib.sha256(
                                goal_jpg_bytes).hexdigest(),
                        ))'''
    result = replace_once(
        result, cached_old, cached_new, "bind cached CEC route proof")

    uncached_old = '''            elif guidance_mode == "monocular_route_tangent":
                bearing_vector, guidance_diagnostics = (
                    self._certified_monocular_route_tangent_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                        goal_pose9=goal_pose9,
                    ))'''
    uncached_new = '''            elif guidance_mode == "monocular_route_tangent":
                route_anchor_record = self._certified_anchor_image_record(
                    selected_anchor)
                bearing_vector, guidance_diagnostics = (
                    self._certified_monocular_route_tangent_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                        goal_pose9=goal_pose9,
                        authority_proof={
                            "ok": True,
                            "accepted": True,
                            "reason": final_attempt["reason"],
                            "selected_anchor": selected_anchor,
                            "selected_anchor_image_sha256": (
                                route_anchor_record["sha256"]),
                            "certificate": certificate,
                            "authority": final_attempt["authority"],
                            "pnp": pnp,
                        },
                        goal_image_sha256=hashlib.sha256(
                            goal_jpg_bytes).hexdigest(),
                    ))'''
    result = replace_once(
        result, uncached_old, uncached_new, "bind uncached CEC route proof")
    return result


def minimal_eval_shared(formal: str) -> str:
    old = '''    if args.cec_initial_bearing_alignment != "off":
        require(
            arm == "cec_portability",
            "CEC bearing alignment requires the proof-carrying portability hub",
        )'''
    new = '''    if (args.cec_initial_bearing_alignment
            == "first_route_tangent_rear_bounded"):
        require(
            arm == "certified"
            and args.hybrid_route == "certified_relocalization"
            and args.certified_guidance_mode == "monocular_route_tangent"
            and args.revisit_adapter == "verified_bearing_v1",
            "route alignment requires the certified monocular route tangent",
        )
        require(
            args.role_pair_scope == "table3_longrange_route_tangent"
            and args.role_pair_query_role == "revisit",
            "route alignment is restricted to the consumed long-range "
            "mechanism population",
        )
    elif args.cec_initial_bearing_alignment != "off":
        require(
            arm == "cec_portability",
            "CEC bearing alignment requires the proof-carrying portability hub",
        )'''
    return replace_once(
        formal, old, new, "authorize only the consumed route-alignment scope")


def minimal_evaluator(formal: str, current: str) -> str:
    """Use the tested evaluator U-turn plumbing, removing later adapters."""

    result = current
    result = replace_once(
        result, "    FIXED_BEARING_MODES,\n", "",
        "remove support-projection import")
    result = replace_once(
        result,
        '''          "exactly to native ImageGoal. "
          "verified_navdp_support_projection_v1 is a route-tangent-only "
          "attribution arm that preserves the fixed token norm when a "
          "verified bearing lies behind NavDP's representable half-plane. "
          "verified_bounded_metric_v1 is an "''',
        '''          "exactly to native ImageGoal. verified_bounded_metric_v1 is an "''',
        "remove support-projection CLI prose",
    )
    result = replace_once(
        result,
        '''                                  navdp_support_projection_schema_version=(
                                      response.get(
                                          "navdp_support_projection_schema_version")),
                                  memory_navdp_support_projection_applied=(
                                      response.get(
                                          "memory_navdp_support_projection_applied")),
                                  memory_controller_bearing_unit=response.get(
                                      "memory_controller_bearing_unit"),
                                  memory_source_bearing_heading_deg=(
                                      response.get(
                                          "memory_source_bearing_heading_deg")),
                                  memory_controller_bearing_heading_deg=(
                                      response.get(
                                          "memory_controller_bearing_heading_deg")),
''',
        "",
        "remove inactive support-projection result fields",
    )
    result = replace_once(
        result,
        '''                "verified_bearing_v1",
                "verified_navdp_support_projection_v1",
                "verified_bounded_metric_v1",
''',
        '''                "verified_bearing_v1", "verified_bounded_metric_v1",
''',
        "restore formal certified adapter set",
    )
    result = replace_once(
        result,
        '''            "--revisit_adapter verified_bearing_v1 or the explicit "
            "verified_navdp_support_projection_v1, "
            "verified_bounded_metric_v1 or verified_metric_v1 challenger")''',
        '''            "--revisit_adapter verified_bearing_v1 or the explicit "
            "verified_bounded_metric_v1 or verified_metric_v1 challenger")''',
        "restore formal adapter error",
    )
    support_validation = '''        adapter_ok = bool(
            args.revisit_adapter == "verified_bearing_v1"
            or (
                args.certified_guidance_mode == "monocular_route_tangent"
                and args.revisit_adapter
                == "verified_navdp_support_projection_v1"
            )
        )
        if (args.server_backend != "hybrid_pose"
                or args.hybrid_route != "certified_relocalization"
                or not adapter_ok
                or args.navdp_depth_source != "monocular_sidecar"):
            raise ValueError(
                "long-range guidance requires full-mono hybrid_pose "
                "certified_relocalization with verified_bearing_v1; only "
                "monocular_route_tangent may use the explicit frozen-NavDP "
                "support-projection attribution adapter")
'''
    formal_validation = '''        if (args.server_backend != "hybrid_pose"
                or args.hybrid_route != "certified_relocalization"
                or args.revisit_adapter != "verified_bearing_v1"
                or args.navdp_depth_source != "monocular_sidecar"):
            raise ValueError(
                "long-range guidance requires full-mono hybrid_pose "
                "certified_relocalization with verified_bearing_v1")
'''
    result = replace_once(
        result, support_validation, formal_validation,
        "restore formal long-range adapter validation")
    result = result.replace(
        "if args.revisit_adapter in FIXED_BEARING_MODES else None",
        "if args.revisit_adapter == \"verified_bearing_v1\" else None",
    )
    if result.count("verified_navdp_support_projection_v1") != 0:
        raise RuntimeError("sanitized evaluator retained support projection")

    # The remaining diff against the sealed evaluator must be the initial-yaw
    # treatment and its frame-bound action receipts only.
    for required in (
        "first_route_tangent_rear_bounded",
        "route_alignment_executor_source",
        "bind_next_route_alignment_receipt",
        "certified_route_alignment_turn",
    ):
        if required not in result:
            raise RuntimeError(f"minimal evaluator lacks {required}")
    if "first_route_tangent_rear_bounded" in formal:
        raise RuntimeError("formal evaluator unexpectedly contains treatment")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    formal = args.formal_root.resolve()
    stage = args.stage_root.resolve()
    if formal.name != FORMAL_BUNDLE_NAME:
        raise RuntimeError("unexpected formal route-tangent bundle")
    if not (formal / "SOURCE_BUNDLE.sha256").is_file():
        raise RuntimeError("formal source receipt is missing")
    if stage.exists() and any(stage.iterdir()):
        raise RuntimeError("stage must be empty")
    shutil.copytree(formal, stage, dirs_exist_ok=True, copy_function=shutil.copy2)
    for path in (stage, *stage.rglob("*")):
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR)
    shutil.rmtree(stage / ".pytest_cache", ignore_errors=True)
    for stale in ("SOURCE_BUNDLE.sha256", "source_bundle_manifest.json"):
        (stage / stale).unlink(missing_ok=True)

    for relative in OVERLAY_FILES:
        source = workspace / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"missing physical overlay source {relative}")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    policy_path = Path("NavDP/baselines/memnav/policy_agent.py")
    formal_policy = (formal / policy_path).read_text()
    current_policy = (workspace / policy_path).read_text()
    (stage / policy_path).write_text(
        minimal_policy(formal_policy, current_policy))

    evaluator_path = Path("MemNavData/eval_2leg_habitat.py")
    (stage / evaluator_path).write_text(minimal_evaluator(
        (formal / evaluator_path).read_text(),
        (workspace / evaluator_path).read_text(),
    ))

    shared_path = Path("MemNavData/eval_shared_online_role_pairs.py")
    (stage / shared_path).write_text(minimal_eval_shared(
        (formal / shared_path).read_text()))

    # These files are intentionally byte-identical to the verified parent.
    for relative in (
        "MemNavData/monocular_adjacent_motion.py",
        "MemNavData/monocular_route_tangent_runtime.py",
        "MemNavData/run_hm3d_fullmono_server_scene.sh",
        "NavDP/baselines/memnav/memnav_server.py",
        "NavDP/baselines/navdp/policy_agent.py",
        "NavDP/baselines/navdp/navdp_server.py",
    ):
        if (stage / relative).read_bytes() != (formal / relative).read_bytes():
            raise RuntimeError(f"formal runtime drifted during overlay: {relative}")


if __name__ == "__main__":
    main()
