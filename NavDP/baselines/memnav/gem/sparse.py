"""Frozen image retrieval, geometric certification and current-relative readout."""

import hashlib
import os
import time

import numpy as np
import torch

try:
    from ..router_candidates import temporal_nms_candidates
except ImportError:  # script-local memnav_server entrypoint
    from router_candidates import temporal_nms_candidates

CERTIFIED_GUIDANCE_MODES = (
    "endpoint_bearing",
    "episodic_path_field",
    "action_coordinate_compass",
    "se2_route_compass",
    "monocular_route_tangent",
)


class SparseReadout:
    def read_sparse(
            self, goal_jpg_bytes, candidates, *, route_start_anchor=None,
            graph_rescue=False, allow_learned_rescue=False,
            proposal_order="geometry_first", goal_camera_intrinsic=None,
            authority_policy="strict_certificate",
            guidance_mode="endpoint_bearing", goal_key_override=None,
            reference_depth_source=None):
        """Rank/localize/certify once; update only scale-free bearing later."""
        backend = self.backend
        import hashlib
        import time
        from pathlib import Path

        from MemNavData.certified_relocalization_runtime import (
            CERTIFIED_CANDIDATE_TOP_K,
            CERTIFIED_EPIPOLAR_THRESHOLD_PX,
            CERTIFIED_MINIMUM_ANCHOR,
            COVERAGE_ABLATION_AUTHORITY_POLICY,
            STRICT_AUTHORITY_POLICY,
            SUPPORTED_AUTHORITY_POLICIES,
            UNTHRESHOLDED_WITNESS_AUTHORITY_POLICY,
            fundamental_can_reach_certificate,
            fundamental_support,
            operational_authority_decision,
            rank_candidates,
            runtime_contract,
        )
        from MemNavData.lingbot_pnp_localization import (
            SiftPnPConfig,
            correspondence_pnp_localize,
            jsonable_pnp,
            map_raw_intrinsic_to_lingbot_pad,
        )

        started = time.perf_counter()
        frame_idx = self.frame_count - 1
        proposal_order = str(proposal_order)
        authority_policy = str(authority_policy)
        guidance_mode = str(guidance_mode)
        reference_depth_source = str(
            reference_depth_source if reference_depth_source is not None
            else getattr(backend, "certified_reference_depth_source", "canonical"))
        if goal_camera_intrinsic is None:
            raw_goal_intrinsic = None
        else:
            try:
                raw_goal_intrinsic = np.asarray(
                    goal_camera_intrinsic, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raw_goal_intrinsic = np.empty((0, 0), dtype=np.float64)
            if (raw_goal_intrinsic.shape != (3, 3)
                    or not np.isfinite(raw_goal_intrinsic).all()
                    or raw_goal_intrinsic[0, 0] <= 0.0
                    or raw_goal_intrinsic[1, 1] <= 0.0):
                return {
                    "ok": False, "accepted": False,
                    "reason": "invalid_goal_camera_intrinsic",
                    "cached": False,
                    "relocalization_ms": 1000.0 * (
                        time.perf_counter() - started),
                }
        base = {
            "certified_relocalization_schema_version": (
                runtime_contract()["schema_version"]),
            "certified_relocalization_contract": runtime_contract(),
            "frame_idx": frame_idx,
            "aux_pose": None,
            "learned_rescue_requested": bool(allow_learned_rescue),
            "learned_rescue_available": (
                getattr(backend, "cdec_pairwise_ranker", None) is not None),
            "proposal_order": proposal_order,
            "authority_policy": authority_policy,
            "guidance_mode": guidance_mode,
            "reference_depth_source": reference_depth_source,
            "goal_camera_calibration": (
                "explicit_distinct_intrinsic"
                if raw_goal_intrinsic is not None
                else "legacy_shared_history_intrinsic"
            ),
        }
        if proposal_order not in (
                "geometry_first", "dino_first_certified"):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_proposal_order", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if guidance_mode not in CERTIFIED_GUIDANCE_MODES:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_guidance_mode", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        # route_sparse is retained for already-frozen diagnostic callers.
        if reference_depth_source not in ("canonical", "online_history", "route_sparse"):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_reference_depth_source", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if (guidance_mode in (
                "episodic_path_field", "action_coordinate_compass",
                "se2_route_compass", "monocular_route_tangent")
                and (graph_rescue or route_start_anchor is not None)):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "path_field_incompatible_with_graph_rescue",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if authority_policy not in SUPPORTED_AUTHORITY_POLICIES:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_authority_policy", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if (authority_policy != STRICT_AUTHORITY_POLICY
                and (proposal_order != "geometry_first"
                     or allow_learned_rescue)):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_authority_ablation_configuration",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if (proposal_order == "dino_first_certified"
                and allow_learned_rescue):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "learned_rescue_incompatible_with_proposal_order",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        if backend.certified_relocalization_matcher is None:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "certified_relocalizer_disabled",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        goal_key = (
            hashlib.md5(goal_jpg_bytes).hexdigest()
            if goal_key_override is None else str(goal_key_override)
        )
        if (len(goal_key) != 64 and goal_key_override is not None) or any(
                character not in "0123456789abcdef" for character in goal_key):
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_internal_goal_key", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        goal_start = self.goal_start_frames.get(goal_key)
        if goal_start is None:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "goal_not_probed_causally", "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        try:
            if not isinstance(candidates, list):
                raise ValueError("candidates must be a list")
            if not 0 <= len(candidates) <= CERTIFIED_CANDIDATE_TOP_K:
                raise ValueError("candidate count outside frozen top-k contract")
            canonical = []
            seen = set()
            for dino_rank, item in enumerate(candidates, start=1):
                if not isinstance(item, dict):
                    raise ValueError("candidate is not an object")
                anchor = int(item["anchor"])
                score = float(item["score"])
                if (anchor in seen or anchor < CERTIFIED_MINIMUM_ANCHOR
                        or anchor >= int(goal_start)
                        or not np.isfinite(score)):
                    raise ValueError("candidate violates causal shortlist")
                seen.add(anchor)
                canonical.append({
                    "anchor": anchor,
                    "score": score,
                    "dino_rank": dino_rank,
                })
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return {
                **base, "ok": False, "accepted": False,
                "reason": "invalid_candidate_contract",
                "error": f"{type(error).__name__}: {error}",
                "cached": False,
                "relocalization_ms": 1000.0 * (
                    time.perf_counter() - started),
            }
        fingerprint = (
            ("learned_rescue_requested", bool(allow_learned_rescue)),
            ("proposal_order", proposal_order),
            ("authority_policy", authority_policy),
            ("reference_depth_source", reference_depth_source),
            ("goal_camera_intrinsic", (
                None if raw_goal_intrinsic is None
                else tuple(float(value) for value in raw_goal_intrinsic.flat)
            )),
            *tuple(
                (item["anchor"], float(item["score"]))
                for item in canonical
            ),
        )
        cached = self.certificates.get(goal_key)
        if cached is not None:
            if cached["candidate_fingerprint"] != fingerprint:
                return {
                    **base, "ok": False, "accepted": False,
                    "reason": "candidate_contract_changed",
                    "cached": True,
                    "relocalization_ms": 1000.0 * (
                        time.perf_counter() - started),
                }
            result = dict(cached["result"])
            result.update(base, cached=True)
            if result.get("accepted"):
                direct_bearing = backend._certified_bearing_vector(
                    cached["goal_pose9"])
                view_alignment = backend._certified_view_alignment(
                    cached["goal_pose9"])
                if guidance_mode == "episodic_path_field":
                    tracked = backend.certified_path_field_reanchor(
                        goal_jpg_bytes)
                    bearing_vector = tracked.get("direction_vector")
                    guidance_diagnostics = {
                        key: value for key, value in tracked.items()
                        if (key.startswith("episodic_path_field_")
                            or key.startswith("path_"))
                    }
                    guidance_diagnostics.update(
                        route_coordinate_state_updated=bool(
                            tracked.get("state_updated")),
                        route_coordinate_reason=tracked.get("reason"),
                        route_coordinate_local_witness=tracked.get(
                            "local_witness"),
                        route_coordinate_endpoint_fallback_available=(
                            tracked.get("endpoint_fallback_available")),
                        route_coordinate_native_fallback_available=(
                            tracked.get("native_fallback_available")),
                        route_coordinate_distance_gate_present=tracked.get(
                            "distance_gate_present"),
                    )
                    if tracked.get("state_updated") is not True:
                        guidance_diagnostics.update(
                            episodic_path_field_status="geometry_failure",
                            episodic_path_field_error_type=(
                                "RouteCoordinateObservationError"),
                            episodic_path_field_error=tracked.get("reason"),
                        )
                elif guidance_mode == "action_coordinate_compass":
                    bearing_vector, guidance_diagnostics = (
                        backend._certified_action_coordinate_direction(
                            goal_key=goal_key,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                        ))
                elif guidance_mode == "se2_route_compass":
                    bearing_vector, guidance_diagnostics = (
                        backend._certified_se2_route_direction(
                            goal_key=goal_key,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                        ))
                elif guidance_mode == "monocular_route_tangent":
                    bearing_vector, guidance_diagnostics = (
                        backend._certified_monocular_route_tangent_direction(
                            goal_key=goal_key,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                            goal_pose9=np.asarray(
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
                        ))
                else:
                    bearing_vector, guidance_diagnostics = (
                        backend._certified_graph_direction(
                            goal_key=goal_key,
                            direct_bearing=direct_bearing,
                            target_anchor=int(result["selected_anchor"]),
                            goal_start_frame=int(goal_start),
                            route_start_anchor=route_start_anchor,
                            graph_rescue=graph_rescue,
                        ))
                result.update(
                    aux_pose=bearing_vector,
                    direction_vector=bearing_vector,
                    pointgoal_units="lingbot_raw_direction_only",
                    metric_scale=None,
                    **view_alignment,
                    **guidance_diagnostics,
                )
            result["relocalization_ms"] = 1000.0 * (
                time.perf_counter() - started)
            return result

        if not canonical:
            uncached_ms = 1000.0 * (time.perf_counter() - started)
            result = {
                **base,
                "ok": True,
                "accepted": False,
                "reason": "no_causal_candidate",
                "authority": None,
                "certificate": None,
                "selected_anchor": None,
                "selected_dino_rank": None,
                "candidate_count": 0,
                "ranked_candidates": [],
                "pnp": {"status": "no_causal_candidate"},
                "cached": False,
                "uncached_relocalization_ms": uncached_ms,
                "relocalization_ms": uncached_ms,
            }
            cache_result = dict(result)
            cache_result.pop("frame_idx", None)
            self.certificates[goal_key] = {
                "candidate_fingerprint": fingerprint,
                "goal_pose9": None,
                "result": cache_result,
            }
            return result

        goal_path = Path(self.rgb_dir) / f"_cert_goal_{goal_key}.jpg"
        goal_path.write_bytes(goal_jpg_bytes)
        evidence = []
        matched_by_anchor = {}
        for item in canonical:
            anchor = item["anchor"]
            try:
                matched = backend.certified_relocalization_matcher.match_paths(
                    Path(self.rgb_dir) / f"{anchor}.jpg", goal_path,
                    target_height=518, target_width=518,
                    patch_size=int(backend.lb.patch_size))
                validate_reference = getattr(self.online_depths, 'validate_reference', None)
                if reference_depth_source == 'online_history' and validate_reference is not None:
                    validate_reference(anchor, Path(self.rgb_dir) / f"{anchor}.jpg",
                        matched['reference_points'],
                        confidence_quantile=SiftPnPConfig().depth_confidence_quantile)
                support = fundamental_support(
                    matched["reference_raw_points"],
                    matched["query_raw_points"], matched["scores"],
                    tuple(matched["reference_raw_hw"]),
                    tuple(matched["query_raw_hw"]),
                    threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX)
                matched_by_anchor[anchor] = matched
                error = None
            except Exception as exception:  # one bad image must fail closed
                support = {
                    "lightglue_matches": 0,
                    "lightglue_score_median": 0.0,
                    "fundamental_inliers": 0,
                    "fundamental_inlier_ratio": 0.0,
                    "fundamental_query_grid_coverage": 0.0,
                    "fundamental_query_hull_coverage": 0.0,
                    "fundamental_reference_grid_coverage": 0.0,
                    "fundamental_reference_hull_coverage": 0.0,
                }
                error = f"{type(exception).__name__}: {exception}"
            evidence.append({
                **support,
                "anchor": anchor,
                "dino_cosine": item["score"],
                "dino_rank": item["dino_rank"],
                "error": error,
            })
        ranked = rank_candidates(evidence)
        evidence_by_anchor = {
            int(candidate["anchor"]): candidate for candidate in evidence}

        def attempt_proposal(selected, source):
            selected_anchor = int(selected["anchor"])
            if authority_policy in (
                    STRICT_AUTHORITY_POLICY, COVERAGE_ABLATION_AUTHORITY_POLICY):
                possible, precheck_reason = (
                    fundamental_can_reach_certificate(
                        selected,
                        require_coverage=(
                            authority_policy == STRICT_AUTHORITY_POLICY)))
            else:
                # This is an algorithmic minimum for PnP, not an operational
                # certificate threshold.  The diagnostic arm still requires
                # local correspondences and a finite geometric pose witness.
                possible = bool(
                    int(selected.get("lightglue_matches", 0))
                    >= int(SiftPnPConfig().min_correspondences)
                    and selected_anchor in matched_by_anchor
                )
                precheck_reason = (
                    "pnp_attemptable"
                    if possible else "precheck_lightglue_matches"
                )
            pnp = {"status": precheck_reason}
            goal_pose9 = None
            reference_depth_cache = None
            if possible and selected_anchor in matched_by_anchor:
                try:
                    if reference_depth_source in ("online_history", "route_sparse"):
                        depth, confidence = (
                            backend._certified_route_reference_depth(
                                selected_anchor))
                    else:
                        depth, confidence = backend._certified_reference_depth(
                            selected_anchor)
                    stats = getattr(
                        backend, "_certified_dense_replay_last_stats", None)
                    reference_depth_cache = (
                        dict(stats) if stats is not None else {
                            "enabled": False,
                            "anchor": selected_anchor,
                            "cache_hit": False,
                            "cache_source": "legacy_full_replay",
                        })
                    matched = matched_by_anchor[selected_anchor]
                    reference_pose = (
                        self.poses[selected_anchor].float().numpy())
                    query_intrinsic = None
                    if raw_goal_intrinsic is not None:
                        query_height, query_width = (
                            int(value) for value in matched["query_raw_hw"])
                        query_intrinsic = map_raw_intrinsic_to_lingbot_pad(
                            raw_goal_intrinsic,
                            raw_height=query_height,
                            raw_width=query_width,
                            target_height=int(depth.shape[-2]),
                            target_width=int(depth.shape[-1]),
                            patch_size=int(backend.lb.patch_size),
                        )
                    pnp = correspondence_pnp_localize(
                        matched["reference_points"], matched["query_points"],
                        depth, confidence, reference_pose,
                        config=SiftPnPConfig(),
                        match_scores=matched["scores"],
                        epipolar_threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX,
                        query_intrinsic=query_intrinsic)
                    pnp = jsonable_pnp(pnp)
                    if "pose9" in pnp:
                        candidate_pose9 = np.asarray(
                            pnp["pose9"], dtype=np.float64)
                        if (candidate_pose9.shape == (9,)
                                and np.isfinite(candidate_pose9).all()):
                            goal_pose9 = candidate_pose9
                except Exception as exception:
                    pnp = {
                        "status": "runtime_exception",
                        "error": f"{type(exception).__name__}: {exception}",
                    }
            authority = operational_authority_decision(
                pnp, policy=authority_policy)
            certificate = authority["strict_certificate"]
            accepted = bool(authority["accepted"])
            reason = (
                precheck_reason if not possible else authority["reason"])
            return {
                "source": source,
                "selected": selected,
                "selected_anchor": selected_anchor,
                "possible": possible,
                "precheck_reason": precheck_reason,
                "pnp": pnp,
                "certificate": certificate,
                "authority": authority,
                "accepted": accepted,
                "reason": reason,
                "goal_pose9": goal_pose9,
                "reference_depth_cache": reference_depth_cache,
            }

        def public_attempt(attempt):
            return {
                "source": attempt["source"],
                "selected_anchor": attempt["selected_anchor"],
                "selected_dino_rank": attempt["selected"].get("dino_rank"),
                "accepted": attempt["accepted"],
                "reason": attempt["reason"],
                "precheck_passed": attempt["possible"],
                "pnp_status": attempt["pnp"].get("status"),
                "certificate": attempt["certificate"],
                "authority": attempt["authority"],
                "reference_depth_cache": attempt[
                    "reference_depth_cache"],
            }

        geometry_attempt = None
        if proposal_order == "geometry_first":
            geometry_attempt = attempt_proposal(ranked[0], "geometry")
            final_attempt = geometry_attempt
            proposal_attempts = [public_attempt(geometry_attempt)]
        else:
            semantic_attempts = []
            accepted_semantic_attempt = None
            for canonical_item in canonical:
                semantic_attempt = attempt_proposal(
                    evidence_by_anchor[int(canonical_item["anchor"])],
                    "dino_first_certified",
                )
                semantic_attempts.append(semantic_attempt)
                if semantic_attempt["accepted"]:
                    accepted_semantic_attempt = semantic_attempt
                    break
            final_attempt = (
                accepted_semantic_attempt
                if accepted_semantic_attempt is not None
                else semantic_attempts[0]
            )
            proposal_attempts = [
                public_attempt(attempt) for attempt in semantic_attempts]
        counterfactual_audit = None
        counterfactual_dino_order_audit = None
        if (proposal_order == "geometry_first"
                and getattr(backend, "certified_counterfactual_audit", False)):
            dino_order_attempts = []
            accepted_dino_order_attempt = None
            for canonical_item in canonical:
                dino_selected = evidence_by_anchor[
                    int(canonical_item["anchor"])]
                if (int(dino_selected["anchor"])
                        == int(geometry_attempt["selected_anchor"])):
                    public = {
                        **public_attempt(geometry_attempt),
                        "source": "dino_order_geometry_attempt_reuse",
                        "action_authority": False,
                    }
                else:
                    dino_attempt = attempt_proposal(
                        dino_selected, "dino_order_counterfactual")
                    public = {
                        **public_attempt(dino_attempt),
                        "action_authority": False,
                    }
                dino_order_attempts.append(public)
                if len(dino_order_attempts) == 1:
                    counterfactual_audit = {
                        **public,
                        "source": (
                            "dino_top1_same_anchor_reuse"
                            if public["source"]
                            == "dino_order_geometry_attempt_reuse"
                            else "dino_top1_counterfactual"),
                    }
                if public["accepted"]:
                    accepted_dino_order_attempt = public
                    break
            counterfactual_dino_order_audit = {
                "accepted": accepted_dino_order_attempt is not None,
                "selected_anchor": (
                    accepted_dino_order_attempt["selected_anchor"]
                    if accepted_dino_order_attempt is not None else None),
                "selected_dino_rank": (
                    accepted_dino_order_attempt["selected_dino_rank"]
                    if accepted_dino_order_attempt is not None else None),
                "attempt_count": len(dino_order_attempts),
                "attempts": dino_order_attempts,
                "action_authority": False,
            }
        learned_proposal = None
        ranker = getattr(backend, "cdec_pairwise_ranker", None)
        if proposal_order != "geometry_first":
            learned_proposal = {
                "status": "not_applicable_semantic_first",
                "activation_authorized": False,
            }
        elif ranker is not None and not allow_learned_rescue:
            learned_proposal = {
                "status": "not_requested",
                "activation_authorized": False,
            }
        elif ranker is not None:
            if geometry_attempt["accepted"]:
                learned_proposal = {
                    "status": "not_evaluated_geometry_accepted",
                    "activation_authorized": False,
                }
            else:
                try:
                    learned_proposal = backend._cdec_pairwise_proposal(
                        goal_path, canonical)
                    learned_anchor = int(
                        learned_proposal["selected_anchor"])
                    if learned_anchor == geometry_attempt["selected_anchor"]:
                        learned_proposal["status"] = (
                            "same_anchor_certificate_reused")
                        proposal_attempts.append({
                            **public_attempt(geometry_attempt),
                            "source": "learned_same_anchor_reuse",
                        })
                    else:
                        learned_selected = evidence_by_anchor.get(
                            learned_anchor)
                        if learned_selected is None:
                            raise RuntimeError(
                                "learned anchor escaped the frozen shortlist")
                        learned_attempt = attempt_proposal(
                            learned_selected, "learned_on_geometry_reject")
                        proposal_attempts.append(public_attempt(learned_attempt))
                        if learned_attempt["accepted"]:
                            final_attempt = learned_attempt
                        learned_proposal["status"] = (
                            "certificate_accepted"
                            if learned_attempt["accepted"]
                            else "certificate_rejected")
                except Exception as exception:
                    learned_proposal = {
                        "status": "runtime_exception_fail_closed",
                        "error": f"{type(exception).__name__}: {exception}",
                        "activation_authorized": False,
                    }

        selected = final_attempt["selected"]
        selected_anchor = final_attempt["selected_anchor"]
        possible = final_attempt["possible"]
        precheck_reason = final_attempt["precheck_reason"]
        pnp = final_attempt["pnp"]
        certificate = final_attempt["certificate"]
        accepted = final_attempt["accepted"]
        goal_pose9 = final_attempt["goal_pose9"]
        bearing_vector = None
        guidance_diagnostics = {}
        if accepted:
            direct_bearing = backend._certified_bearing_vector(goal_pose9)
            view_alignment = backend._certified_view_alignment(goal_pose9)
            if guidance_mode == "episodic_path_field":
                bearing_vector, guidance_diagnostics = (
                    backend._certified_path_field_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                        goal_pose9=goal_pose9,
                    ))
            elif guidance_mode == "action_coordinate_compass":
                bearing_vector, guidance_diagnostics = (
                    backend._certified_action_coordinate_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                    ))
            elif guidance_mode == "se2_route_compass":
                bearing_vector, guidance_diagnostics = (
                    backend._certified_se2_route_direction(
                        goal_key=goal_key,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                    ))
            elif guidance_mode == "monocular_route_tangent":
                route_anchor_record = backend._certified_anchor_image_record(
                    selected_anchor)
                bearing_vector, guidance_diagnostics = (
                    backend._certified_monocular_route_tangent_direction(
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
                    ))
            else:
                bearing_vector, guidance_diagnostics = (
                    backend._certified_graph_direction(
                        goal_key=goal_key,
                        direct_bearing=direct_bearing,
                        target_anchor=selected_anchor,
                        goal_start_frame=int(goal_start),
                        route_start_anchor=route_start_anchor,
                        graph_rescue=graph_rescue,
                    ))
        uncached_ms = 1000.0 * (time.perf_counter() - started)
        result = {
            **base,
            "ok": True,
            "accepted": accepted,
            "reason": final_attempt["reason"],
            "certificate": certificate,
            "authority": final_attempt["authority"],
            "selected_anchor": selected_anchor,
            "selected_dino_rank": selected.get("dino_rank"),
            "selected_proposal_source": final_attempt["source"],
            "proposal_attempts": proposal_attempts,
            "counterfactual_dino_top1_audit": counterfactual_audit,
            "counterfactual_dino_order_audit": (
                counterfactual_dino_order_audit),
            "learned_proposal": learned_proposal,
            "candidate_count": len(canonical),
            "ranked_candidates": ranked,
            "pnp": pnp,
            "reference_depth_cache": final_attempt[
                "reference_depth_cache"],
            "cached": False,
            "uncached_relocalization_ms": uncached_ms,
            "relocalization_ms": uncached_ms,
        }
        if accepted:
            anchor_record = backend._certified_anchor_image_record(
                selected_anchor)
            result.update(
                aux_pose=bearing_vector,
                direction_vector=bearing_vector,
                pointgoal_units="lingbot_raw_direction_only",
                metric_scale=None,
                selected_anchor_image_sha256=anchor_record["sha256"],
                **view_alignment,
                **guidance_diagnostics,
            )
        cache_result = dict(result)
        cache_result.pop("frame_idx", None)
        for key in tuple(cache_result):
            if (key.startswith("episodic_path_field_")
                    or key.startswith("action_coordinate_")
                    or key.startswith("se2_route_")
                    or key.startswith("local_tangent_")):
                cache_result.pop(key)
        # Bearing is current-relative and must be recomputed after motion.
        cache_result["aux_pose"] = None
        self.certificates[goal_key] = {
            "candidate_fingerprint": fingerprint,
            "goal_pose9": (goal_pose9.tolist() if accepted else None),
            "result": cache_result,
        }
        return result

    @torch.no_grad()
    def bearing(self, goal_pose9):
        """Convert a certified pose to scale-free ``[forward, left]``.

        LingBot translation scale is monocular and is not certified by the v2
        image-geometry checks.  The relative direction is scale invariant, so
        the runtime boundary deliberately does not call ``_get_metric_scale``.
        ``verified_bearing_v1`` performs the only metric operation later: a
        frozen projection to its already validated 2.5 m controller radius.
        """
        backend = self.backend
        from MemNavData.certified_relocalization_runtime import (
            scale_free_relative_xy,
        )

        pose = np.asarray(goal_pose9, dtype=np.float64)
        if pose.shape != (9,) or not np.isfinite(pose).all():
            raise ValueError("certified goal pose must be finite pose9")
        current_pose = self.poses[-1].float().cpu().numpy()
        return scale_free_relative_xy(current_pose, pose)

    @torch.no_grad()
    def view_alignment(self, goal_pose9):
        """Return PnP-derived terminal view residuals, never a STOP decision."""
        backend = self.backend
        from MemNavData.goat_terminal_alignment import (
            relative_optical_yaw_pitch_deg,
        )

        pose = np.asarray(goal_pose9, dtype=np.float64)
        if pose.shape != (9,) or not np.isfinite(pose).all():
            raise ValueError("certified goal pose must be finite pose9")
        current_pose = self.poses[-1].float().cpu().numpy()
        yaw_right, pitch_up = relative_optical_yaw_pitch_deg(
            current_pose, pose)
        return {
            "terminal_yaw_right_deg": float(yaw_right),
            "terminal_pitch_up_deg": float(pitch_up),
            "terminal_alignment_source": (
                "certified_lingbot_current_to_pnp_goal_rotation"),
            "terminal_alignment_stop_authority": False,
        }

    def anchor_record(self, anchor):
        """Read one immutable causal RGB anchor and bind it to a digest."""
        backend = self.backend
        if isinstance(anchor, bool) or not isinstance(anchor, (int, np.integer)):
            raise ValueError("certified anchor must be an integer")
        anchor = int(anchor)
        if anchor < int(backend.S) or anchor >= int(self.frame_count):
            raise ValueError(
                f"certified anchor {anchor} outside [{backend.S}, {self.frame_count - 1}]")
        path = os.path.join(self.rgb_dir, f"{anchor}.jpg")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as stream:
            image = stream.read()
        if not image:
            raise ValueError("certified anchor image is empty")
        return {
            "anchor": anchor,
            "image": image,
            "sha256": hashlib.sha256(image).hexdigest(),
        }

    def read_anchor_image(
            self, goal_jpg_bytes, selected_anchor, *, expected_sha256):
        """Return only the history JPEG authorized by a cached CEC proof.

        Goal bytes select the immutable certificate cache entry. A rejected,
        absent, or mismatched proof cannot become a generic memory-image read.
        """
        backend = self.backend
        if (not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or expected_sha256 != expected_sha256.lower()):
            raise ValueError("expected anchor SHA-256 is invalid")
        try:
            int(expected_sha256, 16)
        except ValueError as exc:
            raise ValueError("expected anchor SHA-256 is invalid") from exc
        if isinstance(selected_anchor, bool) or not isinstance(
                selected_anchor, (int, np.integer)):
            raise ValueError("selected anchor must be an integer")
        selected_anchor = int(selected_anchor)
        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        cached = self.certificates.get(goal_key)
        if not isinstance(cached, dict):
            raise ValueError("no cached CEC proof for this goal")
        result = cached.get("result")
        if not isinstance(result, dict) or result.get("accepted") is not True:
            raise ValueError("CEC proof did not authorize a history anchor")
        if result.get("selected_anchor") != selected_anchor:
            raise ValueError("requested anchor differs from the certified anchor")
        goal_start = self.goal_start_frames.get(goal_key)
        if goal_start is None or not int(backend.S) <= selected_anchor < int(goal_start):
            raise ValueError("certified anchor violates the causal goal boundary")
        record = backend._certified_anchor_image_record(selected_anchor)
        if record["sha256"] != expected_sha256:
            raise ValueError("certified anchor image digest changed")
        if result.get("selected_anchor_image_sha256") != expected_sha256:
            raise ValueError("anchor digest is not bound to the cached CEC proof")
        return record

    @torch.no_grad()
    def retrieve(
            self, goal_jpg_bytes, goal_key, frame_index,
            candidate_ceiling):
        """Build a certificate shortlist before the learned decoder is warm.

        The learned MemNav decoder requires ``S + W`` streamed frames, but
        Certified Episodic Compass does not consume that decoder. Its DINO
        shortlist and anchor-depth replay are valid once the eight-frame
        LingBot scale block has produced dense CLS features and camera poses.
        Keeping the decoder warm-up on this path silently rejects short GOAT
        revisits even when they have a valid causal history.
        """
        backend = self.backend

        frame_index = int(frame_index)
        candidate_ceiling = int(candidate_ceiling)
        if not backend._has_frozen_visual_relocalizer():
            return [], None
        cache_key = (goal_key, candidate_ceiling)
        dense_cls = getattr(backend, "dino_cls", ())
        if (frame_index < backend.S - 1
                or len(dense_cls) != self.frame_count
                or frame_index >= len(dense_cls)):
            frozen = self.shortlists.setdefault(cache_key, [])
            return [dict(item) for item in frozen], None

        goal_path = os.path.join(
            self.rgb_dir, "_cert_shortlist_goal_{}.jpg".format(goal_key))
        if not os.path.isfile(goal_path):
            with open(goal_path, "wb") as handle:
                handle.write(goal_jpg_bytes)
        goal_cls = self.goal_embeddings_and_poses.get(("cls", goal_key))
        if goal_cls is None:
            goal_image = backend.lb.load_images([goal_path])[0][None].to(
                backend.device)
            goal_cls = backend.lb.dino(goal_image)["cls"]
            self.goal_embeddings_and_poses[("cls", goal_key)] = goal_cls

        memory_cls = self.retrieval_descriptors(dense_cls)
        visual_cosine = torch.nn.functional.cosine_similarity(
            goal_cls.unsqueeze(1), memory_cls, dim=-1)[0]
        current_goal_cosine = float(visual_cosine[frame_index].item())
        candidates = self.shortlist_from_scores(
            goal_key, candidate_ceiling, frame_index, visual_cosine)
        return candidates, current_goal_cosine

    def retrieval_descriptors(self, descriptors):
        """Original contiguous full-history input; episodic storage may reuse it."""
        return torch.stack(descriptors, 0)[None].to(self.backend.device)

    def shortlist_from_scores(
            self, goal_key, candidate_ceiling, frame_index, visual_cosine):
        """One frozen DINO shortlist for both short and warmed-up histories."""
        from MemNavData.certified_relocalization_runtime import (
            CERTIFIED_CANDIDATE_MIN_GAP,
            CERTIFIED_CANDIDATE_TOP_K,
            CERTIFIED_MINIMUM_ANCHOR,
        )

        cache_key = (goal_key, candidate_ceiling)
        cached = self.shortlists.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        eligible = torch.zeros(
            frame_index + 1, dtype=torch.bool, device=visual_cosine.device)
        high = min(frame_index - 1, candidate_ceiling)
        if high >= CERTIFIED_MINIMUM_ANCHOR:
            eligible[CERTIFIED_MINIMUM_ANCHOR:high + 1] = True
        candidates = temporal_nms_candidates(
            visual_cosine.detach().float().cpu().tolist(),
            eligible.detach().cpu().tolist(),
            top_k=CERTIFIED_CANDIDATE_TOP_K,
            min_frame_gap=CERTIFIED_CANDIDATE_MIN_GAP,
        )
        self.shortlists[cache_key] = [
            dict(item) for item in candidates]
        return [dict(item) for item in candidates]

    def clear_goal(self, goal_key):
        """Forget one goal session without touching causal visual history.

        Anchor depth is intentionally retained because it is a property of an
        immutable history frame, not of a query.  Every cache listed below is
        query-bound through a goal hash, candidate ceiling, or sticky action
        decision and must be recomputed if the same image becomes a later goal.
        """
        backend = self.backend
        goal_starts = getattr(backend, "_goal_start_frame", None)
        if goal_starts is not None:
            goal_starts.pop(goal_key, None)
        cache_names = (
            "_goal_cache", "_anchor_state", "_graph_routes",
            "_retrieval_verification_cache", "_phase_b_rank_cache",
            "_phase_b_scale_cache", "_phase_b_geometry_cache",
            "_certified_relocalization_cache", "_pi3x_relocalization_cache",
            "_certified_graph_routes", "_certified_path_field_routes",
            "_certified_action_coordinate_routes",
            "_certified_se2_route_compasses",
            "_certified_monocular_route_tangents",
            "_certified_candidate_cache",
        )
        for name in cache_names:
            mapping = getattr(backend, name, None)
            if mapping is None:
                continue
            stale = [
                key for key in mapping
                if backend._cache_key_contains_goal(key, goal_key)
            ]
            for key in stale:
                del mapping[key]
        # Dense query depths are transient evidence for the active goal.  The
        # fixed-stride causal tape remains in the persistent route cache.
        backend._certified_route_live_depth_cache.clear()

    def begin_goal(self, goal_key):
        """Open a contiguous goal session while preserving lifelong memory."""
        backend = self.backend
        goal_key = str(goal_key)
        active_goal_key = getattr(backend, "_active_goal_key", None)
        switched = goal_key != active_goal_key
        if switched:
            if active_goal_key is not None:
                backend._clear_goal_conditioned_state(active_goal_key)
            # Defensive cleanup makes a repeated A->B->A query independent of
            # any legacy A cache that predates this lifecycle contract.
            backend._clear_goal_conditioned_state(goal_key)
            self.active_goal_key = goal_key
            self.goal_session_index = int(
                getattr(backend, "_goal_session_index", 0)) + 1
        self.goal_session_started = bool(switched)
        return switched

    def goal_status(self):
        backend = self.backend
        return {
            "goal_session_index": int(getattr(
                backend, "_goal_session_index", 0)),
            "goal_session_started": bool(getattr(
                backend, "_last_goal_session_started", False)),
            "long_term_memory_preserved": True,
        }

    def replay_goal_session(self, goal_jpg_bytes, expected_start_frame):
        """Restore a frozen query boundary without appending or planning.

        Paired lifelong evaluations replay an already executed C trajectory
        before branching at B2.  Replaying RGB frames alone is insufficient:
        the original C query also opened a goal session at the first C frame.
        This method restores only that lifecycle boundary.  It deliberately
        performs no retrieval, certificate inference, or controller action.
        """
        backend = self.backend
        import hashlib

        expected_start_frame = int(expected_start_frame)
        if expected_start_frame != int(self.frame_count):
            raise ValueError(
                "replayed goal session does not start at the current frame")
        goal_key = hashlib.md5(goal_jpg_bytes).hexdigest()
        switched = backend._begin_goal_session(goal_key)
        if not switched:
            raise ValueError("replayed goal session did not switch goals")
        goal_start_frame = self.goal_start_frames.setdefault(
            goal_key, expected_start_frame)
        if int(goal_start_frame) != expected_start_frame:
            raise ValueError("replayed goal-session boundary changed")
        return {
            **backend.goal_session_status(),
            "goal_start_frame": int(goal_start_frame),
            "candidate_ceiling": int(goal_start_frame) - 1,
            "frame_count": int(self.frame_count),
            "diffusion_sampled": False,
            "memory_appended": False,
        }
