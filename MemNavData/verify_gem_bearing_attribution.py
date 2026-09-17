"""Independently check paired prefixes and removal of later bearing authority."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from MemNavData.gem_bearing_attribution import ARMS, ROOT, dump, load, sha
from MemNavData.verify_repaired_fullmono_local import verify_rollout
from MemNavData.verify_habitat_minimal_repair import read_rows


def non_timing(value):
    """Timing, locations of artifacts and HTTP transaction IDs are not content."""
    if isinstance(value, dict):
        return {k: non_timing(v) for k, v in value.items()
                if not any(word in k.lower() for word in
                           ("_ms", "seconds", "elapsed", "time", "transaction", "artifact", "cache"))}
    if isinstance(value, list):
        return [non_timing(v) for v in value]
    return value


def verify(out):
    plan, summary = load(out / "plan.json"), load(out / "summary.json")
    assert summary["completed"] and len(summary["records"]) == 8
    assert summary["plan_sha256"] == sha(out / "plan.json")
    assert sha(out / "plan.json") == (out / "plan.sha256").read_text().strip()
    for path, digest in plan["source_sha256"].items():
        saved = out / "source_snapshot" / Path(path).relative_to(ROOT)
        assert sha(saved) == digest, str(saved)
    details, indexed = [], {}
    for row in summary["records"]:
        verified, evidence, plans, actions = verify_rollout(row)
        folder = Path(row["directory"])
        intervention = load(folder / "bearing_intervention.json")
        decisions = read_rows(folder / "bearing_decisions.jsonl")
        http = read_rows(folder / "navdp_http_receipts.jsonl")
        boundary = read_rows(folder / "memory_http_boundary.jsonl")
        poses = load(folder / "lingbot_frame_poses.json")
        query_plans = load(folder / "query_plans.json")["query_leg"]
        assert len(decisions) == len(plans)
        cell = next(c for c in plan["cells"] if c["scene"] == row["scene"])
        for item, recorded in zip(decisions, plans):
            assert item["goal_sha256"] == cell["goal_rgb_sha256"]
            assert item["image_sha256"] == recorded["receipt"]["execution_input_audit"]["image_jpeg_sha256"]
            assert item["diffusion_seed_requested"] == item["diffusion_seed_returned"]
        cutoff = intervention["cutoff_action"]
        if row["arm"] == "initial_turn_only":
            assert cutoff is not None and intervention["phase"] == "native"
            later = [p for p in plans if p["next_action_index"] >= cutoff]
            assert later and all(p["receipt"]["revisit_adapter_takeover"] is False for p in later)
            assert all(p["receipt"]["memory_controller_pointgoal"] is None for p in later)
            assert all(a["action_kind"] != "heading" for a in actions[cutoff:])
            requests = [r for r in http if r["query_active"] and r["next_action_index"] >= cutoff]
            assert requests and all(r["path"] == "/imagegoal_step" for r in requests)
            calls = [r for r in boundary if r["query_active"] and r["next_action_index"] >= cutoff]
            assert calls and not any(r["path"] in ("/certified_relocalize", "/retrieval_probe_step") for r in calls)
            assert len([d for d in decisions if d["native_after_cutoff"]]) == len(later)
        else:
            assert not any(d["native_after_cutoff"] for d in decisions)
        key = (row["scene"], row["arm"])
        assert key not in indexed
        indexed[key] = dict(row=row, verified=verified, evidence=evidence, plans=plans,
                            actions=actions, intervention=intervention, decisions=decisions,
                            http=http, poses=poses, query_plans=query_plans)
        details.append(verified)

    pairs = []
    for cell in plan["cells"]:
        full, initial = (indexed[(cell["scene"], arm)] for arm in ARMS)
        first_full, first_initial = full["plans"][0], initial["plans"][0]
        assert full["intervention"]["initial_rearward"] == initial["intervention"]["initial_rearward"]
        for field in ("position", "yaw"):
            assert first_full[field] == first_initial[field]
        for field in ("anchor", "diffusion_seed"):
            assert first_full["receipt"][field] == first_initial["receipt"][field]
        for field in ("certified_relocalization_accepted", "certified_relocalization_certificate",
                      "certified_relocalization_pnp", "router_selected_anchor"):
            assert field in full["query_plans"][0]
            assert non_timing(full["query_plans"][0][field]) == non_timing(initial["query_plans"][0][field])
        if initial["intervention"]["initial_rearward"]:
            # Every actual initial sample, turn action, and pre-cutoff geometry
            # must agree. A bitwise difference is reported as failed isolation.
            cutoff = initial["intervention"]["cutoff_action"]
            assert full["intervention"]["cutoff_action"] == cutoff
            assert first_full["receipt"]["memory_controller_pointgoal"] == first_initial["receipt"]["memory_controller_pointgoal"]
            for field in ("selected_trajectory", "all_trajectory", "all_values"):
                np.testing.assert_array_equal(first_full[field], first_initial[field])
            assert full["actions"][:cutoff] == initial["actions"][:cutoff]
            full_post = next(p for p in full["plans"] if p["next_action_index"] == cutoff)
            initial_post = next(p for p in initial["plans"] if p["next_action_index"] == cutoff)
            for field in ("position", "yaw"):
                assert full_post[field] == initial_post[field]
            for field in ("memory_frame_idx", "diffusion_seed"):
                assert full_post["receipt"][field] == initial_post["receipt"][field]
            for field in ("image_jpeg_sha256", "queue_before", "queue_after"):
                # Queue field names depend on the shared audit wrapper. The
                # full geometry and RGB prefix below is checked regardless.
                if field in full_post["receipt"]["execution_input_audit"]:
                    assert full_post["receipt"]["execution_input_audit"][field] == initial_post["receipt"]["execution_input_audit"][field]
            k = full_post["receipt"]["memory_frame_idx"]
            assert len(full["poses"]) > k and len(initial["poses"]) > k
            for pf, pi in zip(full["poses"][:k+1], initial["poses"][:k+1]):
                for field in ("frame_idx", "image_sha256", "camera_pose9", "pose_count", "motion_receipt_recorded"):
                    assert pf[field] == pi[field], (cell["scene"], pf["frame_idx"], field)
            fd = full_post["receipt"]["monocular_depth_receipt"]
            td = initial_post["receipt"]["monocular_depth_receipt"]
            for field in ("input_tensor_sha256", "output_tensor_sha256", "contract"):
                assert fd["navdp_depth_raster"][field] == td["navdp_depth_raster"][field]
            post_position = np.asarray(full_post["position"])
            distance_after_turn = float(np.linalg.norm(post_position[[0, 2]] - full["row"]["goal_xz_evaluator_only"]))
        else:
            cutoff, distance_after_turn = 0, None
        pairs.append(dict(scene=cell["scene"], initial_turn_actions=cutoff,
                          distance_after_turn_m=distance_after_turn,
                          initial_rearward=initial["intervention"]["initial_rearward"],
                          full={k: full["row"][k] for k in ("reached", "steps", "actual_path_len_m", "final_goal_dist_m", "spl")},
                          initial_only={k: initial["row"][k] for k in ("reached", "steps", "actual_path_len_m", "final_goal_dist_m", "spl")},
                          full_post_turn_plans=sum(p["next_action_index"] >= cutoff for p in full["plans"]),
                          full_explicit_turn_actions=full["verified"]["heading"]["turn_actions"],
                          full_post_turn_takeovers=sum(p["next_action_index"] >= cutoff and p["receipt"]["revisit_adapter_takeover"] is True for p in full["plans"]),
                          initial_native_decisions=initial["intervention"]["native_decisions"]))
    wins = sum(p["full"]["reached"] > p["initial_only"]["reached"] for p in pairs)
    losses = sum(p["full"]["reached"] < p["initial_only"]["reached"] for p in pairs)
    discordant = wins + losses
    p_exact = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses)+1)) / 2**discordant) if discordant else 1.
    result = dict(verified=True, histories=4, rollouts=8, pairs=pairs,
                  full_success=sum(p["full"]["reached"] for p in pairs),
                  initial_only_success=sum(p["initial_only"]["reached"] for p in pairs),
                  paired_wins=wins, paired_losses=losses, exact_mcnemar_p=p_exact,
                  scope=plan["scope"], checks=details)
    dump(out / "independent_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    verify(parser.parse_args().out.resolve())
