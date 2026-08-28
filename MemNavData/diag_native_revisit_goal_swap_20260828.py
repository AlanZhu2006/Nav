"""Goal-swap probe: does the revisit goal image influence mono-native NavDP?

Label: POST-HOC DIAGNOSTIC (plan-level, open-loop) on the consumed fresh
HM3D full-mono mixed-role population. No SR claim.

For each of the 28 evaluated histories, this probe rebuilds the mono-native
state at the A-end (exact causal RGB replay into the MemNav sidecar for
LingBot depth state; the recorded replay decision frames into frozen
NavDP's 8-frame FIFO via ``/memory_replay_step``), then requests read-only
candidate resamples (``/imagegoal_resample``, which asserts FIFO
non-mutation) under four goal variants:

  a. the history's own Revisit goal image, formal first-plan seed S;
  b. a donor history's Revisit goal image (different scene), same seed S;
  c. the correct goal image, seed S+1  (ordinary diffusion variation);
  d. the history's paired Novel goal image, seed S.

Same seed + same FIFO + same depth transaction mean any (a) vs (b)
difference is caused only by the goal image. If swap divergence (a,b) is
of the same order as seed noise (a,c), NavDP's goal conditioning carries
no usable information at the query state.

State fidelity note: the formal query's first plan appended a fresh A-end
render via ``/imagegoal_step``; the probe instead ends the FIFO with the
last causal history frame and resamples without appending. All variants
share this state exactly, so the paired contrast is unaffected.

Run through ``run_native_goal_swap_probe_local.sh`` (starts the server
pair), or against already-running servers:

    python MemNavData/diag_native_revisit_goal_swap_20260828.py \
        --pulled-root .diagnostics/hm3d_fresh_fullmono_mixed_role_20260820/pulled_20260828 \
        --out-dir .diagnostics/native_goal_swap_probe_20260828 \
        --memnav_port 23140 --navdp_port 23141
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

VARIANTS = ("correct", "swapped", "seed_shift", "novel_goal")


# ---------------------------------------------------------------- pure helpers

def donor_assignment(labels: list[str]) -> dict[str, str]:
    """Map each history label to a donor with a different scene (cyclic)."""
    scenes = {label: label.split("_", 2)[1] for label in labels}
    donors = {}
    n = len(labels)
    for i, label in enumerate(labels):
        for offset in range(1, n):
            candidate = labels[(i + offset) % n]
            if scenes[candidate] != scenes[label]:
                donors[label] = candidate
                break
        else:
            raise ValueError("no donor with a different scene exists")
    return donors


def rms_divergence(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def endpoint_heading_deg(trajectory) -> float:
    """Heading of the selected trajectory endpoint in the robot frame."""
    arr = np.asarray(trajectory, dtype=np.float64)
    while arr.ndim > 2:  # server returns a leading batch dimension
        arr = arr[0]
    end = arr[-1]
    return float(math.degrees(math.atan2(end[1], end[0])))


def heading_delta_deg(t_a, t_b) -> float:
    delta = endpoint_heading_deg(t_a) - endpoint_heading_deg(t_b)
    return float(abs((delta + 180.0) % 360.0 - 180.0))


def probe_contrasts(responses: dict[str, dict]) -> dict:
    """Per-history divergence metrics between the goal variants."""
    ref = responses["correct"]
    out = {}
    for name in ("swapped", "seed_shift", "novel_goal"):
        row = responses[name]
        out[name] = {
            "candidate_rms": rms_divergence(
                ref["all_trajectory"], row["all_trajectory"]),
            "selected_heading_delta_deg": heading_delta_deg(
                ref["trajectory"], row["trajectory"]),
            "selected_rms": rms_divergence(
                ref["trajectory"], row["trajectory"]),
            "critic_max_delta": float(abs(
                np.max(np.asarray(ref["all_values"], dtype=np.float64))
                - np.max(np.asarray(row["all_values"], dtype=np.float64)))),
        }
    noise = out["seed_shift"]["candidate_rms"]
    out["swap_over_seed_rms_ratio"] = (
        out["swapped"]["candidate_rms"] / noise if noise > 0 else None)
    return out


def aggregate_probe(per_history: list[dict]) -> dict:
    ratios = [row["contrasts"]["swap_over_seed_rms_ratio"]
              for row in per_history
              if row["contrasts"]["swap_over_seed_rms_ratio"] is not None]
    swap_heading = [row["contrasts"]["swapped"]["selected_heading_delta_deg"]
                    for row in per_history]
    seed_heading = [row["contrasts"]["seed_shift"][
        "selected_heading_delta_deg"] for row in per_history]

    def stats(values):
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "iqr": [float(np.percentile(arr, 25)),
                    float(np.percentile(arr, 75))],
            "mean": float(arr.mean()),
        }

    return {
        "swap_over_seed_rms_ratio": stats(ratios),
        "histories_with_ratio_above_1": int(sum(r > 1.0 for r in ratios)),
        "swapped_selected_heading_delta_deg": stats(swap_heading),
        "seed_shift_selected_heading_delta_deg": stats(seed_heading),
        "interpretation_rule": (
            "ratio ~= 1 and swapped heading delta ~= seed-shift heading "
            "delta => the goal image exerts no systematic influence at the "
            "query state; ratio >> 1 => conditioning is active"),
    }


# ---------------------------------------------------------------- HTTP client

def run_history(session, label: str, pulled_root: Path, donors: dict,
                memnav_base: str, navdp_base: str,
                intrinsic: list, camera_height: float) -> dict:
    _, scene, episode = label.split("_", 2)
    eval_dir = pulled_root / "evaluation_natural_direction" / label
    plans = json.loads(
        (eval_dir / "mono_native"
         / f"{episode}_pair_00_revisit_plans.json").read_text())
    decision_frames = set(int(x) for x in plans["replay"]["decision_steps"])
    seed = plans["query_leg"][0].get("requested_diffusion_seed")
    if seed is None:
        raise RuntimeError(f"{label}: formal first-plan seed missing")
    seed = int(seed)

    rgb_dir = pulled_root / "online_a" / label / "rgb"
    frames = sorted(rgb_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"{label}: no RGB frames pulled")

    bench = pulled_root / "benchmarks" / "natural_direction" / scene / episode
    goals = {
        "correct": (bench / "pair_00/revisit/goal.jpg").read_bytes(),
        "seed_shift": (bench / "pair_00/revisit/goal.jpg").read_bytes(),
        "novel_goal": (bench / "pair_00/novel/goal.jpg").read_bytes(),
    }
    donor = donors[label]
    _, donor_scene, donor_episode = donor.split("_", 2)
    goals["swapped"] = (
        pulled_root / "benchmarks" / "natural_direction" / donor_scene
        / donor_episode / "pair_00/revisit/goal.jpg").read_bytes()

    # --- reset both servers. Reset seeds must fit numpy's 32-bit range;
    # the formal 64-bit plan seed travels only as the diffusion_seed field.
    reset_seed = seed % (2 ** 32)
    reset = session.post(f"{memnav_base}/navigator_reset", json={
        "camera_height": camera_height,
        "camera_intrinsic": intrinsic,
        "seed": reset_seed,
        "episode_len": len(frames) + 8,
    })
    reset.raise_for_status()
    contract = reset.json().get("monocular_depth")
    if (not isinstance(contract, dict) or contract.get("enabled") is not True
            or contract.get("depth_contract") != "raw_lingbot_depth_first40_v1"
            or contract.get("metric_depth_sensor_consumed") is not False):
        raise RuntimeError(f"{label}: mono depth contract missing: {contract}")
    reset = session.post(f"{navdp_base}/navigator_reset", json={
        "intrinsic": intrinsic,
        "stop_threshold": -0.5,
        "batch_size": 1,
        "depth_source": "monocular_sidecar",
        "seed": reset_seed,
    })
    reset.raise_for_status()

    # --- causal replay
    token = frame_index = None
    for idx, frame_path in enumerate(frames):
        frame = frame_path.read_bytes()
        last = idx == len(frames) - 1
        step = session.post(
            f"{memnav_base}/memory_step",
            files={"image": ("image.jpg", frame)},
            data={"materialize_monocular_depth": "1" if last else "0"},
        )
        step.raise_for_status()
        receipt = step.json()
        if idx in decision_frames or last:
            replay = session.post(
                f"{navdp_base}/memory_replay_step",
                files={"image": ("image.jpg", frame)},
            )
            replay.raise_for_status()
        if last:
            token = receipt.get("monocular_depth_transaction_token")
            frame_index = receipt.get("monocular_depth_frame_index")
            if not token or frame_index != receipt.get("frame_idx"):
                raise RuntimeError(f"{label}: invalid depth transaction")

    # --- read-only goal-variant resamples on the identical state
    current = frames[-1].read_bytes()
    responses = {}
    queue_hashes = set()
    for name in VARIANTS:
        variant_seed = seed + 1 if name == "seed_shift" else seed
        resample = session.post(
            f"{navdp_base}/imagegoal_resample",
            files={"image": ("image.jpg", current),
                   "goal": ("goal.jpg", goals[name])},
            data={"diffusion_seed": str(variant_seed),
                  "monocular_depth_transaction_token": token,
                  "monocular_depth_frame_index": str(frame_index)},
        )
        resample.raise_for_status()
        out = resample.json()
        if out.get("memory_mutated") is not False:
            raise RuntimeError(f"{label}/{name}: resample mutated memory")
        queue_hashes.add(json.dumps(out.get("queue_hashes_after")))
        responses[name] = out
    if len(queue_hashes) != 1:
        raise RuntimeError(f"{label}: FIFO state drifted across variants")

    return {
        "label": label,
        "scene": scene,
        "episode": episode,
        "donor": donor,
        "seed": seed,
        "replayed_frames": len(frames),
        "navdp_decision_frames_replayed": sorted(decision_frames),
        "contrasts": probe_contrasts(responses),
        "selected_headings_deg": {
            name: endpoint_heading_deg(responses[name]["trajectory"])
            for name in VARIANTS},
    }


def main() -> None:
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulled-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--memnav_port", type=int, default=23140)
    parser.add_argument("--navdp_port", type=int, default=23141)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--limit", type=int, default=0,
                        help="probe only the first N histories (0 = all)")
    args = parser.parse_args()

    labels = sorted(
        p.name for p in
        (args.pulled_root / "evaluation_natural_direction").iterdir()
        if p.is_dir())
    donors = donor_assignment(labels)
    if args.limit:
        labels = labels[:args.limit]

    # Exact K from the source-generation parquet (observation.camera_intrinsic
    # of the fresh HM3D render contract: 480x270, hfov 68 deg); fy differs
    # from the square-pixel derivation, so never re-derive it.
    intrinsic = json.loads(
        (args.pulled_root / "camera_intrinsic.json").read_text())
    receipt = json.loads(
        (args.pulled_root / "online_a" / labels[0]
         / "receipt.json").read_text())
    camera_height = float(receipt.get("camera_height_m", 0.5))

    memnav_base = f"http://{args.host}:{args.memnav_port}"
    navdp_base = f"http://{args.host}:{args.navdp_port}"
    session = requests.Session()

    per_history = []
    for i, label in enumerate(labels):
        start = time.time()
        row = run_history(session, label, args.pulled_root, donors,
                          memnav_base, navdp_base, intrinsic, camera_height)
        row["wall_seconds"] = round(time.time() - start, 1)
        per_history.append(row)
        c = row["contrasts"]
        print(f"[{i + 1}/{len(labels)}] {label} "
              f"swap_rms={c['swapped']['candidate_rms']:.4f} "
              f"seed_rms={c['seed_shift']['candidate_rms']:.4f} "
              f"ratio={c['swap_over_seed_rms_ratio']:.3f} "
              f"({row['wall_seconds']}s)", flush=True)

    report = {
        "schema": "native_revisit_goal_swap_probe_v1_20260828",
        "analysis_label": "posthoc_diagnostic_plan_level",
        "variants": list(VARIANTS),
        "aggregate": aggregate_probe(per_history),
        "per_history": per_history,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "goal_swap_probe.json"
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out_path}")
    print(json.dumps(report["aggregate"], indent=1))


if __name__ == "__main__":
    main()
