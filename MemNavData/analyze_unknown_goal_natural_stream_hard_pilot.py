#!/usr/bin/env python3
"""Audit frozen single-state baselines on causal natural-stream labels.

This analysis never uses A/B/C role as an input to a learned decision.  The
role is retained only as an audit stratum.  It reconstructs the already-frozen
geometry gate from raw candidate evidence, so the impossible collection-time
thresholds used to guarantee zero takeover cannot contaminate the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


LABEL_NAMES = {-1: "ambiguous", 0: "negative", 1: "positive"}
LEGS = ("legA", "legB", "legC")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hard_pass(candidate: Mapping[str, object], *, min_matches: int,
              min_inliers: int, min_inlier_ratio: float) -> bool:
    matches = candidate.get("matches")
    inliers = candidate.get("inliers")
    ratio = candidate.get("inlier_ratio")
    require(isinstance(matches, int) and isinstance(inliers, int)
            and isinstance(ratio, (int, float)),
            "candidate geometry evidence is malformed")
    return (matches >= min_matches and inliers >= min_inliers
            and float(ratio) >= min_inlier_ratio)


def longest_consecutive(values: Iterable[int]) -> int:
    ordered = sorted(set(values))
    best = current = 0
    previous = None
    for value in ordered:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        best = max(best, current)
        previous = value
    return best


def identity(payload: Mapping[str, object], path: Path) -> tuple[str, str]:
    inputs = payload.get("inputs")
    require(isinstance(inputs, Mapping), "teacher inputs are absent")
    episode_root = inputs.get("episode_root")
    require(isinstance(episode_root, str), "teacher episode root is absent")
    root = Path(episode_root)
    require(root.name.startswith("episode_") and bool(root.parent.name),
            "teacher episode identity is malformed")
    return root.parent.name, root.name


def analyze_teacher(path: Path, *, min_matches: int, min_inliers: int,
                    min_inlier_ratio: float) -> dict[str, object]:
    require(path.is_file() and not path.is_symlink(),
            f"teacher input is absent/symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, Mapping)
            and payload.get("status") == "complete",
            f"teacher is incomplete: {path}")
    records = payload.get("records")
    require(isinstance(records, list) and bool(records),
            f"teacher records are absent: {path}")
    scene, episode = identity(payload, path)

    rows: list[dict[str, object]] = []
    positive_anchor_plans: dict[tuple[str, int], list[int]] = defaultdict(list)
    false_support_anchor_plans: dict[tuple[str, int], list[int]] = defaultdict(list)

    for record in records:
        require(isinstance(record, Mapping), "teacher record is malformed")
        leg = record.get("leg")
        label = record.get("topk_support_label")
        candidates = record.get("candidates")
        plan_index = record.get("leg_plan_index")
        require(leg in LEGS and label in LABEL_NAMES
                and isinstance(candidates, list) and isinstance(plan_index, int),
                "teacher record fields are malformed")
        checked = []
        for candidate in candidates:
            require(isinstance(candidate, Mapping)
                    and candidate.get("label") in LABEL_NAMES
                    and isinstance(candidate.get("anchor"), int),
                    "teacher candidate is malformed")
            checked.append({
                "anchor": int(candidate["anchor"]),
                "rank": int(candidate["rank"]),
                "label": int(candidate["label"]),
                "hard_pass": hard_pass(
                    candidate,
                    min_matches=min_matches,
                    min_inliers=min_inliers,
                    min_inlier_ratio=min_inlier_ratio,
                ),
            })
        checked.sort(key=lambda item: item["rank"])
        hard_candidates = [item for item in checked if item["hard_pass"]]
        selected = hard_candidates[0] if hard_candidates else None
        top1_label = checked[0]["label"] if checked else None
        positive_anchors = [item["anchor"] for item in checked
                            if item["label"] == 1]
        false_support_anchors = [item["anchor"] for item in hard_candidates
                                 if item["label"] == 0]
        for anchor in positive_anchors:
            positive_anchor_plans[(leg, anchor)].append(plan_index)
        for anchor in false_support_anchors:
            false_support_anchor_plans[(leg, anchor)].append(plan_index)

        geometry_selected_label = selected["label"] if selected else None
        geometry_miss = label == 1 and geometry_selected_label != 1
        top1_miss = label == 1 and top1_label != 1
        strict_false_support = label == 0 and selected is not None
        rows.append({
            "scene": scene,
            "episode": episode,
            "decision_index": int(record["decision_index"]),
            "leg": leg,
            "leg_plan_index": plan_index,
            "step": int(record["step"]),
            "candidate_pool_size": len(checked),
            "teacher_label": int(label),
            "teacher_label_name": LABEL_NAMES[int(label)],
            "first_positive_rank": record.get("first_positive_rank"),
            "dino_top1_label": top1_label,
            "geometry_selected_label": geometry_selected_label,
            "geometry_selected_rank": selected["rank"] if selected else None,
            "geometry_miss_on_positive": geometry_miss,
            "dino_top1_miss_on_positive": top1_miss,
            "strict_negative_geometry_false_support": strict_false_support,
            "residual_opportunity": bool(geometry_miss or top1_miss
                                         or strict_false_support),
            "positive_anchors": positive_anchors,
            "hard_pass_anchors": [item["anchor"] for item in hard_candidates],
            "hard_pass_labels": [item["label"] for item in hard_candidates],
        })

    by_leg: dict[str, object] = {}
    for leg in LEGS:
        subset = [row for row in rows if row["leg"] == leg]
        candidate_subset = [row for row in subset
                            if row["candidate_pool_size"] > 0]
        teacher_counts = Counter(row["teacher_label_name"] for row in subset)
        positive_plans = [row for row in candidate_subset
                          if row["teacher_label"] == 1]
        negative_plans = [row for row in candidate_subset
                          if row["teacher_label"] == 0]
        positive_streak = longest_consecutive(
            int(row["leg_plan_index"]) for row in positive_plans)
        residual_streak = longest_consecutive(
            int(row["leg_plan_index"]) for row in candidate_subset
            if row["residual_opportunity"])
        positive_anchor_counts = {
            str(anchor): len(indices)
            for (anchor_leg, anchor), indices in positive_anchor_plans.items()
            if anchor_leg == leg
        }
        false_anchor_counts = {
            str(anchor): len(indices)
            for (anchor_leg, anchor), indices in false_support_anchor_plans.items()
            if anchor_leg == leg
        }
        by_leg[leg] = {
            "plans": len(subset),
            "candidate_plans": len(candidate_subset),
            "teacher_labels": {
                name: teacher_counts.get(name, 0)
                for name in ("positive", "negative", "ambiguous")
            },
            "positive_support_plans": len(positive_plans),
            "strict_negative_plans": len(negative_plans),
            "geometry_correct_positive_plans": sum(
                row["geometry_selected_label"] == 1 for row in positive_plans),
            "geometry_miss_positive_plans": sum(
                row["geometry_miss_on_positive"] for row in positive_plans),
            "dino_top1_correct_positive_plans": sum(
                row["dino_top1_label"] == 1 for row in positive_plans),
            "dino_top1_miss_positive_plans": sum(
                row["dino_top1_miss_on_positive"] for row in positive_plans),
            "strict_negative_geometry_false_support_plans": sum(
                row["strict_negative_geometry_false_support"]
                for row in negative_plans),
            "residual_opportunity_plans": sum(
                row["residual_opportunity"] for row in candidate_subset),
            "max_consecutive_positive_support_plans": positive_streak,
            "max_consecutive_residual_opportunity_plans": residual_streak,
            "max_positive_anchor_plan_count": max(
                positive_anchor_counts.values(), default=0),
            "max_false_support_anchor_plan_count": max(
                false_anchor_counts.values(), default=0),
            "positive_anchor_plan_counts": positive_anchor_counts,
            "false_support_anchor_plan_counts": false_anchor_counts,
        }

    return {
        "scene": scene,
        "episode": episode,
        "teacher_path": str(path.resolve()),
        "teacher_sha256": sha256_file(path),
        "by_leg": by_leg,
        "plan_rows": rows,
    }


def build_report(paths: list[Path], *, min_matches: int, min_inliers: int,
                 min_inlier_ratio: float) -> dict[str, object]:
    require(bool(paths), "at least one teacher input is required")
    episodes = [analyze_teacher(
        path,
        min_matches=min_matches,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
    ) for path in paths]
    identities = [(item["scene"], item["episode"]) for item in episodes]
    require(len(identities) == len(set(identities)),
            "duplicate scene/episode teacher input")

    combined = Counter()
    for episode in episodes:
        for leg in LEGS:
            summary = episode["by_leg"][leg]
            for key in (
                "plans", "candidate_plans", "positive_support_plans",
                "strict_negative_plans", "geometry_correct_positive_plans",
                "geometry_miss_positive_plans",
                "dino_top1_correct_positive_plans",
                "dino_top1_miss_positive_plans",
                "strict_negative_geometry_false_support_plans",
                "residual_opportunity_plans",
            ):
                combined[key] += int(summary[key])

    report: dict[str, object] = {
        "schema_version": "unknown_goal_natural_stream_hard_pilot_audit_v1",
        "status": "complete",
        "scope": ("frozen-baseline audit on causal shadow labels; no training, "
                  "policy intervention, SR, or deployable-method claim"),
        "geometry_reference": {
            "min_matches": min_matches,
            "min_inliers": min_inliers,
            "min_inlier_ratio": min_inlier_ratio,
            "source": "frozen deployed geometry-router thresholds",
        },
        "role_usage": ("A/B/C retained only for post-hoc audit strata; forbidden "
                       "as a learned decision feature"),
        "episodes": episodes,
        "combined": dict(sorted(combined.items())),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["content_sha256_without_self"] = hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teachers", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.5)
    args = parser.parse_args()
    require(args.min_matches >= 8 and args.min_inliers >= 0
            and 0.0 <= args.min_inlier_ratio <= 1.0,
            "geometry thresholds are invalid")
    report = build_report(
        args.teachers,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["combined"], sort_keys=True))
    print(f"COMPLETE {args.output}")


if __name__ == "__main__":
    main()
