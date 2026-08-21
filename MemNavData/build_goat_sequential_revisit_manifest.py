#!/usr/bin/env python3
"""Build an outcome-blind GOAT sequential-Revisit evaluation manifest.

The manifest is derived only from released episode task lists. It never reads
rollout outcomes, observations, retrieval scores, geometry, or method output.
For each scene, the selected episode has the earliest exact-instance Revisit
ImageGoal; SHA-256 is used only to break ties. This deliberately maximizes the
chance that a sequential policy reaches the target while preserving one
scene-balanced, pre-outcome target per scene.

Task-list recurrence is evaluator metadata, not a controller input. At run
time CEC must still establish support from the causal RGB history and abstain
when it cannot certify an anchor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "goat_sequential_revisit_manifest_v2_20260815"
DEFAULT_SALT = "goat-cec-sequential-revisit-constructibility-v1"


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_id(episode: Mapping[str, Any], content_path: pathlib.Path) -> str:
    raw = str(episode.get("scene_id", ""))
    if raw:
        name = pathlib.PurePosixPath(raw).parent.name
        if "-" in name:
            return name.split("-", 1)[1]
    # ``content/<scene>.json.gz`` is the official GOAT sharding contract.
    return content_path.name[:-8]


def repeated_image_subtasks(
        tasks: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Return ImageGoals whose exact instance appeared in an earlier task."""
    prior: Dict[str, List[Dict[str, Any]]] = {}
    repeated = []
    for index, task in enumerate(tasks):
        if len(task) < 3:
            continue
        modality = str(task[1])
        instance = task[2]
        if instance is None:
            continue
        instance_id = str(instance)
        if modality == "image" and instance_id in prior:
            repeated.append({
                "subtask_index": int(index),
                "instance_id": instance_id,
                "prior_instance_subtasks": list(prior[instance_id]),
            })
        prior.setdefault(instance_id, []).append({
            "subtask_index": int(index),
            "modality": modality,
            "instance_id": instance_id,
        })
    return repeated


def build_manifest(
        split_root: pathlib.Path,
        excluded_scenes: Iterable[str] = (),
        included_scenes: Optional[Iterable[str]] = None,
        salt: str = DEFAULT_SALT,
        engineering_smoke: bool = False) -> Dict[str, Any]:
    split_root = pathlib.Path(split_root)
    content_root = split_root / "content"
    paths = sorted(content_root.glob("*.json.gz"))
    if not paths:
        raise ValueError("no GOAT content shards under {}".format(content_root))
    excluded = {str(value) for value in excluded_scenes}
    included = (None if included_scenes is None else
                {str(value) for value in included_scenes})
    if included is not None and excluded.intersection(included):
        raise ValueError("a scene cannot be both included and excluded")

    episode_count = 0
    image_goal_count = 0
    repeated_target_count = 0
    eligible_episode_count = 0
    candidates_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    content_receipts = []
    for path in paths:
        digest = _sha256_file(path)
        content_receipts.append((path.name, digest))
        with gzip.open(str(path), "rt") as handle:
            payload = json.load(handle)
        for episode in payload.get("episodes", []):
            episode_count += 1
            tasks = episode.get("tasks", [])
            image_goal_count += sum(
                1 for task in tasks
                if len(task) >= 2 and str(task[1]) == "image")
            repeated = repeated_image_subtasks(tasks)
            repeated_target_count += len(repeated)
            if not repeated:
                continue
            eligible_episode_count += 1
            scene = _scene_id(episode, path)
            first = repeated[0]
            episode_id = str(episode["episode_id"])
            tie_break = hashlib.sha256(
                "{}|{}|{}".format(salt, scene, episode_id).encode("utf-8")
            ).hexdigest()
            candidates_by_scene.setdefault(scene, []).append({
                "scene_id": scene,
                "episode_id": episode_id,
                "target_subtask_index": first["subtask_index"],
                "target_instance_id": first["instance_id"],
                "prior_instance_subtasks": first[
                    "prior_instance_subtasks"],
                "all_repeated_image_subtask_indices": [
                    item["subtask_index"] for item in repeated],
                "task_count": len(tasks),
                "source_content_shard": path.name,
                "source_content_sha256": digest,
                "selection_tie_break_sha256": tie_break,
            })

    all_eligible_scenes = sorted(candidates_by_scene)
    if included is not None:
        absent = sorted(included.difference(all_eligible_scenes))
        if absent:
            raise ValueError("included scenes are not eligible: {}".format(absent))
    selected = []
    for scene in all_eligible_scenes:
        if scene in excluded or (included is not None and scene not in included):
            continue
        candidates = candidates_by_scene[scene]
        # Outcome-blind: only task-list position and salted identity are used.
        winner = min(candidates, key=lambda item: (
            item["target_subtask_index"],
            item["selection_tie_break_sha256"],
        ))
        selected.append(dict(winner))
    selected.sort(key=lambda item: item["scene_id"])
    for index, item in enumerate(selected):
        item["index"] = index
        item["arm_order"] = (
            ["native", "cec"] if index % 2 == 0 else ["cec", "native"])

    receipt_digest = hashlib.sha256()
    for name, digest in content_receipts:
        receipt_digest.update(name.encode("utf-8"))
        receipt_digest.update(b"\0")
        receipt_digest.update(digest.encode("ascii"))
        receipt_digest.update(b"\n")
    aggregate = split_root / "val_unseen.json.gz"
    aggregate_sha = _sha256_file(aggregate) if aggregate.is_file() else None
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "val_unseen",
        "purpose": (
            "consumed-scene engineering smoke"
            if engineering_smoke else
            "scene-balanced held-out sequential-Revisit external evaluation"),
        "evaluation_stage": (
            "engineering_smoke" if engineering_smoke else
            "formal_targeted_external_evaluation"),
        "is_full_goat_benchmark_score": False,
        "paper_claim_authorized": not engineering_smoke,
        "paper_claim_scope": (
            None if engineering_smoke else
            "targeted GOAT val_unseen sequential-Revisit evaluation; not a "
            "full GOAT benchmark score"),
        "method_or_threshold_selection_allowed": False,
        "controller_reads_target_metadata": False,
        "target_metadata_use": "evaluator stopping and stratification only",
        "selection_rule": {
            "target": (
                "first exact-instance ImageGoal preceded by any "
                "instance-specific task"),
            "episode_per_scene": (
                "minimum target_subtask_index, with salted SHA256 identity "
                "tie-break; no rollout or method output is read"),
            "salt": salt,
            "excluded_scenes": sorted(excluded),
            "included_scenes": (None if included is None else sorted(included)),
        },
        "source_population": {
            "content_scenes": len(paths),
            "episodes": episode_count,
            "image_goals": image_goal_count,
            "exact_repeated_image_targets": repeated_target_count,
            "episodes_with_exact_recurrence": eligible_episode_count,
            "scenes_with_exact_recurrence": len(all_eligible_scenes),
            "selected_scenes": len(selected),
        },
        "dataset_receipt": {
            "aggregate_index_sha256": aggregate_sha,
            "content_shard_count": len(content_receipts),
            "sorted_content_receipts_sha256": receipt_digest.hexdigest(),
        },
        "analysis_contract": {
            "base_seed": 100,
            "maximum_steps_per_arm": 5000,
            "primary_population": (
                "all selected episodes, intention-to-treat; no filtering by "
                "target reach, prior success, certificate support, or outcome"),
            "primary_outcome": "success of the frozen repeated ImageGoal target",
            "primary_comparison": "role-free CEC minus released official GOAT",
            "arm_order": (
                "manifest-frozen alternating order after scene-id sort; "
                "17 native-first and 17 CEC-first in the 34-scene formal set"),
            "semantic_stop_contract": (
                "official GOAT SUBTASK_STOP is executed exactly; CEC may only "
                "override non-stop motion after certificate acceptance; no "
                "terminal U-turn or orientation alignment"),
            "paired_test": "two-sided exact McNemar",
            "effect_interval": (
                "scene-cluster percentile bootstrap risk-difference interval"),
            "constructibility_diagnostics_only": [
                "target entered by each arm",
                "prior exact-instance task success",
                "target certificate candidate/accept coverage",
                "non-recurrent pre-target certificate takeover",
            ],
            "minimum_mechanistic_coverage_for_interpretation": {
                "paired_episodes_entering_target": 20,
                "distinct_scenes": 12,
            },
            "post_result_method_or_threshold_changes_forbidden": True,
        },
        "episodes": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=pathlib.Path, required=True)
    parser.add_argument("--exclude-scene", action="append", default=[])
    parser.add_argument("--include-scene", action="append")
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest(
        args.split_root, args.exclude_scene, args.include_scene, args.salt,
        args.engineering_smoke)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
