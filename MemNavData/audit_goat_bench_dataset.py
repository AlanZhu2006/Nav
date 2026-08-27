#!/usr/bin/env python3
"""Audit the public GOAT-Bench episode package without loading HM3D meshes.

The audit deliberately distinguishes task-list recurrence from causal visual
support.  A target instance appearing in an earlier task is a useful stratum,
but it is not proof that an online agent actually observed that instance.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any


def audit_split(split_dir: Path) -> dict[str, Any]:
    content_dir = split_dir / "content"
    files = sorted(content_dir.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no GOAT content files under {content_dir}")

    modality_counts: collections.Counter[str] = collections.Counter()
    image_by_index: collections.Counter[int] = collections.Counter()
    exact_recurrence_by_index: collections.Counter[int] = collections.Counter()
    episode_count = 0
    image_count = 0
    exact_prior_instance_count = 0
    prior_image_instance_count = 0
    prior_description_instance_count = 0
    prior_category_count = 0
    episodes_with_exact_prior_instance = 0

    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        for episode in payload["episodes"]:
            episode_count += 1
            prior_instances: set[str] = set()
            prior_image_instances: set[str] = set()
            prior_description_instances: set[str] = set()
            prior_categories: set[str] = set()
            episode_has_exact_recurrence = False

            for task_index, task in enumerate(episode["tasks"]):
                category, modality, instance_id = task[:3]
                modality_counts[modality] += 1

                if modality == "image":
                    image_count += 1
                    image_by_index[task_index] += 1
                    if instance_id in prior_instances:
                        exact_prior_instance_count += 1
                        exact_recurrence_by_index[task_index] += 1
                        episode_has_exact_recurrence = True
                    if instance_id in prior_image_instances:
                        prior_image_instance_count += 1
                    if instance_id in prior_description_instances:
                        prior_description_instance_count += 1
                    if category in prior_categories:
                        prior_category_count += 1

                if instance_id is not None:
                    prior_instances.add(instance_id)
                if modality == "image":
                    prior_image_instances.add(instance_id)
                elif modality == "description":
                    prior_description_instances.add(instance_id)
                prior_categories.add(category)

            episodes_with_exact_prior_instance += int(
                episode_has_exact_recurrence
            )

    return {
        "split": split_dir.name,
        "scene_count": len(files),
        "episode_count": episode_count,
        "subtask_count": sum(modality_counts.values()),
        "modality_counts": dict(sorted(modality_counts.items())),
        "image_subtask_count": image_count,
        "image_tasklist_exact_prior_instance_count": (
            exact_prior_instance_count
        ),
        "image_tasklist_exact_prior_instance_rate": (
            exact_prior_instance_count / image_count
        ),
        "image_with_prior_image_same_instance_count": (
            prior_image_instance_count
        ),
        "image_with_prior_description_same_instance_count": (
            prior_description_instance_count
        ),
        "image_with_prior_same_category_count": prior_category_count,
        "episodes_with_exact_prior_instance_image_query": (
            episodes_with_exact_prior_instance
        ),
        "image_subtasks_by_index": dict(sorted(image_by_index.items())),
        "exact_prior_instance_image_subtasks_by_index": dict(
            sorted(exact_recurrence_by_index.items())
        ),
        "interpretation_boundary": (
            "task-list recurrence is not causal visual support; actual online "
            "RGB history and successful/failed earlier subtasks must be audited "
            "during rollout"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="directory containing train/val_seen/val_seen_synonyms/val_unseen",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val_seen", "val_seen_synonyms", "val_unseen"),
        help="split to audit; repeat the option to audit more than one",
    )
    args = parser.parse_args()

    splits = args.split or ["val_seen", "val_seen_synonyms", "val_unseen"]
    output = {split: audit_split(args.root / split) for split in splits}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
