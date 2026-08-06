import copy
import unittest

from MemNavData.novel_candidate_set_schema_v2 import (
    CANDIDATE_FEATURE_KEYS,
    CandidateSetValidationError,
    DUSTBIN_CANDIDATE_ID,
    PRIVILEGED_FEATURE_DENY_FRAGMENTS,
    PRIVILEGED_LABEL_DENY_LIST,
    SCHEMA_VERSION,
    canonical_candidate_set_sha256,
    validate_candidate_dataset,
    validate_candidate_set,
)


SHA = "a" * 64


def features(candidate_type):
    kinds = ("native", "memory_graph", "frontier", "dustbin")
    return {
        "candidate_type_onehot": [
            float(kind == candidate_type) for kind in kinds],
        "goal_patch_relation": [0.1, 0.2],
        "goal_temporal_relation": [0.3, 0.4],
        "local_map_relation": [0.5, 0.6],
        "native_proposal_relation": [0.7, 0.8],
        "feature_presence_mask": [1.0] * 7,
        "subgoal_forward_m": 1.0,
        "subgoal_left_m": 0.1,
        "graph_path_m": 1.25,
        "graph_hops": 1.0,
        "frontier_boundary_m": 0.8,
        "frontier_novelty_m": 0.7,
        "pose_translation_p90_m": 0.2,
        "pose_yaw_p90_deg": 10.0,
        "depth_confidence_mean": 0.9,
        "clearance_lower_m": 0.4,
    }


def labels(*, advantage=0.0, useful=False):
    return {
        "geodesic_progress_h8_m": 0.2,
        "geodesic_progress_h24_m": 0.5 + advantage,
        "advantage_h24_m": advantage,
        "harm": False,
        "useful": useful,
        "reachable": True,
        "collision_h8": False,
        "regression_h24": False,
        "proposal_proxy_progress_m": advantage,
        "proposal_proxy_reachable": True,
        "proposal_proxy_positive": advantage > 0.0,
        "proposal_proxy_label_valid": True,
        "rollout_label_valid": True,
        "teacher_covisibility": 0.0,
        "covisibility_label_valid": True,
        "pose_residual_forward_m": 0.0,
        "pose_residual_left_m": 0.0,
        "pose_residual_yaw_rad": 0.0,
        "pose_label_valid": False,
    }


def candidate(candidate_id, candidate_type, *, advantage=0.0, useful=False):
    value = {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "features": features(candidate_type),
        "labels": labels(advantage=advantage, useful=useful),
    }
    if candidate_type == "native":
        value["labels"].update({
            "proposal_proxy_progress_m": 0.0,
            "proposal_proxy_reachable": False,
            "proposal_proxy_positive": False,
            "proposal_proxy_label_valid": False,
        })
    if candidate_type == "dustbin":
        for key, feature in value["features"].items():
            if key == "candidate_type_onehot":
                continue
            if isinstance(feature, list):
                value["features"][key] = [0.0] * len(feature)
            else:
                value["features"][key] = 0.0
        for key, label in value["labels"].items():
            value["labels"][key] = False if isinstance(label, bool) else 0.0
    return value


def record(*, plan_index=0, scene="scene-a", role="train",
           group=None, positive=True, episode=None,
           goal_source_episode=None, state_source="expert",
           state_id=None, goal_epoch="goal-b"):
    if episode is None:
        episode = f"{scene}/episode-0"
    if goal_source_episode is None:
        goal_source_episode = episode
    if group is None:
        group = episode
    if state_id is None:
        state_id = f"{episode}/state-{plan_index}"
    residual = candidate(
        "frontier-1", "frontier",
        advantage=(0.5 if positive else 0.0), useful=positive)
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "dataset_id": "candidate-data-v2",
            "scene_id": scene,
            "episode_id": episode,
            "session_id": f"{episode}/session-0",
            "group_id": group,
            "goal_epoch": goal_epoch,
            "state_id": state_id,
            "state_source": state_source,
            "goal_source_episode_id": goal_source_episode,
            "plan_index": plan_index,
            "prefix_frames": 100,
            "prefix_sha256": SHA,
            "goal_sha256": SHA,
            "navdp_fifo_sha256": SHA,
            "split_role": role,
            "split_sha256": SHA,
            "source_policy_sha256": SHA,
            "candidate_generator_sha256": SHA,
            "feature_builder_sha256": SHA,
            "rollout_labeler_sha256": SHA,
            "environment_id": f"environment/{scene}",
            "navmesh_sha256": SHA,
        },
        "set_features": {
            "feature_presence_mask": [1.0] * 6,
            "native_stagnation_plans": 3,
            "graph_node_count": 20,
            "graph_edge_count": 19,
            "graph_age_frames": 0,
            "memory_candidate_count": 0,
            "frontier_candidate_count": 1,
        },
        "candidates": [
            candidate("native", "native"),
            residual,
            candidate("dustbin", "dustbin"),
        ],
        "set_labels": {
            "global_match": False,
            "strict_no_match": True,
            "ambiguous": False,
            "candidate_set_has_positive": positive,
            "candidate_universe_has_positive": positive,
            "candidate_coverage_miss": False,
            "coverage_label_valid": True,
            "proposal_proxy_set_has_positive": positive,
            "proposal_proxy_universe_has_positive": positive,
            "proposal_proxy_coverage_miss": False,
            "proposal_proxy_coverage_label_valid": True,
            "oracle_best_candidate_id": (
                "frontier-1" if positive else DUSTBIN_CANDIDATE_ID),
        },
    }


class NovelCandidateSetSchemaV2Test(unittest.TestCase):
    def test_valid_record_and_canonical_hash(self):
        value = record()
        shapes = validate_candidate_set(value)
        self.assertEqual(shapes["set.feature_presence_mask"], (6,))
        self.assertEqual(shapes["candidate.goal_patch_relation"], (2,))
        self.assertEqual(shapes["candidate.feature_presence_mask"], (7,))
        digest = canonical_candidate_set_sha256(value)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, canonical_candidate_set_sha256(
            copy.deepcopy(value)))

    def test_allow_list_is_explicit_and_has_no_privileged_name(self):
        self.assertIn("goal_patch_relation", CANDIDATE_FEATURE_KEYS)
        self.assertTrue(
            CANDIDATE_FEATURE_KEYS.isdisjoint(PRIVILEGED_LABEL_DENY_LIST))
        for key in CANDIDATE_FEATURE_KEYS:
            self.assertFalse(any(
                fragment in key.lower()
                for fragment in PRIVILEGED_FEATURE_DENY_FRAGMENTS))

    def test_native_must_be_candidate_zero(self):
        value = record()
        value["candidates"][0], value["candidates"][1] = (
            value["candidates"][1], value["candidates"][0])
        with self.assertRaisesRegex(
                CandidateSetValidationError, "candidate 0"):
            validate_candidate_set(value)

    def test_dustbin_is_required_and_final(self):
        value = record()
        value["candidates"] = value["candidates"][:-1]
        with self.assertRaisesRegex(
                CandidateSetValidationError, "final candidate"):
            validate_candidate_set(value)

    def test_dustbin_cannot_carry_features_or_privileged_labels(self):
        value = record()
        value["candidates"][-1]["features"]["subgoal_forward_m"] = 1.0
        with self.assertRaisesRegex(
                CandidateSetValidationError, "dustbin feature"):
            validate_candidate_set(value)
        value = record()
        value["candidates"][-1]["labels"]["rollout_label_valid"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "dustbin label|valid rollout"):
            validate_candidate_set(value)

    def test_duplicate_candidate_id_fails_closed(self):
        value = record()
        value["candidates"][1]["candidate_id"] = "native"
        with self.assertRaisesRegex(
                CandidateSetValidationError, "unique"):
            validate_candidate_set(value)

    def test_missing_provenance_fails_closed(self):
        value = record()
        del value["provenance"]["group_id"]
        with self.assertRaisesRegex(
                CandidateSetValidationError, "missing=.*group_id"):
            validate_candidate_set(value)

    def test_truthful_provenance_fields_are_strict(self):
        validate_candidate_set(record(state_source="on_policy"))

        value = record()
        value["provenance"]["state_source"] = "synthetic"
        with self.assertRaisesRegex(
                CandidateSetValidationError, "state_source"):
            validate_candidate_set(value)

        value = record()
        value["provenance"]["rollout_labeler_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(
                CandidateSetValidationError, "rollout_labeler_sha256"):
            validate_candidate_set(value)

        value = record()
        value["provenance"]["environment_id"] = ""
        with self.assertRaisesRegex(
                CandidateSetValidationError, "environment_id"):
            validate_candidate_set(value)

        first = record(plan_index=0)
        second = record(plan_index=1)
        second["provenance"]["rollout_labeler_sha256"] = "b" * 64
        with self.assertRaisesRegex(
                CandidateSetValidationError, "signature"):
            validate_candidate_dataset([first, second])

    def test_same_state_and_goal_epoch_provenance_are_immutable(self):
        factual = record(
            state_id="scene-a/episode-0/state-shared",
            goal_epoch="goal-factual",
        )
        counterfactual = record(
            state_id="scene-a/episode-0/state-shared",
            goal_epoch="goal-counterfactual",
        )
        validate_candidate_dataset([factual, counterfactual])

        changed_state = copy.deepcopy(counterfactual)
        changed_state["provenance"]["prefix_sha256"] = "b" * 64
        with self.assertRaisesRegex(
                CandidateSetValidationError, "inconsistent causal"):
            validate_candidate_dataset([factual, changed_state])

        next_plan = record(plan_index=1, goal_epoch="goal-factual")
        next_plan["provenance"]["goal_sha256"] = "b" * 64
        with self.assertRaisesRegex(
                CandidateSetValidationError, "changes source or content"):
            validate_candidate_dataset([factual, next_plan])

    def test_dataset_rejects_missing_or_cross_environment_goal_source(self):
        missing = record(goal_source_episode="absent/episode")
        with self.assertRaisesRegex(
                CandidateSetValidationError, "goal source episode is absent"):
            validate_candidate_dataset([missing])

        source = record(scene="scene-a", episode="scene-a/source")
        cross_environment = record(
            scene="scene-b",
            episode="scene-b/current",
            goal_source_episode="scene-a/source",
        )
        with self.assertRaisesRegex(
                CandidateSetValidationError,
                "cross-scene/environment goal source"):
            validate_candidate_dataset([source, cross_environment])

        same_environment = record(
            scene="scene-a",
            episode="scene-a/current",
            goal_source_episode="scene-a/source",
        )
        validate_candidate_dataset([source, same_environment])

        changed_navmesh = record(
            scene="scene-a", episode="scene-a/changed", plan_index=1)
        changed_navmesh["provenance"]["navmesh_sha256"] = "b" * 64
        with self.assertRaisesRegex(
                CandidateSetValidationError, "environment/navmesh"):
            validate_candidate_dataset([source, changed_navmesh])

    def test_privileged_or_unknown_feature_cannot_enter_input(self):
        value = record()
        value["candidates"][1]["features"]["goal_geodesic_m"] = 2.0
        with self.assertRaisesRegex(
                CandidateSetValidationError, "extra=.*goal_geodesic_m"):
            validate_candidate_set(value)

    def test_missing_allowed_feature_fails_closed(self):
        value = record()
        del value["candidates"][1]["features"]["goal_patch_relation"]
        with self.assertRaisesRegex(
                CandidateSetValidationError, "missing=.*goal_patch_relation"):
            validate_candidate_set(value)

    def test_nested_nan_fails_closed(self):
        value = record()
        value["candidates"][1]["features"]["goal_patch_relation"][1] = float("nan")
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be finite"):
            validate_candidate_set(value)

    def test_feature_shape_mismatch_fails_closed(self):
        value = record()
        value["candidates"][1]["features"]["goal_patch_relation"].append(0.3)
        with self.assertRaisesRegex(
                CandidateSetValidationError, "feature shapes differ"):
            validate_candidate_set(value)

    def test_final_reserved_is_not_a_trainable_artifact_role(self):
        with self.assertRaisesRegex(
                CandidateSetValidationError, "not train/development"):
            validate_candidate_set(record(role="final_reserved"))

    def test_positive_summary_must_match_candidate_labels(self):
        value = record(positive=False)
        value["set_labels"]["candidate_set_has_positive"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "disagrees"):
            validate_candidate_set(value)

    def test_useful_label_cannot_contradict_harm(self):
        value = record()
        value["candidates"][1]["labels"]["harm"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "contradicts"):
            validate_candidate_set(value)

    def test_advantage_must_equal_residual_minus_native_progress(self):
        value = record()
        value["candidates"][1]["labels"]["geodesic_progress_h24_m"] = -10.0
        with self.assertRaisesRegex(
                CandidateSetValidationError, "advantage disagrees"):
            validate_candidate_set(value)

    def test_regression_must_match_negative_advantage_margin(self):
        value = record(positive=False)
        residual_labels = value["candidates"][1]["labels"]
        residual_labels["geodesic_progress_h24_m"] = 0.1
        residual_labels["advantage_h24_m"] = -0.4
        with self.assertRaisesRegex(
                CandidateSetValidationError, "regression_h24 disagrees"):
            validate_candidate_set(value)

    def test_useful_must_match_margin_and_safety(self):
        value = record()
        value["candidates"][1]["labels"]["useful"] = False
        with self.assertRaisesRegex(
                CandidateSetValidationError, "useful label disagrees"):
            validate_candidate_set(value)

        value = record(positive=False)
        value["candidates"][1]["labels"]["advantage_h24_m"] = 0.1
        value["candidates"][1]["labels"]["geodesic_progress_h24_m"] = 0.6
        value["candidates"][1]["labels"]["useful"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "useful label disagrees"):
            validate_candidate_set(value)

    def test_collision_requires_harm(self):
        value = record(positive=False)
        value["candidates"][1]["labels"]["collision_h8"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "collision-or-regression"):
            validate_candidate_set(value)

        value = record(positive=False)
        value["candidates"][1]["labels"]["harm"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "collision-or-regression"):
            validate_candidate_set(value)

        value = record(positive=False)
        value["candidates"][1]["labels"]["geodesic_progress_h24_m"] = 0.0
        value["candidates"][1]["labels"]["advantage_h24_m"] = -0.5
        value["candidates"][1]["labels"]["regression_h24"] = True
        value["candidates"][1]["labels"]["harm"] = True
        validate_candidate_set(value)

    def test_rollout_validity_and_reachability_are_consistent(self):
        value = record(positive=False)
        value["candidates"][1]["labels"]["rollout_label_valid"] = False
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be neutral"):
            validate_candidate_set(value)

    def test_oracle_must_be_maximum_valid_useful_advantage(self):
        value = record()
        better = candidate(
            "memory-1", "memory_graph", advantage=0.75, useful=True)
        value["candidates"].insert(-1, better)
        value["set_features"]["memory_candidate_count"] = 1
        with self.assertRaisesRegex(
                CandidateSetValidationError, "maximum valid useful"):
            validate_candidate_set(value)
        value["set_labels"]["oracle_best_candidate_id"] = "memory-1"
        validate_candidate_set(value)

    def test_candidate_universe_coverage_miss_contract(self):
        value = record(positive=False)
        value["set_labels"]["candidate_universe_has_positive"] = True
        value["set_labels"]["candidate_coverage_miss"] = True
        validate_candidate_set(value)
        value["set_labels"]["candidate_coverage_miss"] = False
        with self.assertRaisesRegex(
                CandidateSetValidationError, "coverage_miss disagrees"):
            validate_candidate_set(value)

        value = record(positive=False)
        value["set_labels"]["coverage_label_valid"] = False
        value["set_labels"]["candidate_universe_has_positive"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be neutral"):
            validate_candidate_set(value)

    def test_proposal_proxy_labels_are_strict_but_independent_from_actual(self):
        # Actual H24 useful can be true while the generator proxy is negative.
        value = record(positive=True)
        proxy = value["candidates"][1]["labels"]
        proxy["proposal_proxy_progress_m"] = -0.1
        proxy["proposal_proxy_positive"] = False
        value["set_labels"]["proposal_proxy_set_has_positive"] = False
        value["set_labels"]["proposal_proxy_universe_has_positive"] = False
        validate_candidate_set(value)

        # Conversely, a proxy-positive proposal need not have actual H24 utility.
        value = record(positive=False)
        proxy = value["candidates"][1]["labels"]
        proxy["proposal_proxy_progress_m"] = 1.0
        proxy["proposal_proxy_positive"] = True
        value["set_labels"]["proposal_proxy_set_has_positive"] = True
        value["set_labels"]["proposal_proxy_universe_has_positive"] = True
        validate_candidate_set(value)

        proxy["proposal_proxy_positive"] = False
        with self.assertRaisesRegex(
                CandidateSetValidationError, "proxy positive disagrees"):
            validate_candidate_set(value)

        # A residual can have only a valid proposal proxy; native actual rollout
        # remains mandatory and supplies the actual-utility reference.
        value = record(positive=False)
        residual = value["candidates"][1]["labels"]
        residual.update({
            "geodesic_progress_h8_m": 0.0,
            "geodesic_progress_h24_m": 0.0,
            "advantage_h24_m": 0.0,
            "reachable": False,
            "rollout_label_valid": False,
            "proposal_proxy_progress_m": 1.0,
            "proposal_proxy_reachable": True,
            "proposal_proxy_positive": True,
        })
        value["set_labels"]["proposal_proxy_set_has_positive"] = True
        value["set_labels"]["proposal_proxy_universe_has_positive"] = True
        validate_candidate_set(value)

    def test_proposal_proxy_validity_and_coverage_masks_fail_closed(self):
        value = record(positive=False)
        proxy = value["candidates"][1]["labels"]
        proxy["proposal_proxy_label_valid"] = False
        proxy["proposal_proxy_reachable"] = False
        validate_candidate_set(value)
        proxy["proposal_proxy_progress_m"] = 0.1
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be neutral"):
            validate_candidate_set(value)

        value = record(positive=False)
        value["set_labels"]["proposal_proxy_universe_has_positive"] = True
        value["set_labels"]["proposal_proxy_coverage_miss"] = True
        validate_candidate_set(value)
        value["set_labels"]["proposal_proxy_coverage_miss"] = False
        with self.assertRaisesRegex(
                CandidateSetValidationError, "coverage_miss disagrees"):
            validate_candidate_set(value)

        value = record(positive=False)
        value["set_labels"]["proposal_proxy_coverage_label_valid"] = False
        value["set_labels"]["proposal_proxy_universe_has_positive"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be neutral"):
            validate_candidate_set(value)

    def test_native_still_requires_actual_rollout_and_no_proxy_label(self):
        value = record()
        native = value["candidates"][0]["labels"]
        native["rollout_label_valid"] = False
        with self.assertRaises(CandidateSetValidationError):
            validate_candidate_set(value)

        value = record()
        native = value["candidates"][0]["labels"]
        native["proposal_proxy_label_valid"] = True
        native["proposal_proxy_reachable"] = True
        with self.assertRaisesRegex(
                CandidateSetValidationError, "native candidate cannot carry"):
            validate_candidate_set(value)

    def test_counts_must_be_nonnegative_integers_and_match_rows(self):
        for key, invalid in (
                ("native_stagnation_plans", -1),
                ("graph_node_count", 1.5),
                ("frontier_candidate_count", 2)):
            with self.subTest(key=key, invalid=invalid):
                value = record()
                value["set_features"][key] = invalid
                with self.assertRaises(CandidateSetValidationError):
                    validate_candidate_set(value)

    def test_presence_mask_and_bounded_features_are_strict(self):
        mutations = (
            ("feature_presence_mask", [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]),
            ("depth_confidence_mean", 1.1),
            ("pose_translation_p90_m", -0.1),
            ("graph_hops", 1.5),
        )
        for key, invalid in mutations:
            with self.subTest(key=key):
                value = record()
                value["candidates"][1]["features"][key] = invalid
                with self.assertRaises(CandidateSetValidationError):
                    validate_candidate_set(value)

        value = record()
        value["candidates"][1]["features"]["feature_presence_mask"][0] = 0.0
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be zero when absent"):
            validate_candidate_set(value)

        grouped_absence = (
            (4, ("pose_translation_p90_m", "pose_yaw_p90_deg")),
            (5, ("depth_confidence_mean",)),
            (6, ("clearance_lower_m",)),
        )
        for mask_index, feature_keys in grouped_absence:
            with self.subTest(mask_index=mask_index):
                value = record()
                value["candidates"][1]["features"][
                    "feature_presence_mask"][mask_index] = 0.0
                with self.assertRaisesRegex(
                        CandidateSetValidationError, "must be zero when absent"):
                    validate_candidate_set(value)
                for feature_key in feature_keys:
                    value["candidates"][1]["features"][feature_key] = 0.0
                validate_candidate_set(value)

    def test_set_presence_mask_represents_unknown_stagnation(self):
        value = record()
        value["set_features"]["native_stagnation_plans"] = 0
        value["set_features"]["feature_presence_mask"][0] = 0.0
        validate_candidate_set(value)

        value["set_features"]["native_stagnation_plans"] = 3
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be zero when absent"):
            validate_candidate_set(value)

        value = record()
        value["set_features"]["feature_presence_mask"][0] = 0.5
        with self.assertRaisesRegex(
                CandidateSetValidationError, "numeric 0 or 1"):
            validate_candidate_set(value)

    def test_residual_candidate_k_is_bounded(self):
        value = record(positive=False)
        value["candidates"] = (
            [value["candidates"][0]]
            + [candidate(f"frontier-{index}", "frontier")
               for index in range(33)]
            + [value["candidates"][-1]]
        )
        value["set_features"]["frontier_candidate_count"] = 33
        with self.assertRaisesRegex(
                CandidateSetValidationError, "exceeds residual K=32"):
            validate_candidate_set(value)

    def test_invalid_label_mask_requires_neutral_targets(self):
        value = record()
        value["candidates"][1]["labels"]["rollout_label_valid"] = False
        value["candidates"][1]["labels"]["useful"] = False
        with self.assertRaisesRegex(
                CandidateSetValidationError, "must be neutral"):
            validate_candidate_set(value)

    def test_dataset_rejects_duplicate_decisions(self):
        value = record()
        with self.assertRaisesRegex(
                CandidateSetValidationError, "duplicate decision"):
            validate_candidate_dataset([value, copy.deepcopy(value)])

    def test_dataset_rejects_group_split_leakage(self):
        first = record(plan_index=0, role="train", group="shared")
        second = record(
            plan_index=1, scene="scene-b", role="development", group="shared")
        with self.assertRaisesRegex(
                CandidateSetValidationError, "crosses scene or split"):
            validate_candidate_dataset([first, second])

    def test_dataset_rejects_artifact_signature_mismatch(self):
        first = record(plan_index=0)
        second = record(plan_index=1)
        second["provenance"]["feature_builder_sha256"] = "b" * 64
        with self.assertRaisesRegex(
                CandidateSetValidationError, "signature"):
            validate_candidate_dataset([first, second])

    def test_dataset_rejects_decreasing_causal_prefix(self):
        first = record(plan_index=0)
        second = record(plan_index=1)
        first["provenance"]["prefix_frames"] = 100
        second["provenance"]["prefix_frames"] = 90
        with self.assertRaisesRegex(
                CandidateSetValidationError, "prefix length decreases"):
            validate_candidate_dataset([first, second])

    def test_dataset_accepts_distinct_groups_and_reports_counts(self):
        first = record(plan_index=0, group="group-a")
        second = record(plan_index=1, group="group-a")
        report = validate_candidate_dataset([first, second])
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["scenes"], 1)
        self.assertEqual(report["groups"], 1)


if __name__ == "__main__":
    unittest.main()
