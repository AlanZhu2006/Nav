import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import MemNavData.test_real_h24_rollout_backend as backend_test_support

from MemNavData.build_nlsr_precollection_candidate_sets import (
    DEPLOYMENT_ARM,
    RELATION_ARTIFACT_SCHEMA,
    TEACHER_ARM,
    PrecollectionBuildError,
    build_precollection_records,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_pinned_canonical_json,
    rollout_labeler_code_sha256,
    write_candidate_records,
)
from MemNavData.collect_real_h24_rollouts import (
    _atomic_write_pair,
    build_plan_diagnostics,
    build_run_signature,
    decision_seeds,
)
from MemNavData.merge_nlsr_h24_candidate_sets import (
    CandidateMergeError,
    H24RunBinding,
    merge_candidate_records,
)
from MemNavData.native_frontier_relation import native_frontier_relation
from MemNavData.novel_rollout_protocol_v2 import (
    CandidateArm,
    atomic_write_artifact,
    collect_paired_rollouts,
)
from MemNavData.real_h24_rollout_backend import PurePursuitConfig


SHA = "a" * 64
POLICY_SHA = "b" * 64
BASE_SEED = 109


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def manifest_fixture():
    sample_id = "train/scene/episode_0/goal_b_t0/factual"
    sample = {
        "sample_id": sample_id,
        "split_role": "train",
        "scene": "scene",
        "state_source": "expert",
        "source_episode": "episode_0",
        "source_episode_id": "scene/episode_0",
        "goal_episode": "episode_0",
        "goal_source_episode_id": "scene/episode_0",
        "goal_variant": "factual",
        "goal_role": "B",
        "state_name": "goal_b_t0",
        "decision_frame": 16,
        "state_frame": {"content_sha256": "1" * 64},
        "causal_prefix": {"causal_prefix_sha256": "2" * 64},
        "navdp_fifo": {"fifo_sha256": "3" * 64},
        "goal": {"content_sha256": "4" * 64},
    }
    return {
        "schema_version": "nlsr_v2_multistage_expert_candidate_manifest_v1",
        "split": {"sha256": "5" * 64},
        "scenes": [
            {
                "scene": "scene",
                "split_role": "train",
                "environment": {"content_sha256": "6" * 64},
                "navmesh": {"content_sha256": "7" * 64},
                "selected_episodes": [],
            }
        ],
        "samples": [sample],
        "sample_group_bindings": [
            {
                "sample_id": sample_id,
                "counterfactual_pair_group_id": "train/scene/episode_0/goal_b_t0",
            }
        ],
    }


def proposal_candidate(candidate_id="deployment-frontier"):
    return {
        "candidate_id": candidate_id,
        "map_xy_m": [1.0, 0.0],
        "subgoal_forward_m": 1.0,
        "subgoal_left_m": 0.0,
        "distance_m": 1.0,
        "bearing_rad": 0.0,
        "frontier_normal_bearing_rad": 0.0,
        "resolution_m": 0.2,
        "grid_cell": [5, 0],
        "frontier_boundary_m": 1.4,
        "frontier_novelty_m": 0.8,
        "clearance_lower_m": 0.5,
        "topology_score": 2.0,
        "context_frame_indices": [8, 15],
        "goal_patch_relation_score": 0.6,
        "goal_patch_relation_present": True,
        "selection_sources": ["goal_patch_top2"],
        "source_scales_m": [0.2],
    }


def proposal(candidate):
    return {
        "schema_version": "nlsr_v2_frontier_proposal_v1",
        "valid": True,
        "invalid_reason": None,
        "pose_frame_index": 15,
        "scan_frame_indices": [0, 4, 8, 12, 15],
        "goal_patch_relation_present": True,
        "goal_patch_relation_mask": 1,
        "shortlist_policy": {"slots": [], "max_candidates": 6},
        "scale_summaries": [],
        "raw_candidate_count": 1,
        "nms_candidate_count": 1,
        "shortlist_count": 1,
        "candidate_universe": [copy.deepcopy(candidate)],
        "shortlist": [copy.deepcopy(candidate)],
        "nms_suppressed": [],
    }


def proxy(proposal_value, *, labeled=False):
    if labeled:
        candidate_id = proposal_value["shortlist"][0]["candidate_id"]
        labels = [
            {
                "candidate_id": candidate_id,
                "reachable": True,
                "progress_m": 0.5,
                "positive": True,
            }
        ]
        status = "labeled"
        summary = True
        margin = 0.0
    else:
        labels = []
        status = "not_requested"
        summary = False
        margin = None
    return {
        "status": status,
        "label_valid": labeled,
        "labeler_provenance": ({"producer_sha256": "f" * 64} if labeled else None),
        "positive_margin_m": margin,
        "labels": labels,
        "universe_has_positive": summary,
        "shortlist_has_positive": summary,
        "coverage_miss": False,
        "proposal_sha256": sha256_bytes(canonical_json_bytes(proposal_value)),
    }


def arm(name, proposal_value, *, labeled=False):
    return {
        "arm": name,
        "deployment_eligible_pose_source": name == DEPLOYMENT_ARM,
        "pose_provenance": {"source": name},
        "proposal": copy.deepcopy(proposal_value),
        "proposal_proxy": proxy(proposal_value, labeled=labeled),
    }


def proposal_artifact(manifest_sha, *, labeled=False):
    manifest = manifest_fixture()
    sample = manifest["samples"][0]
    deployment_candidate = proposal_candidate()
    deployment_proposal = proposal(deployment_candidate)
    teacher_proposal = proposal(proposal_candidate("teacher-only"))
    return {
        "schema_version": "nlsr_v2_frontier_proposal_artifact_v2",
        "provenance": {"input_manifest_sha256": manifest_sha},
        "records": [
            {
                "sample_id": sample["sample_id"],
                "scene": sample["scene"],
                "source_episode": sample["source_episode"],
                "goal_episode": sample["goal_episode"],
                "goal_variant": sample["goal_variant"],
                "goal_role": sample["goal_role"],
                "state_name": sample["state_name"],
                "split_role": sample["split_role"],
                "decision_frame": sample["decision_frame"],
                "causal_prefix_sha256": sample["causal_prefix"]["causal_prefix_sha256"],
                "goal_sha256": sample["goal"]["content_sha256"],
                "patch_score_present": True,
                "arms": {
                    TEACHER_ARM: arm(TEACHER_ARM, teacher_proposal),
                    DEPLOYMENT_ARM: arm(
                        DEPLOYMENT_ARM, deployment_proposal, labeled=labeled
                    ),
                },
            }
        ],
    }


def relation_artifact(manifest_sha, proposal_sha):
    manifest = manifest_fixture()
    sample = manifest["samples"][0]
    return {
        "schema_version": RELATION_ARTIFACT_SCHEMA,
        "input_manifest_sha256": manifest_sha,
        "input_proposal_sha256": proposal_sha,
        "producer_source_sha256": "8" * 64,
        "configuration_sha256": "9" * 64,
        "feature_shapes": {
            "goal_patch_relation": 2,
            "goal_temporal_relation": 3,
            "local_map_relation": 2,
        },
        "records": [
            {
                "sample_id": sample["sample_id"],
                "prefix_sha256": sample["causal_prefix"]["causal_prefix_sha256"],
                "goal_sha256": sample["goal"]["content_sha256"],
                "candidates": [
                    {
                        "candidate_id": "deployment-frontier",
                        "goal_patch_relation": [0.6, 0.4],
                        "goal_patch_relation_present": True,
                        "goal_temporal_relation": [0.2, 0.3, 0.4],
                        "goal_temporal_relation_present": True,
                        "local_map_relation": [0.8, 0.1],
                        "local_map_relation_present": True,
                        "pose_translation_p90_m": 0.25,
                        "pose_yaw_p90_deg": 4.0,
                        "pose_uncertainty_present": True,
                        "depth_confidence_mean": 0.75,
                        "depth_confidence_present": True,
                    }
                ],
            }
        ],
    }


def build_fixture(*, labeled=False, with_relation=False):
    manifest = manifest_fixture()
    manifest_sha = sha256_bytes(canonical_json_bytes(manifest))
    proposals = proposal_artifact(manifest_sha, labeled=labeled)
    proposal_sha = sha256_bytes(canonical_json_bytes(proposals))
    relations = relation_artifact(manifest_sha, proposal_sha) if with_relation else None
    relation_sha = (
        sha256_bytes(canonical_json_bytes(relations)) if relations is not None else None
    )
    records = build_precollection_records(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        proposal_artifact=proposals,
        proposal_sha256=proposal_sha,
        source_policy_sha256=POLICY_SHA,
        relation_artifact=relations,
        relation_sha256=relation_sha,
    )
    return manifest, proposals, relations, records


class NLSRPrecollectionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_deployment_arm_is_selectable_and_labels_are_neutral(self):
        _manifest, _proposals, _relations, records = build_fixture(labeled=True)
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual(
            [candidate["candidate_id"] for candidate in row["candidates"]],
            ["native", "deployment-frontier", "dustbin"],
        )
        self.assertNotIn("teacher-only", str(row))
        for candidate in row["candidates"]:
            self.assertFalse(candidate["labels"]["rollout_label_valid"])
            self.assertFalse(candidate["labels"]["covisibility_label_valid"])
            self.assertFalse(candidate["labels"]["pose_label_valid"])
        residual = row["candidates"][1]
        self.assertTrue(residual["labels"]["proposal_proxy_label_valid"])
        self.assertEqual(residual["labels"]["proposal_proxy_progress_m"], 0.5)
        self.assertEqual(residual["features"]["goal_patch_relation"], [0.6])
        self.assertEqual(residual["features"]["feature_presence_mask"][0], 1.0)
        self.assertEqual(residual["features"]["feature_presence_mask"][3], 0.0)
        self.assertEqual(len(residual["features"]["native_proposal_relation"]), 21)

    def test_pinned_relation_features_are_masked_and_shape_stable(self):
        _manifest, _proposals, _relations, records = build_fixture(with_relation=True)
        residual = records[0]["candidates"][1]
        self.assertEqual(residual["features"]["goal_patch_relation"], [0.6, 0.4])
        self.assertEqual(
            residual["features"]["goal_temporal_relation"], [0.2, 0.3, 0.4]
        )
        self.assertEqual(
            residual["features"]["feature_presence_mask"][:3], [1.0, 1.0, 1.0]
        )
        self.assertEqual(residual["features"]["pose_translation_p90_m"], 0.25)

    def test_missing_extra_teacher_or_feature_leakage_fail_closed(self):
        manifest = manifest_fixture()
        manifest_sha = sha256_bytes(canonical_json_bytes(manifest))
        proposals = proposal_artifact(manifest_sha)

        missing = copy.deepcopy(proposals)
        missing["records"] = []
        with self.assertRaisesRegex(PrecollectionBuildError, "missing or extra"):
            build_precollection_records(
                manifest=manifest,
                manifest_sha256=manifest_sha,
                proposal_artifact=missing,
                proposal_sha256=sha256_bytes(canonical_json_bytes(missing)),
                source_policy_sha256=POLICY_SHA,
            )

        teacher = copy.deepcopy(proposals)
        teacher["records"][0]["arms"][TEACHER_ARM][
            "deployment_eligible_pose_source"
        ] = True
        with self.assertRaisesRegex(PrecollectionBuildError, "eligibility"):
            build_precollection_records(
                manifest=manifest,
                manifest_sha256=manifest_sha,
                proposal_artifact=teacher,
                proposal_sha256=sha256_bytes(canonical_json_bytes(teacher)),
                source_policy_sha256=POLICY_SHA,
            )

        leakage = copy.deepcopy(proposals)
        leakage["records"][0]["arms"][DEPLOYMENT_ARM]["proposal"]["shortlist"][0][
            "goal_geodesic_m"
        ] = 2.0
        with self.assertRaisesRegex(PrecollectionBuildError, "fields changed"):
            build_precollection_records(
                manifest=manifest,
                manifest_sha256=manifest_sha,
                proposal_artifact=leakage,
                proposal_sha256=sha256_bytes(canonical_json_bytes(leakage)),
                source_policy_sha256=POLICY_SHA,
            )

    def test_canonical_json_jsonl_atomic_resume_and_drift_rejection(self):
        _manifest, _proposals, _relations, records = build_fixture()
        json_path = self.root / "candidate.json"
        status, digest = write_candidate_records(records, json_path)
        self.assertEqual(status, "written")
        self.assertEqual(json_path.read_bytes(), canonical_json_bytes(records))
        self.assertEqual(sha256_bytes(json_path.read_bytes()), digest)
        resumed, resumed_digest = write_candidate_records(
            records, json_path, resume=True
        )
        self.assertEqual((resumed, resumed_digest), ("resumed", digest))
        changed = copy.deepcopy(records)
        changed[0]["candidates"][1]["features"]["subgoal_forward_m"] = 2.0
        with self.assertRaisesRegex(PrecollectionBuildError, "differs"):
            write_candidate_records(changed, json_path, resume=True)

        jsonl_path = self.root / "candidate.jsonl"
        write_candidate_records(records, jsonl_path)
        self.assertEqual(jsonl_path.read_bytes(), canonical_jsonl_bytes(records))

    def test_pinned_input_rejects_noncanonical_and_duplicate_keys(self):
        noncanonical = self.root / "noncanonical.json"
        noncanonical.write_text('{"b": 1, "a": 2}\n', encoding="utf-8")
        with self.assertRaisesRegex(PrecollectionBuildError, "noncanonical"):
            load_pinned_canonical_json(
                noncanonical, sha256_bytes(noncanonical.read_bytes())
            )

        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
        with self.assertRaisesRegex(PrecollectionBuildError, "duplicate key"):
            load_pinned_canonical_json(duplicate, sha256_bytes(duplicate.read_bytes()))


def neutral_h24_record(state):
    _manifest, _proposals, _relations, records = build_fixture()
    row = records[0]
    row["provenance"].update(
        {
            "scene_id": state.environment_id,
            "episode_id": state.session_id,
            "session_id": state.session_id,
            "group_id": f"train/{state.environment_id}/episode/state",
            "goal_epoch": state.goal_epoch,
            "state_id": state.state_id,
            "goal_source_episode_id": state.session_id,
            "goal_sha256": state.goal_sha256,
            "navdp_fifo_sha256": state.manifest_fifo_sha256,
            "environment_id": state.environment_id,
            "navmesh_sha256": state.navmesh_sha256,
            "rollout_labeler_sha256": rollout_labeler_code_sha256(),
        }
    )
    residual = row["candidates"][1]
    residual["candidate_id"] = "frontier-1"
    residual["features"]["subgoal_forward_m"] = 1.0
    residual["features"]["subgoal_left_m"] = 0.0
    return row


class NLSRH24MergeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = backend_test_support.RealH24RolloutBackendTests(
            methodName="runTest"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.record = neutral_h24_record(self.fixture.assets.state)
        self.record["provenance"]["source_policy_sha256"] = (
            backend_test_support.FakeTransport.provenance["checkpoint_sha256"]
        )
        self.precollection_sha = sha256_bytes(canonical_json_bytes([self.record]))
        self.manifest_sha = "a" * 64
        self.geometry_map_sha = "b" * 64
        self.server_provenance_sha = sha256_bytes(
            canonical_json_bytes(backend_test_support.FakeTransport.provenance)
        )
        self.run_binding = H24RunBinding(
            manifest_sha256=self.manifest_sha,
            geometry_map_sha256=self.geometry_map_sha,
            server_provenance_sha256=self.server_provenance_sha,
            server_provenance=backend_test_support.FakeTransport.provenance,
            server_url="http://127.0.0.1:28991",
            stop_threshold=-0.5,
            legacy_camera_height_m=0.5,
        )
        self.run_signature = build_run_signature(
            candidate_sha256=self.precollection_sha,
            manifest_sha256=self.manifest_sha,
            geometry_map_sha256=self.geometry_map_sha,
            server_provenance_sha256=self.server_provenance_sha,
            server_url=self.run_binding.server_url,
            base_seed=BASE_SEED,
            stop_threshold=self.run_binding.stop_threshold,
            legacy_camera_height_m=self.run_binding.legacy_camera_height_m,
            controller=PurePursuitConfig(),
        )
        self.seeds = decision_seeds(BASE_SEED, self.fixture.assets.state.state_id)
        self.backends = {}

        def factory(candidate_id):
            backend = self.fixture.backend(
                transport=backend_test_support.FakeTransport(),
                runtime=backend_test_support.FakeRuntime(self.fixture.identity),
            )
            self.backends[candidate_id] = backend
            return backend

        self.artifact = collect_paired_rollouts(
            factory,
            self.fixture.assets.state,
            (
                CandidateArm("native", "native"),
                CandidateArm("frontier-1", "frontier", (0.0, -1.0)),
            ),
            self.seeds,
            run_signature_sha256=self.run_signature,
        )
        self.diagnostics = build_plan_diagnostics(self.artifact, self.backends)
        self.artifact_path = self.root / "shard-0000/state.json"
        self.diagnostics_path = self.root / "shard-0000/state.plans.json"
        atomic_write_artifact(self.artifact_path, self.artifact)
        _atomic_write_pair(self.diagnostics_path, self.diagnostics)

    def tearDown(self):
        self.temporary.cleanup()

    def merge(self, records=None, **kwargs):
        return merge_candidate_records(
            records=[self.record] if records is None else records,
            precollection_sha256=self.precollection_sha,
            rollout_root=self.root,
            expected_run_signature_sha256=kwargs.pop(
                "run_signature", self.run_signature
            ),
            base_seed=kwargs.pop("base_seed", BASE_SEED),
            run_binding=kwargs.pop("run_binding", self.run_binding),
            **kwargs,
        )

    def test_exact_h24_join_fills_labels_and_raw_native_relation(self):
        merged = self.merge()
        row = merged[0]
        residual = row["candidates"][1]
        self.assertTrue(row["candidates"][0]["labels"]["rollout_label_valid"])
        self.assertTrue(residual["labels"]["rollout_label_valid"])
        self.assertEqual(residual["features"]["feature_presence_mask"][3], 1.0)
        native_outcome = next(
            outcome
            for outcome in self.artifact.outcomes
            if outcome.candidate_id == "native"
        )
        plan = next(plan for plan in native_outcome.plans if plan.commitment_index == 0)
        diagnostic = self.diagnostics["by_candidate"]["native"][plan.plan_sha256]
        expected = native_frontier_relation(
            np.asarray(diagnostic["all_trajectory"])[0],
            [1.0, 0.0],
            selected_index=diagnostic["server_selected_trajectory_index"],
            native_values=np.asarray(diagnostic["all_values"])[0],
        )
        np.testing.assert_array_equal(
            residual["features"]["native_proposal_relation"], expected
        )
        self.assertTrue(row["provenance"]["dataset_id"].startswith("nlsr-h24:"))

    def test_duplicate_missing_seed_and_candidate_mismatch_fail_closed(self):
        duplicate_path = self.root / "shard-0001/duplicate.json"
        duplicate_diagnostics = self.root / "shard-0001/duplicate.plans.json"
        atomic_write_artifact(duplicate_path, self.artifact)
        _atomic_write_pair(duplicate_diagnostics, self.diagnostics)
        with self.assertRaisesRegex(CandidateMergeError, "duplicate H24 state"):
            self.merge()
        for path in (
            duplicate_path,
            duplicate_path.with_suffix(".json.sha256"),
            duplicate_diagnostics,
            duplicate_diagnostics.with_suffix(".json.sha256"),
        ):
            path.unlink()

        with self.assertRaisesRegex(CandidateMergeError, "run signature"):
            self.merge(base_seed=BASE_SEED + 1)

        changed = copy.deepcopy(self.record)
        changed["candidates"][1]["candidate_id"] = "other"
        with self.assertRaisesRegex(CandidateMergeError, "candidate arms mismatch"):
            self.merge(records=[changed])

        wrong_policy = copy.deepcopy(self.record)
        wrong_policy["provenance"]["source_policy_sha256"] = "0" * 64
        with self.assertRaisesRegex(CandidateMergeError, "server checkpoint"):
            self.merge(records=[wrong_policy])

    def test_label_leakage_and_tampered_selected_index_fail_closed(self):
        leaked = copy.deepcopy(self.record)
        leaked["candidates"][1]["labels"]["advantage_h24_m"] = 9.0
        with self.assertRaisesRegex(CandidateMergeError, "already carries"):
            self.merge(records=[leaked])

        self.diagnostics_path.unlink()
        self.diagnostics_path.with_suffix(".json.sha256").unlink()
        tampered = copy.deepcopy(self.diagnostics)
        native_outcome = next(
            outcome
            for outcome in self.artifact.outcomes
            if outcome.candidate_id == "native"
        )
        plan = next(plan for plan in native_outcome.plans if plan.commitment_index == 0)
        tampered["by_candidate"]["native"][plan.plan_sha256][
            "server_selected_trajectory_index"
        ] = 99
        _atomic_write_pair(self.diagnostics_path, tampered)
        with self.assertRaisesRegex(CandidateMergeError, "out of range"):
            self.merge()

    def test_match_and_pose_labels_are_not_invented_or_overwritten(self):
        pinned = copy.deepcopy(self.record)
        labels = pinned["candidates"][1]["labels"]
        labels.update(
            {
                "teacher_covisibility": 0.7,
                "covisibility_label_valid": True,
                "pose_residual_forward_m": 0.2,
                "pose_residual_left_m": -0.1,
                "pose_residual_yaw_rad": 0.05,
                "pose_label_valid": True,
            }
        )
        merged = self.merge(records=[pinned])
        after = merged[0]["candidates"][1]["labels"]
        self.assertEqual(after["teacher_covisibility"], 0.7)
        self.assertEqual(after["pose_residual_forward_m"], 0.2)
        self.assertTrue(after["pose_label_valid"])


if __name__ == "__main__":
    unittest.main()
