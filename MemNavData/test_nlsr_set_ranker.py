import copy
from dataclasses import replace
import unittest

import torch

from MemNavData.nlsr_set_ranker import (
    NLSRLossConfig,
    NLSRRankerConfig,
    NLSRRankerError,
    NLSRSetRanker,
    RankerFeatureSpec,
    assert_scene_group_disjoint,
    build_checkpoint_metadata,
    compute_nlsr_losses,
    dataset_provenance_sha256,
    feature_spec_from_dataset,
    load_portable_checkpoint,
    make_portable_checkpoint,
    split_by_declared_role,
    vectorize_candidate_sets,
)
from MemNavData.novel_candidate_set_schema_v2 import validate_candidate_set
from MemNavData.test_novel_candidate_set_schema_v2 import (
    candidate as synthetic_candidate,
    record as synthetic_record,
)


def record_with_two_residuals():
    value = synthetic_record(positive=True)
    memory = synthetic_candidate(
        "memory-1", "memory_graph", advantage=0.3, useful=True)
    value["candidates"].insert(-1, memory)
    value["set_features"]["memory_candidate_count"] = 1
    validate_candidate_set(value)
    return value


def negative_harm_record():
    value = synthetic_record(positive=False)
    labels = value["candidates"][1]["labels"]
    labels.update({
        "geodesic_progress_h8_m": 0.0,
        "geodesic_progress_h24_m": 0.0,
        "advantage_h24_m": -0.5,
        "harm": True,
        "useful": False,
        "reachable": True,
        "collision_h8": False,
        "regression_h24": True,
        "proposal_proxy_progress_m": -0.5,
        "proposal_proxy_reachable": True,
        "proposal_proxy_positive": False,
        "proposal_proxy_label_valid": True,
    })
    validate_candidate_set(value)
    return value


def invalid_rollout_record():
    value = synthetic_record(positive=False)
    labels = value["candidates"][1]["labels"]
    labels.update({
        "geodesic_progress_h8_m": 0.0,
        "geodesic_progress_h24_m": 0.0,
        "advantage_h24_m": 0.0,
        "harm": False,
        "useful": False,
        "reachable": False,
        "collision_h8": False,
        "regression_h24": False,
        "rollout_label_valid": False,
    })
    validate_candidate_set(value)
    return value


def invalid_coverage_record():
    value = synthetic_record(positive=False)
    value["set_labels"].update({
        "candidate_universe_has_positive": False,
        "candidate_coverage_miss": False,
        "coverage_label_valid": False,
    })
    validate_candidate_set(value)
    return value


class NLSRVectorizationTest(unittest.TestCase):
    def test_strict_allow_list_vectorization_preserves_masks_and_padding(self):
        short = synthetic_record(positive=True)
        long = record_with_two_residuals()
        long["provenance"]["state_id"] = "scene-a/episode-0/state-1"
        long["provenance"]["plan_index"] = 1
        spec = feature_spec_from_dataset([short])
        batch = vectorize_candidate_sets([short, long], feature_spec=spec)

        self.assertEqual(batch.candidate_features.shape, (2, 4, 29))
        self.assertEqual(batch.set_features.shape, (2, 12))
        self.assertEqual(batch.candidate_ids[0], (
            "native", "frontier-1", "dustbin"))
        self.assertTrue(batch.valid_mask[0].tolist() == [
            True, True, True, False])
        self.assertTrue(batch.native_mask[0].tolist() == [
            True, False, False, False])
        self.assertTrue(batch.dustbin_mask[0].tolist() == [
            False, False, True, False])
        self.assertTrue(batch.selectable_mask[0].tolist() == [
            True, True, False, False])
        candidate_presence = spec.field_slice(
            "feature_presence_mask", candidate=True)
        set_presence = spec.field_slice(
            "feature_presence_mask", candidate=False)
        self.assertEqual(
            batch.candidate_features[0, 1, candidate_presence].tolist(),
            [1.0] * 7)
        self.assertEqual(
            batch.set_features[0, set_presence].tolist(), [1.0] * 6)

    def test_labels_never_change_model_features(self):
        positive = synthetic_record(positive=True)
        neutral = synthetic_record(positive=False)
        spec = feature_spec_from_dataset([positive])
        positive_batch = vectorize_candidate_sets(
            [positive], feature_spec=spec)
        neutral_batch = vectorize_candidate_sets(
            [neutral], feature_spec=spec)
        self.assertTrue(torch.equal(
            positive_batch.candidate_features,
            neutral_batch.candidate_features))
        self.assertTrue(torch.equal(
            positive_batch.set_features, neutral_batch.set_features))
        self.assertFalse(torch.equal(
            positive_batch.advantage_target,
            neutral_batch.advantage_target))

    def test_fixed_21d_native_relation_is_preserved_without_hardcoding(self):
        value = synthetic_record(positive=True)
        expected = [float(index) / 20.0 for index in range(21)]
        for candidate in value["candidates"]:
            candidate["features"]["native_proposal_relation"] = (
                [0.0] * 21
                if candidate["candidate_type"] == "dustbin"
                else list(expected))
        spec = feature_spec_from_dataset([value])
        relation_slice = spec.field_slice(
            "native_proposal_relation", candidate=True)
        self.assertEqual(relation_slice.stop - relation_slice.start, 21)
        batch = vectorize_candidate_sets([value], feature_spec=spec)
        self.assertTrue(torch.allclose(
            batch.candidate_features[0, 1, relation_slice],
            torch.tensor(expected, dtype=torch.float32),
            atol=0.0,
            rtol=0.0,
        ))
        model = NLSRSetRanker(spec)
        model.assert_lightweight_parameter_budget()
        self.assertEqual(model.forward_batch(batch).rank_score.shape, (1, 3))

    def test_schema_violation_fails_before_vectorization(self):
        value = synthetic_record()
        value["candidates"][1]["features"]["oracle_distance"] = 0.0
        with self.assertRaisesRegex(Exception, "extra=.*oracle_distance"):
            vectorize_candidate_sets([value])

    def test_batch_rejects_split_mixing_and_duplicate_decisions(self):
        first = synthetic_record()
        duplicate = copy.deepcopy(first)
        spec = feature_spec_from_dataset([first])
        with self.assertRaisesRegex(NLSRRankerError, "repeats decision"):
            vectorize_candidate_sets([first, duplicate], feature_spec=spec)

        development = synthetic_record(
            scene="scene-development", role="development",
            episode="scene-development/episode-0")
        with self.assertRaisesRegex(NLSRRankerError, "mixes train/development"):
            vectorize_candidate_sets([first, development], feature_spec=spec)

    def test_feature_spec_round_trip_and_hash_tamper(self):
        spec = feature_spec_from_dataset([synthetic_record()])
        restored = RankerFeatureSpec.from_dict(spec.to_dict())
        self.assertEqual(spec, restored)
        self.assertEqual(
            spec.schema_contract_sha256,
            restored.schema_contract_sha256)
        tampered = copy.deepcopy(spec.to_dict())
        tampered["schema_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(NLSRRankerError, "contract hash"):
            RankerFeatureSpec.from_dict(tampered)


class NLSRModelTest(unittest.TestCase):
    def setUp(self):
        self.value = record_with_two_residuals()
        self.spec = feature_spec_from_dataset([self.value])
        self.batch = vectorize_candidate_sets(
            [self.value], feature_spec=self.spec)

    def test_default_model_is_lightweight_and_deterministic(self):
        torch.manual_seed(31)
        expected_first = torch.rand(1)
        expected_second = torch.rand(1)
        torch.manual_seed(31)
        actual_first = torch.rand(1)
        first = NLSRSetRanker(self.spec)
        actual_second = torch.rand(1)
        second = NLSRSetRanker(self.spec)
        self.assertTrue(torch.equal(expected_first, actual_first))
        self.assertTrue(torch.equal(expected_second, actual_second))
        first.assert_lightweight_parameter_budget()
        self.assertEqual(first.parameter_count, 105_077)
        for first_parameter, second_parameter in zip(
                first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_forward_shapes_and_native_dustbin_rank_mask(self):
        model = NLSRSetRanker(self.spec)
        output = model.forward_batch(self.batch)
        self.assertEqual(output.advantage_mean.shape, (1, 4))
        self.assertEqual(output.advantage_log_scale.shape, (1, 4))
        self.assertEqual(output.harm_logit.shape, (1, 4))
        self.assertEqual(output.rank_score.shape, (1, 4))
        self.assertEqual(output.coverage_logit.shape, (1,))
        masked = output.masked_rank_score()
        self.assertTrue(torch.isfinite(masked[0, 0:3]).all())
        self.assertTrue(torch.isneginf(masked[0, 3]))
        self.assertTrue(torch.isneginf(masked[0, -1]))
        self.assertTrue(torch.all(
            output.advantage_log_scale >= -4.0))
        self.assertTrue(torch.all(
            output.advantage_log_scale <= 2.0))

    def test_residual_permutation_equivariance_and_set_invariance(self):
        permuted = copy.deepcopy(self.value)
        permuted["candidates"][1], permuted["candidates"][2] = (
            permuted["candidates"][2], permuted["candidates"][1])
        validate_candidate_set(permuted)
        model = NLSRSetRanker(self.spec).eval()
        original_output = model.forward_batch(self.batch)
        permuted_batch = vectorize_candidate_sets(
            [permuted], feature_spec=self.spec)
        permuted_output = model.forward_batch(permuted_batch)

        original_by_id = {
            candidate_id: original_output.rank_score[0, index]
            for index, candidate_id in enumerate(self.batch.candidate_ids[0])
        }
        permuted_by_id = {
            candidate_id: permuted_output.rank_score[0, index]
            for index, candidate_id in enumerate(
                permuted_batch.candidate_ids[0])
        }
        for candidate_id in original_by_id:
            self.assertTrue(torch.allclose(
                original_by_id[candidate_id],
                permuted_by_id[candidate_id],
                atol=1e-7,
                rtol=0.0,
            ))
        self.assertTrue(torch.allclose(
            original_output.coverage_logit,
            permuted_output.coverage_logit,
            atol=1e-7,
            rtol=0.0,
        ))

    def test_one_training_step_has_finite_nonzero_gradients(self):
        negative = negative_harm_record()
        negative["provenance"].update({
            "scene_id": "scene-negative",
            "episode_id": "scene-negative/episode-0",
            "session_id": "scene-negative/episode-0/session-0",
            "group_id": "scene-negative/episode-0",
            "state_id": "scene-negative/episode-0/state-0",
            "goal_source_episode_id": "scene-negative/episode-0",
            "environment_id": "environment/scene-negative",
        })
        batch = vectorize_candidate_sets(
            [self.value, negative], feature_spec=self.spec)
        model = NLSRSetRanker(self.spec)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output = model.forward_batch(batch)
        loss = compute_nlsr_losses(output, batch)
        self.assertTrue(torch.isfinite(loss.total))
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(item).all() for item in gradients))
        self.assertGreater(
            sum(float(item.abs().sum()) for item in gradients), 0.0)
        optimizer.step()

    def test_bad_structural_masks_fail_closed(self):
        model = NLSRSetRanker(self.spec)
        bad_native = self.batch.native_mask.clone()
        bad_native[:] = False
        with self.assertRaisesRegex(NLSRRankerError, "one native"):
            model(
                self.batch.candidate_features,
                self.batch.set_features,
                self.batch.valid_mask,
                bad_native,
                self.batch.dustbin_mask,
            )

    def test_finite_extreme_input_cannot_return_nonfinite_logits(self):
        model = NLSRSetRanker(self.spec)
        features = self.batch.candidate_features.clone()
        relation = self.spec.field_slice(
            "goal_patch_relation", candidate=True)
        features[..., relation] = torch.finfo(torch.float32).max
        with self.assertRaisesRegex(NLSRRankerError, "output became non-finite"):
            model(
                features,
                self.batch.set_features,
                self.batch.valid_mask,
                self.batch.native_mask,
                self.batch.dustbin_mask,
            )

    def test_numeric_configs_reject_boolean_bounds_and_weights(self):
        with self.assertRaisesRegex(NLSRRankerError, "finite real"):
            NLSRRankerConfig(advantage_log_scale_min=False)
        with self.assertRaisesRegex(NLSRRankerError, "non-negative"):
            NLSRLossConfig(rank_weight=True)

    def test_loss_rejects_output_from_a_different_masked_batch(self):
        model = NLSRSetRanker(self.spec)
        output = model.forward_batch(self.batch)
        wrong = replace(output, residual_mask=output.native_mask)
        with self.assertRaisesRegex(NLSRRankerError, "residual_mask differs"):
            compute_nlsr_losses(wrong, self.batch)


class NLSRLossTest(unittest.TestCase):
    def test_invalid_rollout_candidate_is_masked_from_all_candidate_losses(self):
        value = invalid_rollout_record()
        spec = feature_spec_from_dataset([value])
        batch = vectorize_candidate_sets([value], feature_spec=spec)
        model = NLSRSetRanker(spec)
        output = model.forward_batch(batch)
        reference = compute_nlsr_losses(output, batch)

        advantage = output.advantage_mean.clone()
        harm = output.harm_logit.clone()
        rank = output.rank_score.clone()
        advantage[0, 1] = 100.0
        harm[0, 1] = -100.0
        rank[0, 1] = 100.0
        changed = replace(
            output,
            advantage_mean=advantage,
            harm_logit=harm,
            rank_score=rank,
        )
        changed_loss = compute_nlsr_losses(changed, batch)
        self.assertEqual(reference.advantage_count, 0)
        self.assertEqual(reference.harm_count, 0)
        self.assertEqual(float(reference.advantage.detach()), 0.0)
        self.assertEqual(float(reference.harm.detach()), 0.0)
        self.assertTrue(torch.allclose(reference.total, changed_loss.total))

    def test_listwise_target_prefers_native_over_harmful_regression(self):
        value = negative_harm_record()
        spec = feature_spec_from_dataset([value])
        batch = vectorize_candidate_sets([value], feature_spec=spec)
        model = NLSRSetRanker(spec)
        output = model.forward_batch(batch)
        native_high = replace(
            output,
            rank_score=torch.tensor([[5.0, -5.0, 0.0]]),
        )
        residual_high = replace(
            output,
            rank_score=torch.tensor([[-5.0, 5.0, 0.0]]),
        )
        native_loss = compute_nlsr_losses(native_high, batch).rank
        residual_loss = compute_nlsr_losses(residual_high, batch).rank
        self.assertLess(float(native_loss), float(residual_loss))

    def test_invalid_residual_only_set_does_not_dilute_listwise_loss(self):
        valid = synthetic_record(positive=True)
        invalid = invalid_rollout_record()
        invalid["provenance"].update({
            "scene_id": "scene-invalid-rank",
            "episode_id": "scene-invalid-rank/episode-0",
            "session_id": "scene-invalid-rank/episode-0/session-0",
            "group_id": "scene-invalid-rank/episode-0",
            "state_id": "scene-invalid-rank/episode-0/state-0",
            "goal_source_episode_id": "scene-invalid-rank/episode-0",
            "environment_id": "environment/scene-invalid-rank",
        })
        spec = feature_spec_from_dataset([valid])
        model = NLSRSetRanker(spec)
        valid_batch = vectorize_candidate_sets([valid], feature_spec=spec)
        combined_batch = vectorize_candidate_sets(
            [valid, invalid], feature_spec=spec)
        valid_loss = compute_nlsr_losses(
            model.forward_batch(valid_batch), valid_batch)
        combined_loss = compute_nlsr_losses(
            model.forward_batch(combined_batch), combined_batch)
        self.assertEqual(valid_loss.rank_set_count, 1)
        self.assertEqual(combined_loss.rank_set_count, 1)
        self.assertTrue(torch.allclose(valid_loss.rank, combined_loss.rank))

    def test_harm_bce_uses_positive_class_weight(self):
        value = negative_harm_record()
        spec = feature_spec_from_dataset([value])
        batch = vectorize_candidate_sets([value], feature_spec=spec)
        model = NLSRSetRanker(spec)
        output = model.forward_batch(batch)
        unweighted = compute_nlsr_losses(
            output, batch,
            NLSRLossConfig(harm_positive_weight=1.0)).harm
        weighted = compute_nlsr_losses(
            output, batch,
            NLSRLossConfig(harm_positive_weight=5.0)).harm
        self.assertGreater(
            float(weighted.detach()), float(unweighted.detach()))

    def test_coverage_loss_uses_only_valid_universe_labels(self):
        valid = synthetic_record(
            positive=False, scene="scene-valid",
            episode="scene-valid/episode-0")
        invalid = invalid_coverage_record()
        invalid["provenance"].update({
            "scene_id": "scene-invalid",
            "episode_id": "scene-invalid/episode-0",
            "session_id": "scene-invalid/episode-0/session-0",
            "group_id": "scene-invalid/episode-0",
            "state_id": "scene-invalid/episode-0/state-0",
            "goal_source_episode_id": "scene-invalid/episode-0",
            "environment_id": "environment/scene-invalid",
        })
        spec = feature_spec_from_dataset([valid, invalid])
        batch = vectorize_candidate_sets([valid, invalid], feature_spec=spec)
        model = NLSRSetRanker(spec)
        output = model.forward_batch(batch)
        baseline = compute_nlsr_losses(output, batch)
        logits = output.coverage_logit.clone()
        logits[1] = 100.0
        changed = compute_nlsr_losses(
            replace(output, coverage_logit=logits), batch)
        self.assertEqual(baseline.coverage_count, 1)
        self.assertTrue(torch.allclose(
            baseline.coverage, changed.coverage))

    def test_student_t_regression_is_stable_for_large_finite_residual(self):
        value = synthetic_record(positive=True)
        spec = feature_spec_from_dataset([value])
        batch = vectorize_candidate_sets([value], feature_spec=spec)
        target = batch.advantage_target.clone()
        target[0, 1] = 1e30
        batch = replace(batch, advantage_target=target)
        output = NLSRSetRanker(spec).forward_batch(batch)
        loss = compute_nlsr_losses(output, batch)
        self.assertTrue(torch.isfinite(loss.advantage))
        self.assertTrue(torch.isfinite(loss.total))

    def test_student_t_zero_residual_has_finite_gradient(self):
        value = synthetic_record(positive=True)
        spec = feature_spec_from_dataset([value])
        batch = vectorize_candidate_sets([value], feature_spec=spec)
        output = NLSRSetRanker(spec).forward_batch(batch)
        exact_mean = batch.advantage_target.clone().requires_grad_(True)
        output = replace(output, advantage_mean=exact_mean)
        loss = compute_nlsr_losses(output, batch)
        loss.advantage.backward()
        self.assertIsNotNone(exact_mean.grad)
        self.assertTrue(torch.isfinite(exact_mean.grad).all())


class NLSRSplitAndCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.train = synthetic_record(
            scene="scene-train", role="train",
            episode="scene-train/episode-0")
        self.development = synthetic_record(
            scene="scene-development", role="development",
            episode="scene-development/episode-0")
        self.records = [self.train, self.development]

    def test_declared_split_is_scene_and_group_disjoint(self):
        split = split_by_declared_role(self.records)
        self.assertEqual(split.train_scenes, ("scene-train",))
        self.assertEqual(
            split.development_scenes, ("scene-development",))
        self.assertTrue(set(split.train_groups).isdisjoint(
            split.development_groups))
        leaked = copy.deepcopy(self.train)
        leaked["provenance"]["split_role"] = "development"
        with self.assertRaisesRegex(NLSRRankerError, "scene_id overlap"):
            assert_scene_group_disjoint([self.train], [leaked])

        environment_alias = copy.deepcopy(self.development)
        environment_alias["provenance"].update({
            "environment_id": self.train["provenance"]["environment_id"],
            "navmesh_sha256": self.train["provenance"]["navmesh_sha256"],
        })
        with self.assertRaisesRegex(NLSRRankerError, "environment_id overlap"):
            assert_scene_group_disjoint([self.train], [environment_alias])

    def test_provenance_hash_is_order_invariant_but_content_bound(self):
        digest = dataset_provenance_sha256(self.records)
        self.assertEqual(
            digest,
            dataset_provenance_sha256(list(reversed(self.records))))
        changed = copy.deepcopy(self.records)
        changed[0]["provenance"]["prefix_sha256"] = "b" * 64
        self.assertNotEqual(digest, dataset_provenance_sha256(changed))

    def test_portable_checkpoint_round_trip_and_expected_data_binding(self):
        spec = feature_spec_from_dataset(self.records)
        model = NLSRSetRanker(spec)
        batch = vectorize_candidate_sets([self.train], feature_spec=spec)
        expected = model.forward_batch(batch)
        payload = make_portable_checkpoint(
            model, self.records, extra={"seed": 0, "purpose": "unit-test"})
        metadata = build_checkpoint_metadata(
            model, self.records, extra={"seed": 0, "purpose": "unit-test"})
        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(metadata["parameter_count"], 105_077)
        restored = load_portable_checkpoint(
            payload, expected_records=self.records)
        actual = restored.forward_batch(batch)
        self.assertTrue(torch.equal(
            expected.advantage_mean, actual.advantage_mean))
        self.assertTrue(torch.equal(
            expected.coverage_logit, actual.coverage_logit))

    def test_checkpoint_tamper_or_wrong_provenance_fails_closed(self):
        spec = feature_spec_from_dataset(self.records)
        model = NLSRSetRanker(spec)
        payload = make_portable_checkpoint(model, self.records)
        tampered = copy.deepcopy(payload)
        tampered["metadata"]["schema_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(NLSRRankerError, "metadata hash"):
            load_portable_checkpoint(tampered)

        changed = copy.deepcopy(self.records)
        changed[0]["provenance"]["prefix_sha256"] = "b" * 64
        with self.assertRaisesRegex(NLSRRankerError, "provenance hash"):
            load_portable_checkpoint(payload, expected_records=changed)

        changed_content = copy.deepcopy(self.records)
        changed_content[0]["candidates"][1]["features"][
            "subgoal_forward_m"] += 1.0
        with self.assertRaisesRegex(NLSRRankerError, "content hash"):
            load_portable_checkpoint(
                payload, expected_records=changed_content)

        changed_weights = copy.deepcopy(payload)
        first_key = next(iter(changed_weights["state_dict"]))
        changed_weights["state_dict"][first_key].view(-1)[0] += 1.0
        with self.assertRaisesRegex(NLSRRankerError, "state_dict hash"):
            load_portable_checkpoint(changed_weights)


if __name__ == "__main__":
    unittest.main()
