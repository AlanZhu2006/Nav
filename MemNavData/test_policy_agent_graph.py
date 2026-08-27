import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from NavDP.baselines.memnav.policy_agent import (
    MemNavAgent,
    effective_candidate_ceiling,
)
from NavDP.baselines.memnav.reverse_memory_graph import (
    metric_nodes_between,
    reverse_metric_nodes,
)


class CandidateCeilingTest(unittest.TestCase):
    def test_default_is_frame_before_goal_start(self):
        self.assertEqual(effective_candidate_ceiling(151), 150)

    def test_evaluation_can_only_tighten_ceiling(self):
        self.assertEqual(effective_candidate_ceiling(250, 150), 150)
        with self.assertRaisesRegex(ValueError, "may not exceed"):
            effective_candidate_ceiling(250, 250)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            effective_candidate_ceiling(250, -1)


class LifelongGoalSessionTest(unittest.TestCase):
    @staticmethod
    def make_agent():
        agent = object.__new__(MemNavAgent)
        agent._active_goal_key = None
        agent._goal_session_index = 0
        agent._last_goal_session_started = False
        for name in (
                "_goal_cache", "_anchor_state", "_goal_start_frame",
                "_graph_routes", "_retrieval_verification_cache",
                "_phase_b_rank_cache", "_phase_b_scale_cache",
                "_phase_b_geometry_cache",
                "_certified_relocalization_cache",
                "_pi3x_relocalization_cache",
                "_certified_graph_routes", "_certified_candidate_cache"):
            setattr(agent, name, {})
        agent._certified_reference_depth_cache = {42: "history-only"}
        return agent

    def test_repeated_goal_opens_a_new_session_without_erasing_history_cache(self):
        agent = self.make_agent()
        self.assertTrue(agent._begin_goal_session("goal-a"))
        agent._goal_start_frame["goal-a"] = 10
        agent._goal_cache[("cls", "goal-a")] = "embedding"
        agent._certified_candidate_cache[("goal-a", 9)] = [1]
        agent._certified_relocalization_cache["goal-a"] = {"accepted": False}

        self.assertFalse(agent._begin_goal_session("goal-a"))
        self.assertEqual(agent._goal_start_frame["goal-a"], 10)

        self.assertTrue(agent._begin_goal_session("goal-b"))
        self.assertNotIn("goal-a", agent._goal_start_frame)
        self.assertNotIn(("cls", "goal-a"), agent._goal_cache)
        self.assertNotIn(("goal-a", 9), agent._certified_candidate_cache)
        self.assertNotIn("goal-a", agent._certified_relocalization_cache)
        self.assertEqual(agent._certified_reference_depth_cache, {42: "history-only"})

        agent._goal_start_frame["goal-b"] = 20
        self.assertTrue(agent._begin_goal_session("goal-a"))
        self.assertNotIn("goal-a", agent._goal_start_frame)
        self.assertNotIn("goal-b", agent._goal_start_frame)
        self.assertEqual(agent.goal_session_status(), {
            "goal_session_index": 3,
            "goal_session_started": True,
            "long_term_memory_preserved": True,
        })

    def test_replay_goal_session_restores_only_the_query_boundary(self):
        agent = self.make_agent()
        agent.n = 80
        history_cache = dict(agent._certified_reference_depth_cache)
        receipt = agent.replay_goal_session(b"goal-c", 80)
        goal_key = hashlib.md5(b"goal-c").hexdigest()
        self.assertEqual(agent._goal_start_frame[goal_key], 80)
        self.assertEqual(receipt["candidate_ceiling"], 79)
        self.assertEqual(receipt["frame_count"], 80)
        self.assertFalse(receipt["diffusion_sampled"])
        self.assertFalse(receipt["memory_appended"])
        self.assertEqual(agent._certified_reference_depth_cache, history_cache)
        with self.assertRaisesRegex(ValueError, "current frame"):
            agent.replay_goal_session(b"another-goal", 79)


class EarlyCertifiedShortlistTest(unittest.TestCase):
    class FakeLingBot:
        @staticmethod
        def load_images(_paths):
            return torch.zeros((1, 3, 4, 4), dtype=torch.float32)

        @staticmethod
        def dino(_images):
            return {"cls": torch.tensor([[1.0, 0.0]])}

    def test_certificate_shortlist_does_not_wait_for_decoder_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent = object.__new__(MemNavAgent)
            agent.S = 8
            agent.W = 32
            agent.n = 12
            agent.device = torch.device("cpu")
            agent.rgb_dir = temporary
            agent.certified_relocalization_matcher = object()
            agent.lb = self.FakeLingBot()
            agent._goal_cache = {}
            agent._certified_candidate_cache = {}
            agent.dino_cls = [
                torch.tensor([0.0, 1.0]) for _ in range(agent.n)]
            agent.dino_cls[8] = torch.tensor([0.8, 0.2])
            agent.dino_cls[10] = torch.tensor([1.0, 0.0])

            candidates, current_cosine = (
                agent._certified_shortlist_before_decoder_warmup(
                    b"goal", "goal-key", frame_index=11,
                    candidate_ceiling=10))

            self.assertLess(agent.n, agent.S + agent.W + 1)
            self.assertEqual(candidates[0]["anchor"], 10)
            self.assertTrue(all(8 <= row["anchor"] <= 10
                                for row in candidates))
            self.assertEqual(current_cosine, 0.0)


class CertifiedReferenceDepthCacheTest(unittest.TestCase):
    def test_same_immutable_anchor_reuses_exact_final_arrays(self):
        agent = object.__new__(MemNavAgent)
        agent.S = 8
        agent._certified_reference_depth_cache = {}
        agent._certified_dense_replay_last_stats = None
        agent._certified_eager_depth_cached_anchors = set()
        calls = []

        def replay(anchor):
            calls.append(anchor)
            return (
                np.full((2, 3), anchor, dtype=np.float32),
                np.full((2, 3), anchor + 1, dtype=np.float32),
            )

        agent._certified_reference_depth_impl = replay
        first = agent._certified_reference_depth(42)
        second = agent._certified_reference_depth(42)

        self.assertEqual(calls, [42])
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertIsNot(first[0], second[0])
        self.assertTrue(agent._certified_dense_replay_last_stats["cache_hit"])
        self.assertEqual(
            agent._certified_dense_replay_last_stats["replayed_frames"], 0)


class CertifiedAnchorImageAuthorizationTest(unittest.TestCase):
    @staticmethod
    def make_agent(root, goal, *, accepted=True):
        anchor = 12
        image = b"immutable-certified-history-jpeg"
        (root / f"{anchor}.jpg").write_bytes(image)
        digest = hashlib.sha256(image).hexdigest()
        key = hashlib.md5(goal).hexdigest()
        agent = object.__new__(MemNavAgent)
        agent.S = 8
        agent.n = 24
        agent.rgb_dir = str(root)
        agent._goal_start_frame = {key: 20}
        agent._certified_relocalization_cache = {key: {
            "result": {
                "accepted": accepted,
                "selected_anchor": anchor,
                "selected_anchor_image_sha256": digest,
            },
        }}
        return agent, anchor, image, digest

    def test_returns_only_hash_bound_anchor_from_accepted_proof(self):
        goal = b"goal-jpeg"
        with tempfile.TemporaryDirectory() as temporary:
            agent, anchor, image, digest = self.make_agent(
                Path(temporary), goal)
            record = agent.certified_anchor_image(
                goal, anchor, expected_sha256=digest)
            self.assertEqual(record["image"], image)
            self.assertEqual(record["sha256"], digest)

    def test_rejects_wrong_goal_anchor_digest_and_abstention(self):
        goal = b"goal-jpeg"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent, anchor, _image, digest = self.make_agent(root, goal)
            bad_calls = (
                (b"other-goal", anchor, digest),
                (goal, anchor + 1, digest),
                (goal, anchor, "0" * 64),
            )
            for goal_bytes, index, sha in bad_calls:
                with self.assertRaises(ValueError):
                    agent.certified_anchor_image(
                        goal_bytes, index, expected_sha256=sha)
            agent, anchor, _image, digest = self.make_agent(
                root, goal, accepted=False)
            with self.assertRaisesRegex(ValueError, "did not authorize"):
                agent.certified_anchor_image(
                    goal, anchor, expected_sha256=digest)


class BidirectionalMemoryGraphTest(unittest.TestCase):
    def test_resamples_forward_and_reverse_recorded_arcs(self):
        poses = np.zeros((7, 3), dtype=np.float64)
        poses[:, 0] = np.arange(7, dtype=np.float64)
        self.assertEqual(
            metric_nodes_between(
                poses, start_index=1, target_index=6,
                metric_scale=1.0, spacing_m=2.0),
            (3, 5, 6),
        )
        self.assertEqual(
            metric_nodes_between(
                poses, start_index=6, target_index=1,
                metric_scale=1.0, spacing_m=2.0),
            (4, 2, 1),
        )

    def test_reverse_wrapper_keeps_old_contract(self):
        poses = np.zeros((4, 3), dtype=np.float64)
        poses[:, 0] = np.arange(4, dtype=np.float64)
        self.assertEqual(
            reverse_metric_nodes(
                poses, start_index=3, anchor_index=0,
                metric_scale=1.0, spacing_m=2.0),
            (1, 0),
        )
        with self.assertRaisesRegex(ValueError, "indices must satisfy"):
            reverse_metric_nodes(
                poses, start_index=1, anchor_index=3,
                metric_scale=1.0, spacing_m=2.0)


class _RelativePoseCore:
    @staticmethod
    def build_revisit(current, target, metric_scale):
        delta = (target[:, 0] - current[:, 0]) * metric_scale
        aux = torch.stack([delta, torch.zeros_like(delta)], dim=-1)
        return torch.zeros((1, 1, 1)), aux, torch.eye(3)[None]


class GraphConditionedPoseTest(unittest.TestCase):
    def make_agent(self):
        agent = object.__new__(MemNavAgent)
        agent.graph_subgoal_spacing_m = 2.0
        agent.graph_subgoal_arrival_m = 0.60
        agent._graph_routes = {}
        agent.core = _RelativePoseCore()
        agent.device = torch.device("cpu")
        return agent

    def query(self, agent, current_x):
        poses = torch.zeros((7, 9), dtype=torch.float32)
        poses[:, 0] = torch.arange(7, dtype=torch.float32)
        current = torch.zeros((1, 9), dtype=torch.float32)
        current[:, 0] = current_x
        return agent._graph_conditioned_pose(
            goal_key="goal",
            cache={"cam_pose_enc": poses},
            current_pose=current,
            goal_aux_pose=torch.tensor([[0.2, 0.0]]),
            anchor=0,
            goal_start_frame=7,
            metric_scale=torch.tensor([1.0]),
        )

    def test_follows_nodes_then_returns_to_image_goal(self):
        agent = self.make_agent()
        target, diag = self.query(agent, 6.0)
        self.assertEqual(target.tolist(), [[-2.0, 0.0]])
        self.assertEqual(diag["graph_subgoal_node"], 4)

        target, diag = self.query(agent, 4.0)
        self.assertEqual(target.tolist(), [[-2.0, 0.0]])
        self.assertEqual(diag["graph_subgoal_node"], 2)

        target, diag = self.query(agent, 2.0)
        self.assertEqual(target.tolist(), [[-2.0, 0.0]])
        self.assertEqual(diag["graph_subgoal_node"], 0)

        target, diag = self.query(agent, 0.0)
        self.assertAlmostEqual(target[0, 0].item(), 0.2, places=6)
        self.assertTrue(diag["graph_subgoal_complete"])

    def test_zero_spacing_is_exact_direct_goal(self):
        agent = self.make_agent()
        agent.graph_subgoal_spacing_m = 0.0
        target, diag = self.query(agent, 6.0)
        self.assertAlmostEqual(target[0, 0].item(), 0.2, places=6)
        self.assertFalse(diag["graph_subgoal_enabled"])


class CertifiedStagnationGraphTest(unittest.TestCase):
    def make_agent(self, current_x):
        agent = object.__new__(MemNavAgent)
        agent.graph_subgoal_spacing_m = 2.0
        agent.graph_subgoal_arrival_m = 0.60
        agent._certified_graph_routes = {}
        agent._metric_scale = 1.0
        agent.n = 8
        agent.S = 1
        agent.core = _RelativePoseCore()
        agent.device = torch.device("cpu")
        poses = []
        for value in range(7):
            pose = torch.zeros(9, dtype=torch.float32)
            pose[0] = float(value)
            poses.append(pose)
        current = torch.zeros(9, dtype=torch.float32)
        current[0] = float(current_x)
        poses.append(current)
        agent.cam_pose = poses
        return agent

    def test_default_off_is_exact_direct_bearing(self):
        agent = self.make_agent(6.0)
        direct = [0.25, -0.75]
        value, diag = agent._certified_graph_direction(
            goal_key="goal", direct_bearing=direct, target_anchor=0,
            goal_start_frame=7, graph_rescue=False)
        self.assertEqual(value, direct)
        self.assertFalse(diag["certified_graph_rescue_active"])
        self.assertEqual(agent._certified_graph_routes, {})

    def test_rescue_uses_nearest_history_node_and_can_move_forward(self):
        agent = self.make_agent(1.0)
        value, diag = agent._certified_graph_direction(
            goal_key="goal", direct_bearing=[9.0, 0.0], target_anchor=5,
            goal_start_frame=7, route_start_anchor=1,
            graph_rescue=True)
        self.assertEqual(value, [2.0, 0.0])
        self.assertTrue(diag["certified_graph_rescue_active"])
        self.assertEqual(diag["certified_graph_route_start_node"], 1)
        self.assertEqual(diag["certified_graph_node"], 3)
        self.assertEqual(diag["certified_graph_temporal_direction"], "forward")

    def test_empty_route_falls_back_to_direct_without_active_graph(self):
        agent = self.make_agent(1.0)
        direct = [0.5, -0.25]
        value, diag = agent._certified_graph_direction(
            goal_key="goal", direct_bearing=direct, target_anchor=1,
            goal_start_frame=7, graph_rescue=True)
        self.assertEqual(value, direct)
        self.assertFalse(diag["certified_graph_rescue_active"])
        self.assertTrue(diag["certified_graph_complete"])
        self.assertEqual(diag["certified_graph_count"], 0)
        self.assertEqual(
            diag["certified_graph_reason"], "route_complete_direct_bearing")

    def test_route_contract_change_fails_closed_to_direct(self):
        agent = self.make_agent(6.0)
        agent._certified_graph_direction(
            goal_key="goal", direct_bearing=[9.0, 0.0], target_anchor=0,
            goal_start_frame=7, graph_rescue=True)
        value, diag = agent._certified_graph_direction(
            goal_key="goal", direct_bearing=[0.5, 0.5], target_anchor=0,
            goal_start_frame=7, route_start_anchor=2,
            graph_rescue=True)
        self.assertEqual(value, [0.5, 0.5])
        self.assertEqual(diag["certified_graph_reason"], "route_contract_changed")

    def test_late_metric_scale_restores_live_stream_caches(self):
        class FakeAggregate:
            def __init__(self):
                self.kv_cache = {"key": torch.tensor([1.0])}
                self.total_frames_processed = 17

        class FakeCameraHead:
            def __init__(self):
                self.kv_cache = [{"camera": torch.tensor([2.0])}]
                self.frame_idx = 9

        class FakeModel:
            def __init__(self):
                self.camera_head = FakeCameraHead()

            def clean_kv_cache(self):
                aggregate.kv_cache.clear()
                aggregate.total_frames_processed = 0
                self.camera_head.kv_cache = None
                self.camera_head.frame_idx = 0

        class FakeLingBot:
            def __init__(self):
                self.agg = aggregate
                self.model = FakeModel()

        aggregate = FakeAggregate()
        agent = object.__new__(MemNavAgent)
        agent.lb = FakeLingBot()
        agent._metric_scale = None
        agent.n = 2
        agent.S = 1
        expected_aggregate_tensor = aggregate.kv_cache["key"]
        expected_camera_tensor = agent.lb.model.camera_head.kv_cache[0]["camera"]

        def destructive_scale():
            agent.lb.model.clean_kv_cache()
            agent._metric_scale = 1.75
            return agent._metric_scale

        with mock.patch.object(
                agent, "_get_metric_scale", side_effect=destructive_scale):
            value = agent._get_metric_scale_preserving_stream()

        self.assertEqual(value, 1.75)
        self.assertIs(aggregate.kv_cache["key"], expected_aggregate_tensor)
        self.assertEqual(aggregate.total_frames_processed, 17)
        camera_head = agent.lb.model.camera_head
        self.assertIs(camera_head.kv_cache[0]["camera"], expected_camera_tensor)
        self.assertEqual(camera_head.frame_idx, 9)


class CertifiedRelocalizationLifecycleTest(unittest.TestCase):
    def test_short_history_exposes_and_freezes_empty_shortlist(self):
        goal = b"short-history-goal"
        key = hashlib.md5(goal).hexdigest()
        agent = object.__new__(MemNavAgent)
        agent.n = 12
        agent.S = 8
        agent.W = 32
        agent.amargin = 39
        agent._goal_start_frame = {}
        agent.certified_relocalization_matcher = object()
        agent._certified_candidate_cache = {}

        receipt = agent.plan(goal, retrieval_only=True)
        self.assertEqual(receipt["goal_start_frame"], 11)
        self.assertEqual(receipt["candidate_ceiling"], 10)
        self.assertEqual(receipt["certified_visual_candidates"], [])
        self.assertEqual(agent._certified_candidate_cache[(key, 10)], [])

    def test_empty_causal_shortlist_is_one_cached_abstention(self):
        goal = b"goal-jpeg-placeholder"
        key = hashlib.md5(goal).hexdigest()
        agent = object.__new__(MemNavAgent)
        agent.n = 12
        agent.certified_relocalization_matcher = object()
        agent._goal_start_frame = {key: 11}
        agent._certified_relocalization_cache = {}

        first = agent.certified_relocalize(goal, [])
        self.assertTrue(first["ok"])
        self.assertFalse(first["accepted"])
        self.assertFalse(first["cached"])
        self.assertEqual(first["reason"], "no_causal_candidate")
        self.assertEqual(first["candidate_count"], 0)

        agent.n += 1
        second = agent.certified_relocalize(goal, [])
        self.assertTrue(second["ok"])
        self.assertFalse(second["accepted"])
        self.assertTrue(second["cached"])
        self.assertEqual(second["reason"], "no_causal_candidate")
        self.assertEqual(second["frame_idx"], 12)


class LearnedPi3XRelocalizationLifecycleTest(unittest.TestCase):
    class FakeRuntime:
        def __init__(self, *, accept=True):
            self.accept = accept
            self.calls = []

        def relocalize(self, **kwargs):
            candidates = [dict(item) for item in kwargs["candidates"]]
            self.calls.append({**kwargs, "candidates": candidates})
            selected = candidates[-1]
            if not self.accept:
                return {
                    "ok": True,
                    "accepted": False,
                    "reason": "learned_spatial_proof_below_consensus_native_fallback",
                    "selected_anchor": selected["anchor"],
                    "selected_dino_rank": selected["dino_rank"],
                    "candidate_count": len(candidates),
                    "ranked_candidates": candidates,
                }
            return {
                "ok": True,
                "accepted": True,
                "reason": "learned_spatial_proof_accepted",
                "selected_anchor": selected["anchor"],
                "selected_dino_rank": selected["dino_rank"],
                "aux_pose": [0.6, -0.8],
                "direction_vector": [0.6, -0.8],
                "pointgoal_units": "pi3x_current_camera_direction_only",
                "candidate_count": len(candidates),
                "ranked_candidates": candidates,
            }

    @staticmethod
    def make_agent(temporary, runtime, goal):
        agent = object.__new__(MemNavAgent)
        agent.n = 21
        agent.rgb_dir = temporary
        agent.pi3x_online_relocalizer = runtime
        agent._goal_start_frame = {hashlib.md5(goal).hexdigest(): 20}
        agent._pi3x_relocalization_cache = {}
        return agent

    def test_acceptance_fixes_anchor_and_reinfers_only_that_anchor(self):
        goal = b"learned-goal"
        candidates = [
            {"anchor": 8, "score": 0.95},
            {"anchor": 12, "score": 0.91},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.FakeRuntime(accept=True)
            agent = self.make_agent(temporary, runtime, goal)
            first = agent.learned_pi3x_relocalize(goal, candidates)
            self.assertTrue(first["accepted"])
            self.assertFalse(first["initial_candidate_selection_cached"])
            self.assertEqual(first["selected_anchor"], 12)

            agent.n += 1
            second = agent.learned_pi3x_relocalize(goal, candidates)
            self.assertTrue(second["accepted"])
            self.assertTrue(second["initial_candidate_selection_cached"])
            self.assertEqual([len(call["candidates"])
                              for call in runtime.calls], [2, 1])
            self.assertEqual(runtime.calls[1]["candidates"], [{
                "anchor": 12, "score": 0.91, "dino_rank": 2,
            }])
            self.assertEqual(second["pointgoal_units"],
                             "pi3x_current_camera_direction_only")

    def test_initial_reject_is_cached_and_never_retried(self):
        goal = b"learned-reject-goal"
        candidates = [{"anchor": 8, "score": 0.95}]
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.FakeRuntime(accept=False)
            agent = self.make_agent(temporary, runtime, goal)
            first = agent.learned_pi3x_relocalize(goal, candidates)
            self.assertFalse(first["accepted"])
            self.assertFalse(first["cached"])

            agent.n += 1
            second = agent.learned_pi3x_relocalize(goal, candidates)
            self.assertFalse(second["accepted"])
            self.assertTrue(second["cached"])
            self.assertTrue(second["initial_candidate_selection_cached"])
            self.assertEqual(len(runtime.calls), 1)

    def test_changed_frozen_shortlist_fails_closed(self):
        goal = b"learned-contract-goal"
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.FakeRuntime(accept=True)
            agent = self.make_agent(temporary, runtime, goal)
            first = agent.learned_pi3x_relocalize(
                goal, [{"anchor": 8, "score": 0.95}])
            self.assertTrue(first["accepted"])
            changed = agent.learned_pi3x_relocalize(
                goal, [{"anchor": 8, "score": 0.94}])
            self.assertFalse(changed["accepted"])
            self.assertEqual(changed["reason"], "candidate_contract_changed")
            self.assertEqual(len(runtime.calls), 1)


class _FakeDino:
    patch_size = 14

    @staticmethod
    def load_images(paths):
        return torch.zeros((len(paths), 3, 8, 8), dtype=torch.float32)

    @staticmethod
    def dino(images):
        generator = torch.Generator().manual_seed(17)
        return {"patch": torch.randn(
            len(images), 1369, 4, generator=generator)}


class _RecordingPairwiseRanker:
    def __init__(self):
        self.calls = 0

    def status(self):
        return {"enabled": True, "authority": "rank_only"}

    def rank_pooled_tokens(self, query, memory, cosine, anchors,
                           **kwargs):
        self.calls += 1
        assert query.shape == (64, 4)
        assert memory.shape == (2, 64, 4)
        assert cosine == [0.9, 0.8]
        assert anchors == [8, 9]
        assert "bearing_vectors" in kwargs
        return {
            "selected_anchor": 9,
            "activation_authorized": False,
        }


class CDECPairwiseAgentBoundaryTest(unittest.TestCase):
    def test_proposal_uses_causal_jpegs_and_has_no_activation_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            goal = root / "goal.jpg"
            goal.write_bytes(b"goal")
            for anchor in (8, 9):
                (root / f"{anchor}.jpg").write_bytes(b"memory")
            agent = object.__new__(MemNavAgent)
            agent.rgb_dir = str(root)
            agent.lb = _FakeDino()
            agent.cdec_pairwise_ranker = _RecordingPairwiseRanker()
            poses = []
            for index in range(11):
                pose = torch.zeros(9)
                pose[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
                pose[2] = -float(11 - index)
                poses.append(pose)
            agent.cam_pose = poses
            result = agent._cdec_pairwise_proposal(goal, [
                {"anchor": 8, "score": 0.9},
                {"anchor": 9, "score": 0.8},
            ])
            self.assertEqual(result["selected_anchor"], 9)
            self.assertFalse(result["activation_authorized"])
            self.assertEqual(agent.cdec_pairwise_ranker.calls, 1)

    def test_status_advertises_ranker_separately_from_certificate(self):
        agent = object.__new__(MemNavAgent)
        agent.certified_relocalization_matcher = object()
        agent.cdec_pairwise_ranker = _RecordingPairwiseRanker()
        status = agent.certified_relocalization_status()
        self.assertTrue(status["enabled"])
        self.assertEqual(
            status["learned_rescue_proposal"]["authority"], "rank_only")


class _CertificateMatcher:
    @staticmethod
    def match_paths(reference, _goal, **_kwargs):
        anchor = int(Path(reference).stem)
        points = np.asarray([
            [float(anchor), 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [3.0, 1.0],
        ], dtype=np.float32)
        return {
            "reference_raw_points": points,
            "query_raw_points": points.copy(),
            "reference_points": points,
            "query_points": points.copy(),
            "scores": np.ones(len(points), dtype=np.float32),
            "reference_raw_hw": (518, 518),
            "query_raw_hw": (518, 518),
        }


class CertifiedCascadeRuntimeTest(unittest.TestCase):
    def make_agent(self, root):
        for anchor in (8, 9):
            (root / f"{anchor}.jpg").write_bytes(b"memory")
        goal = b"cascade-goal"
        key = hashlib.md5(goal).hexdigest()
        agent = object.__new__(MemNavAgent)
        agent.S = 8
        agent.n = 10
        agent.rgb_dir = str(root)
        agent.lb = type("LB", (), {"patch_size": 14})()
        agent.certified_relocalization_matcher = _CertificateMatcher()
        agent.cdec_pairwise_ranker = object()
        agent._goal_start_frame = {key: 10}
        agent._certified_relocalization_cache = {}
        poses = []
        for index in range(10):
            pose = torch.zeros(9)
            pose[0] = float(index)
            pose[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
            poses.append(pose)
        agent.cam_pose = poses
        agent._certified_reference_depth = lambda _anchor: (
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        )
        agent._certified_bearing_vector = lambda _pose: [1.0, 0.0]
        return agent, goal

    @staticmethod
    def support(reference, *_args, **_kwargs):
        # Anchor 8 is the frozen geometry top-1.
        anchor = int(round(float(reference[0, 0])))
        return {
            "lightglue_matches": 32,
            "lightglue_score_median": 1.0,
            "fundamental_inliers": 40 if anchor == 8 else 32,
            "fundamental_inlier_ratio": 1.0,
            "fundamental_query_grid_coverage": 0.5,
            "fundamental_query_hull_coverage": 0.5,
            "fundamental_reference_grid_coverage": 0.5,
            "fundamental_reference_hull_coverage": 0.5,
        }

    @staticmethod
    def accepted_pose(reference_pose):
        anchor = int(round(float(reference_pose[0])))
        return {
            "status": "ok",
            "inliers": 32,
            "query_inlier_coverage": 0.5,
            "reference_inlier_coverage": 0.5,
            "reprojection_rmse_px": 1.0,
            "pose9": [float(anchor), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                      0.0, 0.0],
        }

    def invoke(self, agent, goal, pnp):
        with mock.patch(
                "MemNavData.certified_relocalization_runtime.fundamental_support",
                side_effect=self.support), mock.patch(
                "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                side_effect=pnp), mock.patch(
                "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                side_effect=lambda value: value):
            return agent.certified_relocalize(goal, [
                {"anchor": 8, "score": 0.9},
                {"anchor": 9, "score": 0.8},
            ], allow_learned_rescue=True)

    def test_explicit_goal_intrinsic_is_mapped_and_used_by_pnp(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent.cdec_pairwise_ranker = None
            raw_intrinsic = np.array([
                [420.0, 0.0, 256.0],
                [0.0, 420.0, 256.0],
                [0.0, 0.0, 1.0],
            ])
            mapped_intrinsic = np.diag([123.0, 124.0, 1.0])
            observed = []

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **kwargs):
                observed.append(kwargs.get("query_intrinsic"))
                return self.accepted_pose(reference_pose)

            with mock.patch(
                    "MemNavData.certified_relocalization_runtime.fundamental_support",
                    side_effect=self.support), mock.patch(
                    "MemNavData.lingbot_pnp_localization.map_raw_intrinsic_to_lingbot_pad",
                    return_value=mapped_intrinsic) as mapper, mock.patch(
                    "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                    side_effect=pnp), mock.patch(
                    "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                    side_effect=lambda value: value):
                result = agent.certified_relocalize(
                    goal,
                    [{"anchor": 8, "score": 0.9},
                     {"anchor": 9, "score": 0.8}],
                    goal_camera_intrinsic=raw_intrinsic,
                )

            self.assertTrue(result["accepted"])
            self.assertEqual(
                result["goal_camera_calibration"],
                "explicit_distinct_intrinsic")
            np.testing.assert_allclose(observed[0], mapped_intrinsic)
            mapper.assert_called_once()
            call = mapper.call_args
            np.testing.assert_allclose(call.args[0], raw_intrinsic)
            self.assertEqual(call.kwargs["raw_height"], 518)
            self.assertEqual(call.kwargs["raw_width"], 518)

    def test_invalid_goal_intrinsic_fails_closed_before_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            result = agent.certified_relocalize(
                goal,
                [{"anchor": 8, "score": 0.9}],
                goal_camera_intrinsic=[[0.0, 0.0], [0.0, 0.0]],
            )
            self.assertFalse(result["ok"])
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "invalid_goal_camera_intrinsic")

    def test_unthresholded_witness_keeps_geometry_but_removes_authority_gate(self):
        weak_support = {
            "lightglue_matches": 24,
            "lightglue_score_median": 0.7,
            "fundamental_inliers": 8,
            "fundamental_inlier_ratio": 0.5,
            "fundamental_query_grid_coverage": 0.25,
            "fundamental_query_hull_coverage": 0.01,
            "fundamental_reference_grid_coverage": 0.25,
            "fundamental_reference_hull_coverage": 0.01,
        }

        def weak_pose(_reference_points, _query_points, _depth, _confidence,
                      reference_pose, **_kwargs):
            payload = self.accepted_pose(reference_pose)
            payload.update({
                "status": "insufficient_inliers",
                "inliers": 8,
                "query_inlier_coverage": 0.01,
                "reference_inlier_coverage": 0.01,
                "reprojection_rmse_px": 12.0,
            })
            return payload

        patches = lambda pnp: (
            mock.patch(
                "MemNavData.certified_relocalization_runtime.fundamental_support",
                return_value=weak_support),
            mock.patch(
                "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                side_effect=pnp),
            mock.patch(
                "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                side_effect=lambda value: value),
        )
        with tempfile.TemporaryDirectory() as temporary:
            strict_agent, goal = self.make_agent(Path(temporary))
            strict_agent.cdec_pairwise_ranker = None
            strict_patches = patches(weak_pose)
            with strict_patches[0], strict_patches[1] as strict_pnp, \
                    strict_patches[2]:
                strict = strict_agent.certified_relocalize(
                    goal, [{"anchor": 8, "score": 0.9}])
            self.assertFalse(strict["accepted"])
            self.assertEqual(
                strict["reason"],
                "precheck_fundamental_inliers",
            )
            strict_pnp.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary:
            witness_agent, goal = self.make_agent(Path(temporary))
            witness_agent.cdec_pairwise_ranker = None
            witness_patches = patches(weak_pose)
            with witness_patches[0], witness_patches[1] as witness_pnp, \
                    witness_patches[2]:
                witness = witness_agent.certified_relocalize(
                    goal,
                    [{"anchor": 8, "score": 0.9}],
                    authority_policy="pnp_pose_available",
                )
            witness_pnp.assert_called_once()
            self.assertTrue(witness["accepted"])
            self.assertFalse(witness["certificate"]["accepted"])
            self.assertEqual(witness["authority_policy"], "pnp_pose_available")
            self.assertEqual(witness["reason"], "pnp_pose_available")
            self.assertFalse(
                witness["authority"]["certificate_thresholds_enforced"])

    def test_unregistered_authority_policy_fails_before_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            result = agent.certified_relocalize(
                goal,
                [{"anchor": 8, "score": 0.9}],
                authority_policy="accept_everything",
            )
            self.assertFalse(result["ok"])
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "invalid_authority_policy")

    def test_explicit_off_never_invokes_loaded_learned_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent._cdec_pairwise_proposal = mock.Mock(
                side_effect=AssertionError("disabled proposal must not run"))

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **_kwargs):
                return self.accepted_pose(reference_pose)

            with mock.patch(
                    "MemNavData.certified_relocalization_runtime.fundamental_support",
                    side_effect=self.support), mock.patch(
                    "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                    side_effect=pnp), mock.patch(
                    "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                    side_effect=lambda value: value):
                result = agent.certified_relocalize(
                    goal,
                    [{"anchor": 8, "score": 0.9},
                     {"anchor": 9, "score": 0.8}],
                    allow_learned_rescue=False,
                )
            self.assertTrue(result["accepted"])
            self.assertFalse(result["learned_rescue_requested"])
            self.assertEqual(result["learned_proposal"]["status"], "not_requested")
            agent._cdec_pairwise_proposal.assert_not_called()

    def test_learned_mode_cannot_change_inside_one_cached_goal(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent._cdec_pairwise_proposal = mock.Mock(return_value={
                "selected_anchor": 9,
                "activation_authorized": False,
            })

            def rejected(*_args, **_kwargs):
                return {"status": "ransac_failed"}

            patches = (
                mock.patch(
                    "MemNavData.certified_relocalization_runtime.fundamental_support",
                    side_effect=self.support),
                mock.patch(
                    "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                    side_effect=rejected),
                mock.patch(
                    "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                    side_effect=lambda value: value),
            )
            with patches[0], patches[1], patches[2]:
                first = agent.certified_relocalize(
                    goal,
                    [{"anchor": 8, "score": 0.9},
                     {"anchor": 9, "score": 0.8}],
                    allow_learned_rescue=False,
                )
                second = agent.certified_relocalize(
                    goal,
                    [{"anchor": 8, "score": 0.9},
                     {"anchor": 9, "score": 0.8}],
                    allow_learned_rescue=True,
                )
            self.assertTrue(first["ok"])
            self.assertEqual(first["learned_proposal"]["status"], "not_requested")
            self.assertFalse(second["ok"])
            self.assertEqual(second["reason"], "candidate_contract_changed")
            agent._cdec_pairwise_proposal.assert_not_called()

    def test_accepted_geometry_structurally_skips_learned_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent._cdec_pairwise_proposal = mock.Mock(
                side_effect=AssertionError("learned proposal must not run"))

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **_kwargs):
                return self.accepted_pose(reference_pose)

            result = self.invoke(agent, goal, pnp)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["selected_anchor"], 8)
            self.assertEqual(result["selected_proposal_source"], "geometry")
            self.assertEqual(
                result["learned_proposal"]["status"],
                "not_evaluated_geometry_accepted")
            agent._cdec_pairwise_proposal.assert_not_called()

    def test_dino_top1_counterfactual_has_no_action_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent.certified_counterfactual_audit = True
            agent.cdec_pairwise_ranker = None
            calls = []

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **_kwargs):
                anchor = int(round(float(reference_pose[0])))
                calls.append(anchor)
                if anchor == 8:
                    return {"status": "ransac_failed"}
                return self.accepted_pose(reference_pose)

            with mock.patch(
                    "MemNavData.certified_relocalization_runtime.fundamental_support",
                    side_effect=self.support), mock.patch(
                    "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                    side_effect=pnp), mock.patch(
                    "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                    side_effect=lambda value: value):
                result = agent.certified_relocalize(
                    goal,
                    [{"anchor": 9, "score": 0.9},
                     {"anchor": 8, "score": 0.8}],
                    allow_learned_rescue=False,
                )

            self.assertEqual(calls, [8, 9])
            self.assertFalse(result["accepted"])
            self.assertEqual(result["selected_anchor"], 8)
            audit = result["counterfactual_dino_top1_audit"]
            self.assertTrue(audit["accepted"])
            self.assertEqual(audit["selected_anchor"], 9)
            self.assertFalse(audit["action_authority"])
            ordered = result["counterfactual_dino_order_audit"]
            self.assertTrue(ordered["accepted"])
            self.assertEqual(ordered["selected_anchor"], 9)
            self.assertEqual(ordered["selected_dino_rank"], 1)
            self.assertEqual(ordered["attempt_count"], 1)
            self.assertFalse(ordered["action_authority"])

    def test_semantic_first_action_uses_first_certified_dino_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent.cdec_pairwise_ranker = None
            calls = []

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **_kwargs):
                anchor = int(round(float(reference_pose[0])))
                calls.append(anchor)
                return self.accepted_pose(reference_pose)

            with mock.patch(
                    "MemNavData.certified_relocalization_runtime.fundamental_support",
                    side_effect=self.support), mock.patch(
                    "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                    side_effect=pnp), mock.patch(
                    "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                    side_effect=lambda value: value):
                result = agent.certified_relocalize(
                    goal,
                    [{"anchor": 9, "score": 0.9},
                     {"anchor": 8, "score": 0.8}],
                    proposal_order="dino_first_certified",
                )

            self.assertEqual(calls, [9])
            self.assertTrue(result["accepted"])
            self.assertEqual(result["selected_anchor"], 9)
            self.assertEqual(
                result["selected_proposal_source"],
                "dino_first_certified",
            )
            self.assertEqual(result["proposal_order"], "dino_first_certified")
            self.assertEqual(len(result["proposal_attempts"]), 1)

    def test_semantic_first_tries_next_dino_anchor_after_veto(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent.cdec_pairwise_ranker = None
            calls = []

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **_kwargs):
                anchor = int(round(float(reference_pose[0])))
                calls.append(anchor)
                if anchor == 9:
                    return {"status": "ransac_failed"}
                return self.accepted_pose(reference_pose)

            with mock.patch(
                    "MemNavData.certified_relocalization_runtime.fundamental_support",
                    side_effect=self.support), mock.patch(
                    "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                    side_effect=pnp), mock.patch(
                    "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                    side_effect=lambda value: value):
                result = agent.certified_relocalize(
                    goal,
                    [{"anchor": 9, "score": 0.9},
                     {"anchor": 8, "score": 0.8}],
                    proposal_order="dino_first_certified",
                )

            self.assertEqual(calls, [9, 8])
            self.assertTrue(result["accepted"])
            self.assertEqual(result["selected_anchor"], 8)
            self.assertEqual(len(result["proposal_attempts"]), 2)

    def test_proposal_order_cannot_change_inside_one_cached_goal(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent.cdec_pairwise_ranker = None

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **_kwargs):
                return self.accepted_pose(reference_pose)

            with mock.patch(
                    "MemNavData.certified_relocalization_runtime.fundamental_support",
                    side_effect=self.support), mock.patch(
                    "MemNavData.lingbot_pnp_localization.correspondence_pnp_localize",
                    side_effect=pnp), mock.patch(
                    "MemNavData.lingbot_pnp_localization.jsonable_pnp",
                    side_effect=lambda value: value):
                first = agent.certified_relocalize(
                    goal,
                    [{"anchor": 9, "score": 0.9},
                     {"anchor": 8, "score": 0.8}],
                    proposal_order="geometry_first",
                )
                second = agent.certified_relocalize(
                    goal,
                    [{"anchor": 9, "score": 0.9},
                     {"anchor": 8, "score": 0.8}],
                    proposal_order="dino_first_certified",
                )

            self.assertTrue(first["accepted"])
            self.assertFalse(second["ok"])
            self.assertEqual(second["reason"], "candidate_contract_changed")

    def test_unknown_proposal_order_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            result = agent.certified_relocalize(
                goal,
                [{"anchor": 9, "score": 0.9}],
                proposal_order="unregistered",
            )
            self.assertFalse(result["ok"])
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "invalid_proposal_order")

    def test_learned_anchor_can_only_rescue_a_rejected_geometry_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent._cdec_pairwise_proposal = mock.Mock(return_value={
                "selected_anchor": 9,
                "activation_authorized": False,
            })
            calls = []

            def pnp(_reference_points, _query_points, _depth, _confidence,
                    reference_pose, **_kwargs):
                anchor = int(round(float(reference_pose[0])))
                calls.append(anchor)
                if anchor == 8:
                    return {"status": "ransac_failed"}
                return self.accepted_pose(reference_pose)

            result = self.invoke(agent, goal, pnp)
            self.assertEqual(calls, [8, 9])
            self.assertTrue(result["accepted"])
            self.assertEqual(result["selected_anchor"], 9)
            self.assertEqual(
                result["selected_proposal_source"],
                "learned_on_geometry_reject")
            self.assertEqual(
                [row["source"] for row in result["proposal_attempts"]],
                ["geometry", "learned_on_geometry_reject"])

    def test_same_learned_anchor_reuses_first_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent, goal = self.make_agent(Path(temporary))
            agent._cdec_pairwise_proposal = mock.Mock(return_value={
                "selected_anchor": 8,
                "activation_authorized": False,
            })
            calls = []

            def pnp(*_args, **_kwargs):
                calls.append(8)
                return {"status": "ransac_failed"}

            result = self.invoke(agent, goal, pnp)
            self.assertEqual(calls, [8])
            self.assertFalse(result["accepted"])
            self.assertEqual(
                result["learned_proposal"]["status"],
                "same_anchor_certificate_reused")


if __name__ == "__main__":
    unittest.main()
