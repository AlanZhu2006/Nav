import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from MemNavData.novel_rollout_protocol_v2 import (
    CandidateArm,
    CommitmentReceipt,
    FrozenDecisionState,
    PlanReceipt,
    PreparationReceipt,
    RuntimeGeometrySpec,
    RolloutProtocolError,
    StepReceipt,
    artifact_from_dict,
    artifact_to_dict,
    atomic_write_artifact,
    canonical_pose_sha256,
    canonical_runtime_geometry_signature,
    collect_native_rollout,
    collect_paired_rollouts,
    load_artifact,
    run_candidate_arm,
    world_goal_to_local,
)


def digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def state():
    runtime_geometry = RuntimeGeometrySpec(
        habitat_sim_version="0.3.3",
        agent_radius_m=0.18,
        agent_height_m=0.88,
        agent_max_climb_m=0.2,
        agent_max_slope_deg=45.0,
        navmesh_source="loaded_frozen",
        navmesh_settings_sha256=digest("navmesh-settings"),
    )
    return FrozenDecisionState(
        state_id="scene/episode/state-b0",
        session_id="scene/episode/session-0",
        goal_epoch="goal-b-factual",
        goal_sha256=digest("goal"),
        manifest_fifo_sha256=digest("manifest-fifo"),
        current_rgb_sha256=digest("rgb-t0"),
        current_depth_sha256=digest("depth-t0"),
        start_pose_sha256=canonical_pose_sha256((0.0, 0.0, 0.0)),
        environment_id="mp3d/scene.glb",
        environment_sha256=digest("environment"),
        navmesh_sha256=digest("navmesh"),
        runtime_geometry=runtime_geometry,
    )


NATIVE = CandidateArm("native", "native")
RESIDUAL = CandidateArm("frontier-1", "frontier", (5.0, 2.0))
SEEDS = (11, 22, 33)
RUN_SHA = digest("run-signature")


class FakeBackend:
    """Deterministic stateful adapter with an audited eight-item FIFO."""

    def __init__(
        self,
        candidate_id,
        *,
        speed=None,
        contaminate_prepare=False,
        mismatch_seed=False,
        wrong_goal=False,
        wrong_environment=False,
        wrong_navmesh=False,
        wrong_geometry_signature=False,
        mutate_during_pursuit=False,
        bad_t0_append=False,
        bad_projection=False,
        unreachable_step=None,
        collision_step=None,
    ):
        self.candidate_id = candidate_id
        self.speed = speed if speed is not None else (
            0.10 if candidate_id == "native" else 0.20)
        self.contaminate_prepare = contaminate_prepare
        self.mismatch_seed = mismatch_seed
        self.wrong_goal = wrong_goal
        self.wrong_environment = wrong_environment
        self.wrong_navmesh = wrong_navmesh
        self.wrong_geometry_signature = wrong_geometry_signature
        self.mutate_during_pursuit = mutate_during_pursuit
        self.bad_t0_append = bad_t0_append
        self.bad_projection = bad_projection
        self.unreachable_step = unreachable_step
        self.collision_step = collision_step
        self.distance = 10.0
        self.position = 0.0
        self.fifo = digest("processed-fifo")
        self.queue_length = 7

    def prepare_arm(self, frozen):
        self.distance = 10.0
        self.position = 0.0
        self.fifo = digest("processed-fifo")
        self.queue_length = 7
        item_hashes = tuple(digest(f"fifo-item-{index}") for index in range(7))
        return PreparationReceipt(
            state_id=frozen.state_id,
            manifest_fifo_sha256=frozen.manifest_fifo_sha256,
            processed_fifo_sha256=self.fifo,
            processed_fifo_item_sha256=item_hashes,
            queue_length=self.queue_length,
            current_rgb_sha256=(
                digest("contaminated")
                if self.contaminate_prepare else frozen.current_rgb_sha256),
            current_depth_sha256=frozen.current_depth_sha256,
            start_pose_sha256=frozen.start_pose_sha256,
            environment_sha256=(
                digest("wrong-environment")
                if self.wrong_environment else frozen.environment_sha256
            ),
            navmesh_sha256=(
                digest("wrong-navmesh")
                if self.wrong_navmesh else frozen.navmesh_sha256
            ),
            runtime_geometry_signature=(
                digest("wrong-runtime-geometry")
                if self.wrong_geometry_signature
                else canonical_runtime_geometry_signature(
                    frozen.environment_sha256,
                    frozen.navmesh_sha256,
                    frozen.runtime_geometry,
                )
            ),
            world_pose_xz_yaw=(0.0, 0.0, 0.0),
            initial_goal_distance_m=self.distance,
            goal_reachable=True,
            diffusion_calls=0,
        )

    def plan(self, request):
        before = self.fifo
        before_length = self.queue_length
        # At t0 this depends only on the common current observation.  Later it
        # depends on the arm's factual trajectory and is allowed to diverge.
        append_token = (
            "t0" if request.commitment_index == 0
            else f"t{request.commitment_index * 8}-p{self.position:.8f}"
        )
        if self.bad_t0_append and request.commitment_index == 0:
            append_token += f"-{self.candidate_id}"
        self.fifo = digest(f"{before}|{append_token}")
        self.queue_length = min(8, before_length + 1)
        local = None
        if request.fixed_world_subgoal_xz_m is not None:
            local = world_goal_to_local(
                request.fixed_world_subgoal_xz_m,
                request.current_world_pose_xz_yaw,
            )
            if self.bad_projection:
                local = (local[0] + 1.0, local[1])
        return PlanReceipt(
            state_id=request.state_id,
            candidate_id=request.candidate_id,
            candidate_type=request.candidate_type,
            goal_sha256=(
                digest("wrong-goal") if self.wrong_goal
                else request.goal_sha256
            ),
            commitment_index=request.commitment_index,
            diffusion_seed=(
                request.diffusion_seed + 1
                if self.mismatch_seed else request.diffusion_seed),
            current_rgb_sha256=request.current_rgb_sha256,
            current_depth_sha256=request.current_depth_sha256,
            current_pose_sha256=request.current_pose_sha256,
            current_world_pose_xz_yaw=request.current_world_pose_xz_yaw,
            fixed_world_subgoal_xz_m=request.fixed_world_subgoal_xz_m,
            local_subgoal_forward_left_m=local,
            plan_sha256=digest(
                f"{request.candidate_id}/{request.commitment_index}/"
                f"{request.diffusion_seed}"),
            fifo_sha256_before=before,
            fifo_sha256_after=self.fifo,
            queue_length_before=before_length,
            queue_length_after=self.queue_length,
            diffusion_calls_delta=1,
        )

    def pursue(self, plan, steps):
        rows = []
        for offset in range(steps):
            global_step = plan.commitment_index * 8 + offset
            self.position += self.speed
            self.distance = max(0.0, self.distance - self.speed)
            reachable = global_step != self.unreachable_step
            collision = global_step == self.collision_step
            world_pose = (self.position, 0.0, 0.0)
            rows.append(StepReceipt(
                global_step_index=global_step,
                pose_sha256=canonical_pose_sha256(world_pose),
                world_pose_xz_yaw=world_pose,
                rgb_sha256=digest(f"rgb/{self.position:.8f}"),
                depth_sha256=digest(f"depth/{self.position:.8f}"),
                goal_distance_m=self.distance if reachable else None,
                goal_reachable=reachable,
                moved_m=self.speed,
                collision_detected=collision,
                full_step_rejected=collision,
                creep_used=False,
                zero_motion=False,
            ))
        return CommitmentReceipt(
            state_id=plan.state_id,
            candidate_id=plan.candidate_id,
            commitment_index=plan.commitment_index,
            plan_sha256=plan.plan_sha256,
            fifo_mutations=1 if self.mutate_during_pursuit else 0,
            steps=tuple(rows),
        )


def factory(**options):
    return lambda candidate_id: FakeBackend(candidate_id, **options)


class NovelRolloutProtocolV2Test(unittest.TestCase):
    def test_paired_rollout_derives_h24_advantage(self):
        artifact = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        self.assertEqual(
            [outcome.candidate_id for outcome in artifact.outcomes],
            ["native", "frontier-1"])
        native = artifact.labels_by_candidate["native"]
        residual = artifact.labels_by_candidate["frontier-1"]
        self.assertAlmostEqual(native["geodesic_progress_h24_m"], 2.4)
        self.assertAlmostEqual(residual["geodesic_progress_h24_m"], 4.8)
        self.assertAlmostEqual(residual["advantage_h24_m"], 2.4)
        self.assertTrue(residual["useful"])
        self.assertFalse(residual["harm"])
        native_outcome, residual_outcome = artifact.outcomes
        self.assertEqual(
            native_outcome.plans[0].fifo_sha256_after,
            residual_outcome.plans[0].fifo_sha256_after,
        )
        self.assertNotEqual(
            native_outcome.plans[1].fifo_sha256_after,
            residual_outcome.plans[1].fifo_sha256_after,
        )

    def test_world_subgoal_is_fixed_but_local_projection_changes(self):
        outcome = run_candidate_arm(
            FakeBackend("frontier-1"), state(), RESIDUAL, SEEDS)
        self.assertEqual(
            {plan.fixed_world_subgoal_xz_m for plan in outcome.plans},
            {(5.0, 2.0)})
        locals_ = [plan.local_subgoal_forward_left_m for plan in outcome.plans]
        self.assertEqual(locals_[0], (-2.0, -5.0))
        self.assertNotEqual(locals_[0], locals_[1])
        self.assertNotEqual(locals_[1], locals_[2])

    def test_arm_order_reversal_is_artifact_identical(self):
        first = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        reverse = collect_paired_rollouts(
            factory(), state(), (RESIDUAL, NATIVE), SEEDS,
            run_signature_sha256=RUN_SHA)
        self.assertEqual(artifact_to_dict(first), artifact_to_dict(reverse))

    def test_repeated_native_is_bitwise_identical(self):
        first = run_candidate_arm(
            FakeBackend("native"), state(), NATIVE, SEEDS)
        second = run_candidate_arm(
            FakeBackend("native"), state(), NATIVE, SEEDS)
        self.assertEqual(dataclasses.asdict(first), dataclasses.asdict(second))

    def test_all_arms_start_from_identical_preparation(self):
        calls = 0

        def contaminated(candidate_id):
            nonlocal calls
            calls += 1
            return FakeBackend(
                candidate_id, contaminate_prepare=(candidate_id != "native"))

        with self.assertRaisesRegex(
                RolloutProtocolError, "start RGB|byte-identical"):
            collect_paired_rollouts(
                contaminated, state(), (NATIVE, RESIDUAL), SEEDS,
                run_signature_sha256=RUN_SHA)
        self.assertEqual(calls, 2)

    def test_seed_echo_mismatch_fails(self):
        with self.assertRaisesRegex(RolloutProtocolError, "diffusion_seed"):
            run_candidate_arm(
                FakeBackend("native", mismatch_seed=True),
                state(), NATIVE, SEEDS)

    def test_goal_hash_must_be_echoed_by_every_plan(self):
        with self.assertRaisesRegex(RolloutProtocolError, "goal_sha256"):
            run_candidate_arm(
                FakeBackend("native", wrong_goal=True), state(), NATIVE, SEEDS)

    def test_preparation_must_echo_exact_navmesh(self):
        with self.assertRaisesRegex(RolloutProtocolError, "navmesh"):
            run_candidate_arm(
                FakeBackend("native", wrong_navmesh=True),
                state(), NATIVE, SEEDS)

    def test_preparation_binds_environment_and_runtime_geometry(self):
        with self.assertRaisesRegex(RolloutProtocolError, "environment"):
            run_candidate_arm(
                FakeBackend("native", wrong_environment=True),
                state(), NATIVE, SEEDS)
        with self.assertRaisesRegex(RolloutProtocolError, "runtime geometry"):
            run_candidate_arm(
                FakeBackend("native", wrong_geometry_signature=True),
                state(), NATIVE, SEEDS)

    def test_wrong_world_to_local_projection_fails(self):
        with self.assertRaisesRegex(RolloutProtocolError, "local projection"):
            run_candidate_arm(
                FakeBackend("frontier-1", bad_projection=True),
                state(), RESIDUAL, SEEDS)

    def test_pursuit_cannot_mutate_fifo(self):
        with self.assertRaisesRegex(RolloutProtocolError, "FIFO"):
            run_candidate_arm(
                FakeBackend("native", mutate_during_pursuit=True),
                state(), NATIVE, SEEDS)

    def test_t0_append_must_match_across_arms(self):
        with self.assertRaisesRegex(RolloutProtocolError, "t0 observation"):
            collect_paired_rollouts(
                factory(bad_t0_append=True),
                state(), (NATIVE, RESIDUAL), SEEDS,
                run_signature_sha256=RUN_SHA)

    def test_unreachable_arm_gets_neutral_invalid_labels(self):
        def backend(candidate_id):
            return FakeBackend(
                candidate_id,
                unreachable_step=(10 if candidate_id == "frontier-1" else None),
            )

        artifact = collect_paired_rollouts(
            backend, state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        labels = artifact.labels_by_candidate["frontier-1"]
        self.assertFalse(labels["rollout_label_valid"])
        self.assertFalse(labels["reachable"])
        self.assertEqual(labels["advantage_h24_m"], 0.0)
        self.assertFalse(labels["harm"])

    def test_native_invalid_neutralizes_complete_pair(self):
        def backend(candidate_id):
            return FakeBackend(
                candidate_id,
                unreachable_step=(3 if candidate_id == "native" else None),
            )

        artifact = collect_paired_rollouts(
            backend, state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        self.assertTrue(all(
            not labels["rollout_label_valid"]
            for labels in artifact.labels_by_candidate.values()))

    def test_collision_and_regression_are_harm(self):
        collision = collect_paired_rollouts(
            factory(collision_step=2), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        self.assertTrue(
            collision.labels_by_candidate["frontier-1"]["collision_h8"])
        self.assertTrue(collision.labels_by_candidate["frontier-1"]["harm"])
        self.assertFalse(
            collision.labels_by_candidate["frontier-1"]["useful"])

        def slower_residual(candidate_id):
            speed = 0.10 if candidate_id == "native" else 0.01
            return FakeBackend(candidate_id, speed=speed)

        regression = collect_paired_rollouts(
            slower_residual, state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        labels = regression.labels_by_candidate["frontier-1"]
        self.assertTrue(labels["regression_h24"])
        self.assertTrue(labels["harm"])

    def test_requires_exactly_three_unique_seeds(self):
        with self.assertRaisesRegex(RolloutProtocolError, "exactly 3"):
            run_candidate_arm(FakeBackend("native"), state(), NATIVE, (1, 2))
        with self.assertRaisesRegex(RolloutProtocolError, "unique"):
            run_candidate_arm(FakeBackend("native"), state(), NATIVE, (1, 1, 2))

    def test_atomic_write_and_strict_resume(self):
        artifact = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json"
            self.assertEqual(atomic_write_artifact(path, artifact), "written")
            self.assertEqual(
                atomic_write_artifact(path, artifact, resume=True), "resumed")
            payload = json.loads(path.read_text())
            self.assertEqual(payload["artifact_sha256"], artifact.artifact_sha256)
            sidecar = path.with_suffix(".json.sha256").read_text().split()
            self.assertEqual(sidecar[1], "row.json")
            self.assertEqual(sidecar[0], hashlib.sha256(path.read_bytes()).hexdigest())
            loaded = load_artifact(path)
            self.assertEqual(artifact_to_dict(loaded), artifact_to_dict(artifact))

    def test_artifact_from_dict_rejects_unknown_key(self):
        artifact = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        payload = json.loads(json.dumps(artifact_to_dict(artifact)))
        payload["unexpected"] = True
        with self.assertRaisesRegex(RolloutProtocolError, "unknown keys"):
            artifact_from_dict(payload)

    def test_disk_loader_rejects_noncanonical_json_even_with_valid_sidecar(self):
        artifact = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json"
            atomic_write_artifact(path, artifact)
            payload = json.loads(path.read_text(encoding="utf-8"))
            noncanonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8") + b"\n"
            path.write_bytes(noncanonical)
            path.with_suffix(".json.sha256").write_text(
                f"{hashlib.sha256(noncanonical).hexdigest()}  row.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    RolloutProtocolError, "canonical disk encoding"):
                load_artifact(path)

    def test_explicit_native_only_artifact_is_valid_but_not_paired(self):
        artifact = collect_native_rollout(
            factory(), state(), SEEDS, run_signature_sha256=RUN_SHA)
        self.assertEqual([row.candidate_id for row in artifact.outcomes], ["native"])
        payload = json.loads(json.dumps(artifact_to_dict(artifact)))
        self.assertEqual(
            artifact_to_dict(artifact_from_dict(payload)),
            artifact_to_dict(artifact),
        )
        with self.assertRaisesRegex(RolloutProtocolError, "paired rollout"):
            collect_paired_rollouts(
                factory(), state(), (NATIVE,), SEEDS,
                run_signature_sha256=RUN_SHA)

    def test_resume_rejects_different_run(self):
        artifact = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=RUN_SHA)
        changed = collect_paired_rollouts(
            factory(), state(), (NATIVE, RESIDUAL), SEEDS,
            run_signature_sha256=digest("different-run"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json"
            atomic_write_artifact(path, artifact)
            with self.assertRaisesRegex(RolloutProtocolError, "differs"):
                atomic_write_artifact(path, changed, resume=True)


if __name__ == "__main__":
    unittest.main()
