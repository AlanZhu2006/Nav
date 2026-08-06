import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from MemNavData.habitat_rollout_primitives import (
    FrozenGeometryIdentity,
    HabitatPlanarPose,
    NAVMESH_BOOL_FIELDS,
    NAVMESH_FLOAT_FIELDS,
    habitat_pose_to_parquet_data,
)
from MemNavData.build_novel_candidate_manifest import (
    ROUTED_SCHEMA_VERSION,
    SCHEMA_VERSION,
    canonical_json_bytes as manifest_json_bytes,
)
from MemNavData.novel_rollout_protocol_v2 import (
    CandidateArm,
    FrozenDecisionState,
    PlanRequest,
    RuntimeGeometrySpec,
    canonical_pose_sha256,
    canonical_sha256,
    collect_paired_rollouts,
)
from MemNavData.real_h24_rollout_backend import (
    ATOMIC_PLAN_PROTOCOL,
    MEMORY_AUDIT_PROTOCOL,
    EncodedObservation,
    FrozenStateAssets,
    PinnedHabitatRuntime,
    RealH24BackendError,
    RealH24RolloutBackend,
    load_state_assets_from_manifest,
    sha256_bytes,
)


def settings_record():
    values = {name: 1.0 for name in NAVMESH_FLOAT_FIELDS}
    values.update({name: False for name in NAVMESH_BOOL_FIELDS})
    values["agent_radius"] = 0.30
    values["agent_height"] = 1.50
    values["agent_max_climb"] = 0.20
    values["agent_max_slope"] = 45.0
    return values


class FakeRuntime:
    def __init__(self, geometry_identity):
        self.geometry_identity = geometry_identity
        self._pose = HabitatPlanarPose(99.0, 0.0, 99.0, 1.0)
        self.reset_poses = []
        self.render_count = 0
        self.on_render = None

    @property
    def pose(self):
        return self._pose

    def reset_to(self, pose):
        self._pose = pose
        self.reset_poses.append(pose)

    def set_pose(self, pose):
        self._pose = pose

    def render_encoded(self):
        self.render_count += 1
        if self.on_render is not None:
            self.on_render(self.render_count)
        pose = self._pose
        token = (
            f"{self.render_count}:{pose.x_m:.12f}:{pose.z_m:.12f}:"
            f"{pose.yaw_rad:.12f}"
        ).encode()
        return EncodedObservation(b"jpeg:" + token, b"png:" + token)

    def snap_point(self, world_xyz):
        return np.asarray(world_xyz, dtype=np.float64)

    def is_navigable(self, world_xyz):
        return bool(np.isfinite(np.asarray(world_xyz)).all())

    def geodesic_distance(self, goal_world_xyz):
        goal = np.asarray(goal_world_xyz, dtype=float)
        return True, float(np.linalg.norm(self._pose.position[[0, 2]] - goal[[0, 2]]))


class FakeTransport:
    provenance = {
        "navdp_server_sha256": "1" * 64,
        "policy_agent_sha256": "2" * 64,
        "deterministic_seed_sha256": "3" * 64,
        "checkpoint_sha256": "4" * 64,
        "wrapper_sha256": "5" * 64,
    }

    def __init__(self, *, critic_value=1.0, image_point_x_offset=0.0):
        self.queue = []
        self.reset_count = 0
        self.replay_raw = []
        self.atomic_raw = []
        self.atomic_count = 0
        self.critic_value = float(critic_value)
        self.image_point_x_offset = float(image_point_x_offset)
        self.stop_threshold = None

    @staticmethod
    def processed(payload):
        return hashlib.sha256(b"processed:" + payload).hexdigest()

    def _audit(self):
        padded = None if not self.queue else canonical_sha256({"padded": self.queue})
        identity = {
            "memory_size": 8,
            "queue_lengths": [len(self.queue)],
            "queue_item_sha256": [list(self.queue)],
            "padded_model_tensor_sha256": padded,
        }
        return {
            "algo": "navdp",
            "protocol": MEMORY_AUDIT_PROTOCOL,
            "provenance": dict(self.provenance),
            **identity,
            "fifo_sha256": canonical_sha256(identity),
        }

    def post_json(self, endpoint, payload):
        if endpoint != "/navigator_reset":
            raise AssertionError(endpoint)
        self.reset_count += 1
        self.queue.clear()
        self.stop_threshold = float(payload["stop_threshold"])
        return {"algo": "navdp"}

    def request_json(self, endpoint):
        if endpoint != "/memory_audit":
            raise AssertionError(endpoint)
        return self._audit()

    def post_multipart(self, endpoint, *, files, data=None):
        if endpoint == "/memory_replay_step":
            raw = files["image"][1]
            self.replay_raw.append(raw)
            self.queue.append(self.processed(raw))
            self.queue[:] = self.queue[-8:]
            return {
                "algo": "navdp",
                "queue_lengths": [len(self.queue)],
                "memory_size": 8,
                "diffusion_sampled": False,
            }
        if endpoint != "/navdp_plan_atomic":
            raise AssertionError(endpoint)
        assert data is not None
        mode = data["mode"]
        before = self._audit()
        current = files["image"][1]
        goal = files["image_goal"][1]
        depth = files["depth"][1]
        self.atomic_raw.append((current, depth, goal, dict(data)))
        current_processed = self.processed(current)
        goal_processed = self.processed(goal)
        self.queue.append(current_processed)
        self.queue[:] = self.queue[-8:]
        after = self._audit()
        seed = int(data["diffusion_seed"])
        point_hash = None
        if mode == "image_point":
            point_hash = canonical_sha256(json.loads(data["goal_data"]))
        # A smooth forward path in the planning frame.  The signed receipt
        # binds both the raw critic selection and the executable postprocess.
        x_offset = self.image_point_x_offset if mode == "image_point" else 0.0
        raw_trajectory = [
            [0.10 * (index + 1) + x_offset, 0.25, 0.01 * (index + 1)]
            for index in range(24)
        ]
        fallback = self.critic_value < self.stop_threshold
        executable = [list(row) for row in raw_trajectory]
        if fallback:
            lateral_sign = float(np.sign(np.mean(
                [row[1] for row in raw_trajectory])))
            executable = [
                [0.0, lateral_sign, row[2]] for row in raw_trajectory]
        core = {
            "protocol": ATOMIC_PLAN_PROTOCOL,
            "mode": mode,
            "diffusion_seed": seed,
            "diffusion_call_count": 1,
            "goal_sha256": canonical_sha256({"batch": goal_processed}),
            "goal_item_sha256": [goal_processed],
            "current_sha256": canonical_sha256({"batch": current_processed}),
            "current_item_sha256": [current_processed],
            "fifo_before_sha256": before["fifo_sha256"],
            "fifo_after_append_sha256": after["fifo_sha256"],
            "fifo_item_sha256_before": before["queue_item_sha256"],
            "fifo_item_sha256": after["queue_item_sha256"],
            "fifo_lengths_before": before["queue_lengths"],
            "fifo_lengths_after": after["queue_lengths"],
            "point_goal_sha256": point_hash,
            "critic_max": self.critic_value,
            "stop_threshold": self.stop_threshold,
            "low_critic_fallback_applied": fallback,
            "raw_selected_trajectory": [raw_trajectory],
            "executable_trajectory": [executable],
            "inference_fifo_unchanged": True,
            "append_count_per_environment": 1,
        }
        receipt = {**core, "receipt_sha256": canonical_sha256(core)}
        self.atomic_count += 1
        return {
            "trajectory": [executable],
            "raw_selected_trajectory": [raw_trajectory],
            "all_trajectory": [[raw_trajectory]],
            "all_values": [[self.critic_value]],
            "critic_max": self.critic_value,
            "stop_threshold": self.stop_threshold,
            "low_critic_fallback_applied": fallback,
            "receipt": receipt,
            "provenance": dict(self.provenance),
        }

    def inject_fifo_mutation(self):
        self.queue.append("f" * 64)
        self.queue[:] = self.queue[-8:]


class FakePathfinder:
    def __init__(self):
        self.load_count = 0

    def load_nav_mesh(self, path):
        self.load_count += 1
        return True


class NoRecomputeSimulator:
    def __init__(self):
        self.pathfinder = FakePathfinder()
        self.recompute_count = 0

    def recompute_navmesh(self, *args):
        self.recompute_count += 1
        raise AssertionError("collector must not recompute navmesh")


class RealH24RolloutBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.glb = root / "scene.glb"
        self.navmesh = root / "scene.navmesh"
        self.glb.write_bytes(b"glb")
        self.navmesh.write_bytes(b"navmesh")
        self.settings = settings_record()
        self.identity = FrozenGeometryIdentity.capture(
            glb_path=self.glb,
            navmesh_path=self.navmesh,
            habitat_sim_version="0.3.1",
            agent_radius_m=0.30,
            agent_height_m=1.50,
            navmesh_settings=self.settings,
        )
        start = HabitatPlanarPose(0.0, 0.0, 0.0, 0.0)
        current = EncodedObservation(b"current-jpeg", b"current-depth")
        goal = b"goal-jpeg"
        runtime_geometry = RuntimeGeometrySpec(
            habitat_sim_version="0.3.1",
            agent_radius_m=0.30,
            agent_height_m=1.50,
            agent_max_climb_m=0.20,
            agent_max_slope_deg=45.0,
            navmesh_source="loaded_frozen",
            navmesh_settings_sha256=self.identity.navmesh_settings_sha256,
        )
        state = FrozenDecisionState(
            state_id="train/scene/episode/state/factual",
            session_id="scene/episode",
            goal_epoch="B:test",
            goal_sha256=sha256_bytes(goal),
            manifest_fifo_sha256="a" * 64,
            current_rgb_sha256=current.rgb_sha256,
            current_depth_sha256=current.depth_sha256,
            start_pose_sha256=canonical_pose_sha256((0.0, 0.0, 0.0)),
            environment_id="scene",
            environment_sha256=self.identity.glb_sha256,
            navmesh_sha256=self.identity.navmesh_sha256,
            runtime_geometry=runtime_geometry,
        )
        self.assets = FrozenStateAssets(
            state=state,
            sample_id=state.state_id,
            manifest_sha256="b" * 64,
            camera_intrinsic=(
                (355.0, 0.0, 240.0),
                (0.0, 351.0, 135.0),
                (0.0, 0.0, 1.0),
            ),
            camera_height_m=0.5,
            replay_frame_indices=(3, 11),
            replay_rgb_jpegs=(b"replay-3", b"replay-11"),
            frozen_current=current,
            goal_jpeg=goal,
            start_pose=start,
            label_goal_world_xyz_m=(0.0, 0.0, -4.0),
            geometry_identity=self.identity,
            glb_path=self.glb,
            navmesh_path=self.navmesh,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def backend(self, transport=None, runtime=None):
        return RealH24RolloutBackend(
            self.assets,
            FakeTransport() if transport is None else transport,
            FakeRuntime(self.identity) if runtime is None else runtime,
            expected_server_provenance=FakeTransport.provenance,
            stop_threshold=-0.5,
        )

    def test_reset_replays_only_preceding_frames_and_current_once(self):
        transport = FakeTransport()
        backend = self.backend(transport=transport)
        preparation = backend.prepare_arm(self.assets.state)
        self.assertEqual(transport.replay_raw, [b"replay-3", b"replay-11"])
        self.assertNotIn(b"current-jpeg", transport.replay_raw)
        request = PlanRequest(
            state_id=self.assets.state.state_id,
            candidate_id="native",
            candidate_type="native",
            goal_sha256=self.assets.state.goal_sha256,
            commitment_index=0,
            diffusion_seed=101,
            current_rgb_sha256=self.assets.state.current_rgb_sha256,
            current_depth_sha256=self.assets.state.current_depth_sha256,
            current_pose_sha256=self.assets.state.start_pose_sha256,
            current_world_pose_xz_yaw=(0.0, 0.0, 0.0),
            fixed_world_subgoal_xz_m=None,
        )
        plan = backend.plan(request)
        self.assertEqual(preparation.queue_length, 2)
        self.assertEqual(plan.queue_length_before, 2)
        self.assertEqual(plan.queue_length_after, 3)
        self.assertEqual(
            [row[0] for row in transport.atomic_raw], [b"current-jpeg"])
        diagnostics = backend.plan_diagnostics(plan.plan_sha256)
        self.assertEqual(diagnostics.server_selected_trajectory_index, 0)
        self.assertEqual(len(diagnostics.raw_selected_trajectory), 24)
        self.assertEqual(len(diagnostics.executable_trajectory), 24)
        self.assertEqual(diagnostics.all_values, [[1.0]])
        self.assertEqual(diagnostics.critic_max, 1.0)
        self.assertEqual(diagnostics.stop_threshold, -0.5)
        self.assertFalse(diagnostics.low_critic_fallback_applied)

    def test_low_critic_diagnostics_preserve_raw_and_executable_theta(self):
        transport = FakeTransport(critic_value=-0.5001)
        backend = self.backend(transport=transport)
        backend.prepare_arm(self.assets.state)
        request = PlanRequest(
            state_id=self.assets.state.state_id,
            candidate_id="native",
            candidate_type="native",
            goal_sha256=self.assets.state.goal_sha256,
            commitment_index=0,
            diffusion_seed=101,
            current_rgb_sha256=self.assets.state.current_rgb_sha256,
            current_depth_sha256=self.assets.state.current_depth_sha256,
            current_pose_sha256=self.assets.state.start_pose_sha256,
            current_world_pose_xz_yaw=(0.0, 0.0, 0.0),
            fixed_world_subgoal_xz_m=None,
        )
        plan = backend.plan(request)
        diagnostics = backend.plan_diagnostics(plan.plan_sha256)
        raw = np.asarray(diagnostics.raw_selected_trajectory)
        executable = np.asarray(diagnostics.executable_trajectory)

        self.assertTrue(diagnostics.low_critic_fallback_applied)
        self.assertAlmostEqual(diagnostics.critic_max, -0.5001)
        self.assertEqual(diagnostics.server_selected_trajectory_index, 0)
        self.assertFalse(np.array_equal(raw[:, :2], executable[:, :2]))
        self.assertTrue(np.all(executable[:, 0] == 0.0))
        self.assertTrue(np.all(executable[:, 1] == 1.0))
        self.assertTrue(np.array_equal(raw[:, 2], executable[:, 2]))

    def test_paired_arms_reset_equal_and_world_projection_changes(self):
        transport = FakeTransport()
        runtime = FakeRuntime(self.identity)

        def factory(_candidate_id):
            return self.backend(transport=transport, runtime=runtime)

        artifact = collect_paired_rollouts(
            factory,
            self.assets.state,
            (
                CandidateArm("native", "native"),
                CandidateArm("frontier-0", "frontier", (1.0, -3.0)),
            ),
            (101, 102, 103),
            run_signature_sha256="c" * 64,
        )
        self.assertEqual(transport.reset_count, 2)
        self.assertEqual(
            transport.replay_raw,
            [b"replay-3", b"replay-11", b"replay-3", b"replay-11"],
        )
        self.assertEqual(len(runtime.reset_poses), 2)
        self.assertEqual(runtime.reset_poses[0], runtime.reset_poses[1])
        residual = next(row for row in artifact.outcomes
                        if row.candidate_id == "frontier-0")
        projections = [plan.local_subgoal_forward_left_m for plan in residual.plans]
        self.assertEqual(
            {plan.fixed_world_subgoal_xz_m for plan in residual.plans},
            {(1.0, -3.0)},
        )
        self.assertEqual(len(set(projections)), 3)
        self.assertEqual(len(transport.atomic_raw), 6)

    def test_goal_hash_mismatch_fails_before_http_plan(self):
        transport = FakeTransport()
        backend = self.backend(transport=transport)
        backend.prepare_arm(self.assets.state)
        request = PlanRequest(
            state_id=self.assets.state.state_id,
            candidate_id="native",
            candidate_type="native",
            goal_sha256="d" * 64,
            commitment_index=0,
            diffusion_seed=11,
            current_rgb_sha256=self.assets.state.current_rgb_sha256,
            current_depth_sha256=self.assets.state.current_depth_sha256,
            current_pose_sha256=self.assets.state.start_pose_sha256,
            current_world_pose_xz_yaw=(0.0, 0.0, 0.0),
            fixed_world_subgoal_xz_m=None,
        )
        with self.assertRaisesRegex(RealH24BackendError, "goal hash mismatch"):
            backend.plan(request)
        self.assertEqual(transport.atomic_count, 0)

    def test_fifo_mutation_is_detected_before_pursuit(self):
        transport = FakeTransport()
        backend = self.backend(transport=transport)
        backend.prepare_arm(self.assets.state)
        request = PlanRequest(
            state_id=self.assets.state.state_id,
            candidate_id="native",
            candidate_type="native",
            goal_sha256=self.assets.state.goal_sha256,
            commitment_index=0,
            diffusion_seed=11,
            current_rgb_sha256=self.assets.state.current_rgb_sha256,
            current_depth_sha256=self.assets.state.current_depth_sha256,
            current_pose_sha256=self.assets.state.start_pose_sha256,
            current_world_pose_xz_yaw=(0.0, 0.0, 0.0),
            fixed_world_subgoal_xz_m=None,
        )
        plan = backend.plan(request)
        transport.inject_fifo_mutation()
        with self.assertRaisesRegex(RealH24BackendError, "FIFO changed before pursuit"):
            backend.pursue(plan, 8)

    def test_pinned_runtime_loads_navmesh_and_never_recomputes(self):
        simulator = NoRecomputeSimulator()
        runtime = PinnedHabitatRuntime(
            simulator,
            identity=self.identity,
            glb_path=self.glb,
            navmesh_path=self.navmesh,
            navmesh_settings=self.settings,
            habitat_sim_version="0.3.1",
            agent_radius_m=0.30,
            agent_height_m=1.50,
            camera_height_m=0.5,
        )
        self.assertIs(runtime.pathfinder, simulator.pathfinder)
        self.assertEqual(simulator.pathfinder.load_count, 1)
        self.assertEqual(simulator.recompute_count, 0)

    def test_v1_and_v2_manifest_materialize_same_frozen_state(self):
        root = Path(self.temporary.name)
        episode_root = root / "episodes"
        environment_root = root / "environments"
        navmesh_root = root / "navmeshes"
        episode = episode_root / "scene/episode_0000"
        rgb_root = episode / "videos/chunk-000/observation.images.rgb"
        depth_root = episode / "videos/chunk-000/observation.images.depth"
        data_root = episode / "data/chunk-000"
        meta_root = episode / "meta"
        for directory in (
            rgb_root, depth_root, data_root, meta_root,
            environment_root, navmesh_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        glb = environment_root / "scene.glb"
        navmesh = navmesh_root / "scene.navmesh"
        glb.write_bytes(b"fixture-glb")
        navmesh.write_bytes(b"fixture-navmesh")
        identity = FrozenGeometryIdentity.capture(
            glb_path=glb,
            navmesh_path=navmesh,
            habitat_sim_version="0.3.1",
            agent_radius_m=0.30,
            agent_height_m=1.50,
            navmesh_settings=self.settings,
        )
        identity_path = root / "geometry.json"
        identity.write_json(identity_path)
        metadata = {
            "camera_height_m": 0.5,
            "frame_convention": (
                "positions+parquet in data(Zup,M_W); yaw_habitat in render frame"),
            "goals": [{"kind": "novel", "pos": [1.0, 3.0, 0.0]}],
        }
        metadata_path = meta_root / "gen_meta.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        parquet_path = data_root / "episode_000000.parquet"
        parquet_path.write_bytes(b"parquet-pinned-by-record")
        goal_path = episode / "goal_1.jpg"
        goal_path.write_bytes(b"fixture-goal")
        for frame in range(2):
            (rgb_root / f"{frame}.jpg").write_bytes(f"rgb-{frame}".encode())
            (depth_root / f"{frame}.png").write_bytes(f"depth-{frame}".encode())

        action, extrinsic = habitat_pose_to_parquet_data(
            HabitatPlanarPose(0.0, 0.0, 0.0, 0.0), camera_height_m=0.5)
        rows = [{
            "index": frame,
            "observation.camera_intrinsic": [
                [355.0, 0.0, 240.0],
                [0.0, 351.0, 135.0],
                [0.0, 0.0, 1.0],
            ],
            "observation.camera_extrinsic": extrinsic.tolist(),
            "action": action.tolist(),
        } for frame in range(2)]

        def record(path, base):
            relative = path.relative_to(base).as_posix()
            raw = path.read_bytes()
            return {
                "path": relative,
                "path_sha256": sha256_bytes(relative.encode()),
                "bytes": len(raw),
                "content_sha256": sha256_bytes(raw),
            }

        def sequence_sha(value):
            return sha256_bytes(manifest_json_bytes(value))

        rgb_rows = [record(rgb_root / f"{frame}.jpg", episode_root)
                    for frame in range(2)]
        depth_rows = [record(depth_root / f"{frame}.png", episode_root)
                      for frame in range(2)]
        modalities = {}
        for name, values in (("rgb", rgb_rows), ("depth", depth_rows)):
            modalities[name] = {
                "path_sequence_sha256": sequence_sha(
                    [value["path"] for value in values]),
                "content_sequence_sha256": sequence_sha([{
                    "path": value["path"],
                    "bytes": value["bytes"],
                    "content_sha256": value["content_sha256"],
                } for value in values]),
            }
        parquet_sha = sequence_sha(rows)
        causal_body = {
            "frame_count": 2,
            "rgb": modalities["rgb"],
            "depth": modalities["depth"],
            "parquet_rows_sha256": parquet_sha,
        }
        fifo_content = [{
            "path": value["path"],
            "bytes": value["bytes"],
            "content_sha256": value["content_sha256"],
        } for value in rgb_rows]
        fifo_body = {
            "memory_size": 8,
            "exec_horizon": 8,
            "left_zero_pad_count": 6,
            "replay_frame_indices": [0],
            "current_frame_index": 1,
            "after_append_frame_indices": [0, 1],
            "path_sequence_sha256": sequence_sha(
                [value["path"] for value in rgb_rows]),
            "content_sequence_sha256": sequence_sha(fifo_content),
        }
        episode_record = {
            "episode": "episode_0000",
            "n_frames": 2,
            "metadata": record(metadata_path, episode_root),
            "parquet": record(parquet_path, episode_root),
            "goal_b": record(goal_path, episode_root),
        }
        sample = {
            "sample_id": "train/scene/episode_0000/goal_b_t0/factual",
            "scene": "scene",
            "source_episode": "episode_0000",
            "source_episode_id": "scene/episode_0000",
            "goal_episode": "episode_0000",
            "decision_frame": 2,
            "goal": record(goal_path, episode_root),
            "state_frame": rgb_rows[1],
            "causal_prefix": {
                "exclusive_end_frame": 2,
                "frame_count": 2,
                "modalities": modalities,
                "parquet_columns": [
                    "index", "observation.camera_intrinsic",
                    "observation.camera_extrinsic", "action",
                ],
                "parquet_row_count": 2,
                "parquet_rows_sha256": parquet_sha,
                "causal_prefix_sha256": sha256_bytes(
                    manifest_json_bytes(causal_body)),
            },
            "navdp_fifo": {
                **fifo_body,
                "fifo_sha256": sha256_bytes(manifest_json_bytes(fifo_body)),
            },
        }
        common = {
            "input_roots": {
                "episode_root": str(episode_root),
                "environment_root": str(environment_root),
                "navmesh_root": str(navmesh_root),
            },
            "scenes": [{
                "scene": "scene",
                "environment": record(glb, environment_root),
                "navmesh": record(navmesh, navmesh_root),
                "selected_episodes": [episode_record],
            }],
            "samples": [sample],
        }
        loaded = []
        for schema in (SCHEMA_VERSION, ROUTED_SCHEMA_VERSION):
            manifest = {"schema_version": schema, **common}
            manifest_path = root / f"manifest-{schema}.json"
            raw = manifest_json_bytes(manifest)
            manifest_path.write_bytes(raw)
            with mock.patch(
                "MemNavData.real_h24_rollout_backend.load_parquet_rows",
                return_value=rows,
            ):
                loaded.append(load_state_assets_from_manifest(
                    manifest_path,
                    sha256_bytes(raw),
                    sample["sample_id"],
                    identity_path,
                ))
        self.assertEqual(loaded[0].state, loaded[1].state)
        self.assertEqual(loaded[0].replay_frame_indices, (0,))
        self.assertEqual(loaded[0].replay_rgb_jpegs, (b"rgb-0",))
        self.assertEqual(loaded[0].frozen_current.rgb_jpeg, b"rgb-1")
        self.assertEqual(loaded[0].frozen_current.depth_png, b"depth-1")
        self.assertEqual(loaded[0].label_goal_world_xyz_m, (1.0, 0.0, -3.0))

        # Historical episodes omitted the generator's 0.5 m CLI value.  The
        # loader must not silently guess it: an explicit pin is mandatory.
        legacy_metadata = dict(metadata)
        legacy_metadata.pop("camera_height_m")
        metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
        episode_record["metadata"] = record(metadata_path, episode_root)
        legacy_manifest = {"schema_version": SCHEMA_VERSION, **common}
        legacy_path = root / "manifest-legacy-height.json"
        legacy_raw = manifest_json_bytes(legacy_manifest)
        legacy_path.write_bytes(legacy_raw)
        with mock.patch(
            "MemNavData.real_h24_rollout_backend.load_parquet_rows",
            return_value=rows,
        ):
            with self.assertRaisesRegex(
                    RealH24BackendError,
                    "explicit pinned legacy_camera_height_m"):
                load_state_assets_from_manifest(
                    legacy_path,
                    sha256_bytes(legacy_raw),
                    sample["sample_id"],
                    identity_path,
                )
            legacy_loaded = load_state_assets_from_manifest(
                legacy_path,
                sha256_bytes(legacy_raw),
                sample["sample_id"],
                identity_path,
                legacy_camera_height_m=0.5,
            )
        self.assertEqual(legacy_loaded.camera_height_m, 0.5)

        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        episode_record["metadata"] = record(metadata_path, episode_root)
        conflict_manifest = {"schema_version": SCHEMA_VERSION, **common}
        conflict_path = root / "manifest-height-conflict.json"
        conflict_raw = manifest_json_bytes(conflict_manifest)
        conflict_path.write_bytes(conflict_raw)
        with mock.patch(
            "MemNavData.real_h24_rollout_backend.load_parquet_rows",
            return_value=rows,
        ):
            with self.assertRaisesRegex(RealH24BackendError, "conflicts"):
                load_state_assets_from_manifest(
                    conflict_path,
                    sha256_bytes(conflict_raw),
                    sample["sample_id"],
                    identity_path,
                    legacy_camera_height_m=0.6,
                )


if __name__ == "__main__":
    unittest.main()
