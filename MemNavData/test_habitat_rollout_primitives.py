"""Pure Python/NumPy tests for audited Habitat rollout geometry."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MemNavData.habitat_rollout_primitives import (
    DATA_ZUP_FRAME_CONVENTION_PREFIX,
    FixedWorldPointGoal,
    FrozenGeometryError,
    FrozenGeometryIdentity,
    HabitatPlanarPose,
    NAVMESH_SETTING_FIELDS,
    PoseConventionError,
    habitat_pose_to_parquet_data,
    load_pinned_navmesh_for_collector,
    local_forward_left_to_world,
    navmesh_settings_signature,
    parquet_data_pose_to_habitat,
    relative_yaw,
    world_to_local_forward_left,
    wrap_yaw,
)


FRAME_CONVENTION = (
    DATA_ZUP_FRAME_CONVENTION_PREFIX
    + "; yaw_habitat in render frame"
)


def navmesh_settings() -> dict:
    values = {
        "agent_height": 1.5,
        "agent_max_climb": 0.2,
        "agent_max_slope": 45.0,
        "agent_radius": 0.3,
        "cell_height": 0.2,
        "cell_size": 0.05,
        "detail_sample_dist": 6.0,
        "detail_sample_max_error": 1.0,
        "edge_max_error": 1.3,
        "edge_max_len": 12.0,
        "filter_ledge_spans": True,
        "filter_low_hanging_obstacles": True,
        "filter_walkable_low_height_spans": True,
        "include_static_objects": False,
        "region_merge_size": 20.0,
        "region_min_size": 20.0,
        "verts_per_poly": 6.0,
    }
    assert set(values) == set(NAVMESH_SETTING_FIELDS)
    return values


class PosePrimitiveTests(unittest.TestCase):
    def test_wrap_yaw_and_boundary_relative_yaw(self):
        self.assertEqual(wrap_yaw(math.pi), -math.pi)
        self.assertEqual(wrap_yaw(-math.pi), -math.pi)
        self.assertAlmostEqual(
            math.degrees(relative_yaw(math.radians(-179), math.radians(179))),
            2.0,
        )
        values = wrap_yaw(np.asarray([-3 * math.pi, 0.0, 3 * math.pi]))
        np.testing.assert_allclose(values, [-math.pi, 0.0, -math.pi])

    def test_local_world_roundtrip_for_forward_left_axes(self):
        cases = (
            (HabitatPlanarPose(1.0, 0.2, -2.0, 0.0), [2.0, 0.0],
             [1.0, 0.2, -4.0]),
            (HabitatPlanarPose(1.0, 0.2, -2.0, 0.0), [0.0, 2.0],
             [-1.0, 0.2, -2.0]),
            (HabitatPlanarPose(1.0, 0.2, -2.0, math.pi / 2), [2.0, 0.0],
             [-1.0, 0.2, -2.0]),
            (HabitatPlanarPose(1.0, 0.2, -2.0, math.pi / 2), [0.0, 2.0],
             [1.0, 0.2, 0.0]),
        )
        for pose, local, expected_world in cases:
            with self.subTest(pose=pose, local=local):
                world = local_forward_left_to_world(local, pose)
                np.testing.assert_allclose(world, expected_world, atol=1e-12)
                np.testing.assert_allclose(
                    world_to_local_forward_left(world, pose), local, atol=1e-12)

    def test_fixed_world_goal_is_reprojected_at_every_plan(self):
        first_pose = HabitatPlanarPose(0.0, 0.0, 0.0, 0.0)
        local_input = np.asarray([4.0, 1.0])
        goal = FixedWorldPointGoal.from_local(local_input, first_pose)
        np.testing.assert_allclose(goal.world_point, [-1.0, 0.0, -4.0])

        # A future plan occurs after both translation and rotation.  Its local
        # coordinates change, but mapping them back must recover the same fixed
        # world point.
        second_pose = HabitatPlanarPose(-1.0, 0.0, -1.0, math.pi / 2)
        second_local = goal.reproject_for_plan(second_pose)
        self.assertFalse(np.allclose(second_local, local_input))
        np.testing.assert_allclose(
            local_forward_left_to_world(second_local, second_pose),
            goal.world_point,
            atol=1e-12,
        )

        # Mutating the caller-owned input after fixation cannot move the goal.
        local_input[:] = 999.0
        np.testing.assert_allclose(goal.world_point, [-1.0, 0.0, -4.0])

    def test_parquet_pose_roundtrip_and_legacy_axis_equivalence(self):
        for yaw in (0.0, 0.4, -1.1, math.pi - 1e-7):
            pose = HabitatPlanarPose(2.5, -0.1, 8.0, yaw)
            action, fixed_mount = habitat_pose_to_parquet_data(
                pose, camera_height_m=0.5)
            fixed = parquet_data_pose_to_habitat(
                action,
                fixed_mount,
                camera_height_m=0.5,
                frame_convention=FRAME_CONVENTION,
            )
            legacy = parquet_data_pose_to_habitat(
                action,
                np.eye(4),
                camera_height_m=0.5,
                frame_convention=FRAME_CONVENTION,
            )
            with self.subTest(yaw=yaw):
                np.testing.assert_allclose(fixed.position, pose.position, atol=1e-9)
                np.testing.assert_allclose(legacy.position, pose.position, atol=1e-9)
                self.assertAlmostEqual(relative_yaw(fixed.yaw_rad, yaw), 0.0)
                self.assertAlmostEqual(relative_yaw(legacy.yaw_rad, yaw), 0.0)

    def test_parquet_pose_rejects_unpinned_mount_or_convention(self):
        pose = HabitatPlanarPose(0.0, 0.0, 0.0, 0.0)
        action, mount = habitat_pose_to_parquet_data(
            pose, camera_height_m=0.5)
        wrong_mount = mount.copy()
        wrong_mount[:3, 3] = 0.0
        with self.assertRaises(PoseConventionError):
            parquet_data_pose_to_habitat(
                action,
                wrong_mount,
                camera_height_m=0.5,
                frame_convention=FRAME_CONVENTION,
            )
        with self.assertRaises(PoseConventionError):
            parquet_data_pose_to_habitat(
                action,
                mount,
                camera_height_m=0.5,
                frame_convention="unknown",
            )
        with self.assertRaises(PoseConventionError):
            parquet_data_pose_to_habitat(
                action,
                np.eye(4),
                camera_height_m=0.5,
                frame_convention=FRAME_CONVENTION,
                allow_legacy_identity=False,
            )


class FrozenGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.glb = root / "scene.glb"
        self.navmesh = root / "scene.navmesh"
        self.glb.write_bytes(b"pinned glb bytes\x00")
        self.navmesh.write_bytes(b"pinned navmesh bytes\x00")
        self.settings = navmesh_settings()
        self.identity = FrozenGeometryIdentity.capture(
            glb_path=self.glb,
            navmesh_path=self.navmesh,
            habitat_sim_version="0.3.3",
            agent_radius_m=0.3,
            agent_height_m=1.5,
            navmesh_settings=self.settings,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def validate(self, **overrides):
        arguments = {
            "glb_path": self.glb,
            "navmesh_path": self.navmesh,
            "habitat_sim_version": "0.3.3",
            "agent_radius_m": 0.3,
            "agent_height_m": 1.5,
            "navmesh_settings": self.settings,
        }
        arguments.update(overrides)
        return self.identity.validate_runtime(**arguments)

    def test_record_roundtrip_is_canonical_and_defensive(self):
        restored = FrozenGeometryIdentity.from_dict(self.identity.to_dict())
        self.assertEqual(restored, self.identity)
        self.assertEqual(restored.identity_sha256, self.identity.identity_sha256)
        self.assertTrue(self.identity.canonical_json_bytes().endswith(b"\n"))
        identity_path = self.glb.parent / "geometry_identity.json"
        self.assertEqual(
            self.identity.write_json(identity_path), self.identity.identity_sha256)
        self.assertEqual(
            FrozenGeometryIdentity.load_json(identity_path), self.identity)
        # Re-recording the exact identity is idempotent, not an overwrite.
        self.identity.write_json(identity_path)
        copied = restored.navmesh_settings
        copied["agent_radius"] = 9.0
        self.assertEqual(restored.navmesh_settings["agent_radius"], 0.3)
        self.validate()

    def test_wrong_hash_version_agent_or_settings_is_rejected(self):
        record = self.identity.to_dict()
        record["navmesh"]["content_sha256"] = "0" * 64
        wrong_hash = FrozenGeometryIdentity.from_dict(record)
        with self.assertRaisesRegex(FrozenGeometryError, "navmesh content"):
            wrong_hash.validate_runtime(
                glb_path=self.glb,
                navmesh_path=self.navmesh,
                habitat_sim_version="0.3.3",
                agent_radius_m=0.3,
                agent_height_m=1.5,
                navmesh_settings=self.settings,
            )
        with self.assertRaisesRegex(FrozenGeometryError, "version"):
            self.validate(habitat_sim_version="0.3.2")
        with self.assertRaisesRegex(FrozenGeometryError, "radius"):
            self.validate(agent_radius_m=0.31)
        changed = copy.deepcopy(self.settings)
        changed["cell_size"] = 0.051
        with self.assertRaisesRegex(FrozenGeometryError, "Settings changed"):
            self.validate(navmesh_settings=changed)
        self.assertNotEqual(
            navmesh_settings_signature(changed),
            self.identity.navmesh_settings_sha256,
        )

    def test_future_file_mutation_is_detected(self):
        self.glb.write_bytes(self.glb.read_bytes() + b"future")
        with self.assertRaisesRegex(FrozenGeometryError, "GLB content"):
            self.validate()
        # Restore the GLB exactly, then mutate the navmesh independently.
        self.glb.write_bytes(b"pinned glb bytes\x00")
        self.navmesh.write_bytes(self.navmesh.read_bytes() + b"future")
        with self.assertRaisesRegex(FrozenGeometryError, "navmesh content"):
            self.validate()

    def test_settings_field_addition_or_removal_is_rejected(self):
        missing = copy.deepcopy(self.settings)
        missing.pop("edge_max_len")
        with self.assertRaisesRegex(FrozenGeometryError, "fields changed"):
            FrozenGeometryIdentity.capture(
                glb_path=self.glb,
                navmesh_path=self.navmesh,
                habitat_sim_version="0.3.3",
                agent_radius_m=0.3,
                agent_height_m=1.5,
                navmesh_settings=missing,
            )
        extra = copy.deepcopy(self.settings)
        extra["unknown_new_setting"] = 1.0
        with self.assertRaisesRegex(FrozenGeometryError, "fields changed"):
            FrozenGeometryIdentity.capture(
                glb_path=self.glb,
                navmesh_path=self.navmesh,
                habitat_sim_version="0.3.3",
                agent_radius_m=0.3,
                agent_height_m=1.5,
                navmesh_settings=extra,
            )

    def test_collector_only_loads_pinned_navmesh(self):
        class Pathfinder:
            def __init__(self):
                self.loaded = []

            def load_nav_mesh(self, path):
                self.loaded.append(path)
                return True

        class Simulator:
            def __init__(self):
                self.pathfinder = Pathfinder()
                self.recompute_calls = 0

            def recompute_navmesh(self, *_args, **_kwargs):
                self.recompute_calls += 1
                raise AssertionError("collector must not recompute navmesh")

        simulator = Simulator()
        result = load_pinned_navmesh_for_collector(
            simulator,
            identity=self.identity,
            glb_path=self.glb,
            navmesh_path=self.navmesh,
            habitat_sim_version="0.3.3",
            agent_radius_m=0.3,
            agent_height_m=1.5,
            navmesh_settings=self.settings,
        )
        self.assertIs(result, simulator.pathfinder)
        self.assertEqual(simulator.pathfinder.loaded, [str(self.navmesh)])
        self.assertEqual(simulator.recompute_calls, 0)

    def test_mutation_during_navmesh_load_is_rejected(self):
        navmesh = self.navmesh

        class Pathfinder:
            def load_nav_mesh(self, _path):
                navmesh.write_bytes(navmesh.read_bytes() + b"race")
                return True

        class Simulator:
            pathfinder = Pathfinder()

        with self.assertRaisesRegex(FrozenGeometryError, "navmesh content"):
            load_pinned_navmesh_for_collector(
                Simulator(),
                identity=self.identity,
                glb_path=self.glb,
                navmesh_path=self.navmesh,
                habitat_sim_version="0.3.3",
                agent_radius_m=0.3,
                agent_height_m=1.5,
                navmesh_settings=self.settings,
            )


if __name__ == "__main__":
    unittest.main()
