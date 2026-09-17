"""Delayed goals, gauge invariance, causality, and atomic coordinate updates."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from NavDP.baselines.memnav.gem.bindings import EpisodicBindings


def transform(translation=(0, 0, 0), angle=0., scale=1.):
    result = np.eye(4)
    result[:3, :3] = scale * Rotation.from_euler('y', angle).as_matrix()
    result[:3, 3] = translation
    return result


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.memory = EpisodicBindings(node_stride=8)
        for i in range(25):
            self.memory.append(i, transform((i*.1, 0, i*.2), i*.01))

    def test_unrevised_matches_original_pnp_relative_transform(self):
        goal_pose = transform((1, .2, 2.))
        goal = self.memory.bind_goal(4, goal_pose, evidence_id='unchanged-pnp')
        expected = np.linalg.solve(self.memory.version.frames[-1].original_world_from_camera, goal_pose)
        np.testing.assert_allclose(self.memory.read_goal(goal), expected, atol=1e-12)

    def test_goal_survives_node_update_without_changing_evidence(self):
        original = self.memory.version
        goal = self.memory.bind_goal(4, transform((1, .2, 2.)), evidence_id='proof')
        fixed = goal.reference_from_goal.copy()
        nodes = dict(original.nodes)
        nodes[0] = transform((.4, 0, -.1), .12) @ nodes[0]
        updated = self.memory.publish(nodes, expected_sequence=original.sequence)
        expected = np.linalg.solve(updated.camera(24), updated.camera(4)) @ fixed
        np.testing.assert_allclose(self.memory.read_goal(goal), expected, atol=1e-12)
        np.testing.assert_array_equal(goal.reference_from_goal, fixed)
        np.testing.assert_allclose(self.memory.read_goal(goal, version=original),
            original.relative(4, 24) @ fixed, atol=1e-12)
        self.assertFalse(np.allclose(self.memory.read_goal(goal), self.memory.read_goal(goal, version=original)))

    def test_common_sim3_gauge_does_not_change_goal_readout(self):
        goal = self.memory.bind_goal(4, transform((1, .2, 2.)), evidence_id='proof')
        original = self.memory.read_goal(goal)
        gauge = transform((4, -3, 2), .7, 2.4)
        v = self.memory.version
        self.memory.publish({k: gauge @ t for k, t in v.nodes.items()}, expected_sequence=v.sequence)
        np.testing.assert_allclose(self.memory.read_goal(goal), original, atol=1e-12)

    def test_new_nodes_inherit_latest_coordinate_revision(self):
        v = self.memory.version
        gauge = transform((4, -3, 2), .7, 2.4)
        self.memory.publish({k: gauge @ t for k, t in v.nodes.items()}, expected_sequence=v.sequence)
        for i in range(25, 41):
            raw = transform((i*.1, 0, i*.2), i*.01)
            self.memory.append(i, raw)
            np.testing.assert_allclose(self.memory.version.camera(i), gauge @ raw, atol=1e-12)
        with self.assertRaises(ValueError):
            v.camera(25)

    def test_publication_failure_leaves_version_untouched(self):
        v = self.memory.version
        invalid = dict(v.nodes)
        invalid[0] = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            self.memory.publish(invalid, expected_sequence=v.sequence)
        self.assertIs(self.memory.version, v)
        with self.assertRaises(ValueError):
            self.memory.publish(dict(v.nodes), expected_sequence=v.sequence-1)
        self.assertIs(self.memory.version, v)
        with self.assertRaises(ValueError):
            v.nodes[0].setflags(write=True)

    def test_queries_leave_memory_version_unchanged(self):
        v = self.memory.version
        goal = self.memory.bind_goal(4, transform((1, .2, 2.)), evidence_id='proof')
        for _ in range(5):
            self.memory.bearing(goal)
        self.assertIs(self.memory.version, v)

    def test_goal_and_version_cannot_cross_episode_reset(self):
        goal = self.memory.bind_goal(4, transform((1, .2, 2.)), evidence_id='proof')
        other = EpisodicBindings()
        for i in range(25):
            other.append(i, transform((0, 0, i*.1)))
        with self.assertRaises(ValueError):
            other.read_goal(goal)
        with self.assertRaises(ValueError):
            self.memory.read_goal(goal, version=other.version)


if __name__ == '__main__':
    unittest.main()
