"""Versioned coordinates over immutable episodic evidence.

This is the coordinate contract, independent of the estimator/updater. Original
camera-local geometry and a goal's PnP attachment are never overwritten when a
memory node moves. A read uses one immutable, causally bounded version.
"""
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from uuid import uuid4

import numpy as np


def immutable_similarity(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('A finite 4x4 similarity is required')
    if not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-10, rtol=0):
        raise ValueError('Invalid homogeneous row')
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    if scale <= 0:
        raise ValueError('Similarity must preserve orientation with positive scale')
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Anisotropic or nonrigid transform is not a Sim(3)')
    # bytes owns the storage, so callers cannot re-enable writeability.
    return np.frombuffer(matrix.tobytes(), dtype=np.float64).reshape(4, 4)


@dataclass(frozen=True)
class FrameAttachment:
    frame: int
    node: int
    node_from_camera: np.ndarray
    original_world_from_camera: np.ndarray


@dataclass(frozen=True)
class CoordinateVersion:
    memory_id: str
    sequence: int
    observed_count: int
    nodes: object
    frames: tuple

    def camera(self, frame):
        if not 0 <= frame < self.observed_count:
            raise ValueError('Frame is outside this published causal prefix')
        attachment = self.frames[frame]
        return self.nodes[attachment.node] @ attachment.node_from_camera

    def relative(self, source, target):
        """Transform coordinates from source camera into target camera."""
        return np.linalg.solve(self.camera(target), self.camera(source))


@dataclass(frozen=True)
class GoalAttachment:
    memory_id: str
    reference: int
    reference_from_goal: np.ndarray
    evidence_id: str


class EpisodicBindings:
    """Append observations, publish node revisions, and read stable goals.

    Nodes occur at a fixed frame cadence; frame identity and local attachment
    do not depend on retrieval or on a goal. The estimator supplies proposed
    node coordinates. Publication validates the whole update before swapping
    one version, and rejects an update computed from an obsolete version.
    """

    def __init__(self, node_stride=8):
        if int(node_stride) != node_stride or node_stride < 1:
            raise ValueError('Node stride must be a positive integer')
        self.node_stride = int(node_stride)
        self._lock = RLock()
        self._memory_id = uuid4().hex
        self._version = CoordinateVersion(self._memory_id, 0, 0, MappingProxyType({}), ())

    @property
    def version(self):
        return self._version

    def append(self, frame, original_world_from_camera):
        raw = immutable_similarity(original_world_from_camera)
        with self._lock:
            previous = self._version
            if frame != previous.observed_count:
                raise ValueError('Observation identities must append contiguously')
            node = (frame // self.node_stride) * self.node_stride
            nodes = dict(previous.nodes)
            if frame == node:
                if frame == 0:
                    binding = raw
                else:
                    parent = previous.frames[-1].node
                    raw_parent = previous.frames[parent].original_world_from_camera
                    binding = nodes[parent] @ np.linalg.solve(raw_parent, raw)
                nodes[node] = immutable_similarity(binding)
                attachment = immutable_similarity(np.eye(4))
            else:
                raw_node = previous.frames[node].original_world_from_camera
                attachment = immutable_similarity(np.linalg.solve(raw_node, raw))
            record = FrameAttachment(frame, node, attachment, raw)
            self._version = CoordinateVersion(self._memory_id, previous.sequence + 1, frame + 1,
                MappingProxyType(nodes), previous.frames + (record,))
            return self._version

    def publish(self, nodes, *, expected_sequence):
        with self._lock:
            previous = self._version
            if previous.sequence != expected_sequence:
                raise ValueError('Coordinate update was computed from an obsolete version')
            if set(nodes) != set(previous.nodes):
                raise ValueError('Publication must bind every existing memory node exactly once')
            validated = {key: immutable_similarity(value) for key, value in nodes.items()}
            self._version = CoordinateVersion(self._memory_id, previous.sequence + 1,
                previous.observed_count, MappingProxyType(validated), previous.frames)
            return self._version

    def bind_goal(self, reference, original_world_from_goal, *, evidence_id):
        """Attach an unchanged original PnP result to its historical camera."""
        version = self._version
        if not 0 <= reference < version.observed_count:
            raise ValueError('Goal reference must have been observed')
        raw_goal = immutable_similarity(original_world_from_goal)
        raw_reference = version.frames[reference].original_world_from_camera
        attachment = immutable_similarity(np.linalg.solve(raw_reference, raw_goal))
        return GoalAttachment(self._memory_id, reference, attachment, str(evidence_id))

    def read_goal(self, goal, current=None, *, version=None):
        selected = self._version if version is None else version
        if selected.memory_id != self._memory_id or goal.memory_id != self._memory_id:
            raise ValueError('Goal or coordinate version belongs to another episode memory')
        current = selected.observed_count - 1 if current is None else current
        return selected.relative(goal.reference, current) @ goal.reference_from_goal

    def bearing(self, goal, current=None, *, version=None):
        relative = self.read_goal(goal, current, version=version)
        xy = relative[:3, 3][[2, 0]] * np.array([1., -1.])
        norm = np.linalg.norm(xy)
        if not np.isfinite(norm) or norm <= 1e-10:
            raise ValueError('Goal has no finite nonzero horizontal bearing')
        return xy / norm
