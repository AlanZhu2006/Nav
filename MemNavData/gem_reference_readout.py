"""Experimental reference-bound view of the unchanged sparse PnP evidence.

This is a coordinate/readout adapter, not a new matcher, acceptance policy,
coordinate estimator or production mode. The caller owns a goal session and
passes one immutable published coordinate version for a read. Original PnP
poses must be expressed in first-write coordinates, as specified by bindings.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from NavDP.baselines.memnav.gem.bindings import EpisodicBindings, GoalAttachment
from MemNavData.certified_relocalization_runtime import _quat_xyzw_to_matrix


def pose_matrix(pose9):
    pose = np.asarray(pose9, dtype=np.float64)
    if pose.shape != (9,) or not np.isfinite(pose).all():
        raise ValueError('A finite original PnP pose9 is required')
    matrix = np.eye(4)
    matrix[:3, :3] = _quat_xyzw_to_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3]
    return matrix


def witness_digest(result):
    """Fields that must remain identical within an accepted goal session."""
    witness = {key:result.get(key) for key in (
        'accepted', 'reason', 'selected_anchor', 'selected_anchor_image_sha256',
        'pnp', 'certificate', 'authority')}
    return hashlib.sha256(json.dumps(witness, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class BoundGoalEvidence:
    session: str
    attachment: GoalAttachment
    witness_sha256: str


class ReferenceReadoutSession:
    """One session; construct a new instance on every A/B/A goal switch.

    Keeping raw first-write poses/depth as the PnP inputs preserves its local
    numerical result. Only downstream quantities derived from coordinates are
    resolved through the memory version. Dense metric depth is not touched.
    """

    def __init__(self, memory: EpisodicBindings, session: str):
        if not isinstance(session, str) or not session:
            raise ValueError('A distinct explicit goal session identity is required')
        self.memory = memory
        self.session = session
        self.goal = None
        self._witness_sha256 = None

    def read(self, original, *, version):
        if not original.get('ok'):
            raise ValueError('Failed sparse inference cannot be rebound')
        if version.memory_id != self.memory.version.memory_id:
            raise ValueError('Coordinate version belongs to another episode')
        if original.get('frame_idx') != version.observed_count - 1:
            raise ValueError('Sparse read and coordinate version use different observed prefixes')
        if original.get('certified_graph_enabled') or original.get('certified_graph_rescue_active'):
            raise ValueError('Only the original endpoint-bearing readout is supported')
        result = deepcopy(original)
        digest = witness_digest(original)
        if self._witness_sha256 is not None and digest != self._witness_sha256:
            raise ValueError('Cached localization evidence changed within this goal session')
        if original.get('accepted') is not True:
            # Normal abstention remains normal abstention; no inferred pose.
            self._witness_sha256 = digest
            return result
        if self.goal is None:
            attachment = self.memory.bind_goal(int(original['selected_anchor']),
                pose_matrix(original['pnp']['pose9']), evidence_id=digest)
            goal = BoundGoalEvidence(self.session, attachment, digest)
        else:
            goal = self.goal
            if digest != goal.witness_sha256:
                raise ValueError('PnP or its certificate changed within this goal session')
        # Reading an explicit snapshot also checks the reference's causal
        # existence. No mixed current/reference coordinate versions are used.
        relative = self.memory.read_goal(goal.attachment, version=version)
        vector = relative[:3, 3][[2, 0]] * np.array([1., -1.])
        axis = relative[:3, 2]
        if (not np.isfinite(relative).all() or np.linalg.norm(vector) <= 1e-12
                or np.linalg.norm(axis) <= 1e-12):
            raise ValueError('Updated target relation has no valid bearing or optical axis')
        axis = axis / np.linalg.norm(axis)
        yaw = math.degrees(math.atan2(axis[0], axis[2]))
        pitch = math.degrees(math.atan2(-axis[1], math.hypot(axis[0], axis[2])))
        result.update(aux_pose=vector.tolist(), direction_vector=vector.tolist(),
            terminal_yaw_right_deg=yaw, terminal_pitch_up_deg=pitch,
            terminal_alignment_source='gem_same_version_reference_bound_goal_rotation',
            terminal_alignment_stop_authority=False,
            memory_coordinate_version=version.sequence,
            memory_coordinate_id=version.memory_id,
            memory_goal_session=self.session,
            original_localization_sha256=digest,
            pnp_coordinate_source='unchanged_first_write_world_coordinates')
        # Publish the attachment only after a complete valid read. The caller
        # can still retain and read older coordinate snapshots explicitly.
        self.goal = goal
        self._witness_sha256 = digest
        return result
