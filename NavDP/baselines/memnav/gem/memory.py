"""Geometric Episodic Memory used by the paper's existing navigation pipeline.

GEM owns causal RGB/descriptor/pose history, LingBot working tensors, historical
depth and goal-conditioned certificates. The backend supplies the frozen model,
configuration and optional legacy route adapters; controller planning stays in
MemNavAgent. There is no source rewriting, diagnostic import or second writer.
"""
from weakref import ref

from .writer import CausalWriter
from .dense import DenseReadout
from .sparse import SparseReadout


class MemoryField:
    """Explicit compatibility alias to one GEM-owned value (never a copy)."""

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance.memory, self.name)

    def __set__(self, instance, value):
        setattr(instance.memory, self.name, value)


class GeometricEpisodicMemory(CausalWriter, DenseReadout, SparseReadout):
    """One episode memory, with reset/write/retrieve/read_dense/read_sparse APIs.

    Instantiate through MemNavAgent.memory so reset and model configuration stay
    synchronized. Reads retain the existing canonical/online_history choice.
    Query JPEGs are separate from the numbered causal observation history.
    """

    def __init__(self, backend):
        self._backend = ref(backend)
        self.frame_count = 0
        self.rgb_dir = None
        self.pending_images = []
        self.window_images = []
        self.descriptors = []
        self.poses = []
        self.anchor_frames = []
        self.camera_frames = []
        self.anchor_k = []
        self.anchor_v = []
        self.camera_k = []
        self.camera_v = []
        self.scale_k = None
        self.scale_v = None
        self.last_keyframe_pose = None
        self.last_keyframe_index = -1
        self.current_tokens = None
        self.current_aggregation = None
        self.patch_start_index = None
        self.goal_embeddings_and_poses = {}
        self.goal_start_frames = {}
        self.shortlists = {}
        self.certificates = {}
        self.active_goal_key = None
        self.goal_session_index = 0
        self.goal_session_started = False
        self.replay_depths = {}
        self.online_depths = {}
        self.online_depth_frames = set()
        self.last_reference_read = None
        self.metric_scale_receipt = None
        self.metric_scale_freeze_ms = None
        self.current_rgb_sha256 = None
        self.current_depth_frame = None
        self.current_relative_depth = None

    @property
    def backend(self):
        backend = self._backend()
        if backend is None:
            raise RuntimeError("GEM's model backend has been released")
        return backend

    def status(self):
        """Describe stored evidence without materializing tensors or depth."""
        return {
            "module": "geometric_episodic_memory_v1",
            "frames": self.frame_count,
            "descriptor_frames": len(self.descriptors),
            "pose_frames": len(self.poses),
            "working_keyframes": len(self.anchor_frames),
            "online_depth_frames": len(self.online_depths),
            "online_depth_bytes": sum(a.nbytes for pair in self.online_depths.values() for a in pair),
            "replayed_depth_frames": len(self.replay_depths),
            "cached_goals": len(self.certificates),
            "historical_depth_source": getattr(self.backend, "certified_reference_depth_source", "canonical"),
        }
