"""Causal controls for unverified episodic direction on Novel queries.

This module contains no Habitat or model code.  It implements the two
interventions frozen in ``NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_PROTOCOL``:

* replay a donor RGB history into the long-term sidecar while retaining the
  factual NavDP decision FIFO; and
* preserve raw-proposal availability while replacing only its angle with a
  deterministic uniform bearing.

The helpers are deliberately small so the intervention can be audited without
importing the navigation policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "novel_memory_direction_control_v1_20260816"
ARMS = (
    "native",
    "raw_factual_history",
    "raw_deranged_history",
    "raw_randomized_bearing",
)
FIXED_RADIUS_M = 2.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _canonical_random_key(
    *,
    global_seed: int,
    scene: str,
    episode: str,
    plan_index: int,
) -> bytes:
    payload = [
        int(global_seed),
        str(scene),
        str(episode),
        int(plan_index),
        "random_bearing",
    ]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()


def deterministic_random_bearing(
    *,
    global_seed: int,
    scene: str,
    episode: str,
    plan_index: int,
) -> dict[str, Any]:
    """Map a frozen identity to one uniform angle in ``[-pi, pi)``.

    The first 64 SHA-256 bits are interpreted as a fixed-point value in
    ``[0, 1)``.  This avoids dependence on NumPy/Python RNG implementations.
    """

    if plan_index < 0:
        raise ValueError("plan_index must be non-negative")
    encoded = _canonical_random_key(
        global_seed=global_seed,
        scene=scene,
        episode=episode,
        plan_index=plan_index,
    )
    digest = hashlib.sha256(encoded).hexdigest()
    integer = int(digest[:16], 16)
    unit_interval = integer / float(1 << 64)
    angle = -math.pi + 2.0 * math.pi * unit_interval
    return {
        "key_sha256": digest,
        "angle_rad": angle,
        "unit_bearing": [math.cos(angle), math.sin(angle)],
    }


def _finite_pointgoal(pointgoal: Sequence[float] | None) -> tuple[float, float] | None:
    if pointgoal is None:
        return None
    try:
        if len(pointgoal) != 2:
            return None
        point = (float(pointgoal[0]), float(pointgoal[1]))
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in point):
        return None
    return point


@dataclass
class RandomizedBearingAdapter:
    """Callable adapter wrapper that destroys only raw directional content."""

    original_adapter: Callable[..., Any]
    global_seed: int
    scene: str
    episode: str
    query_id: str
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> Any:
        mode = kwargs.get("mode")
        if mode != "raw_fixed_bearing_v1":
            raise RuntimeError(
                "random-bearing control may wrap only raw_fixed_bearing_v1"
            )
        plan_index = len(self.ledger)
        factual = self.original_adapter(**kwargs)
        raw = _finite_pointgoal(kwargs.get("pointgoal"))
        randomization = deterministic_random_bearing(
            global_seed=self.global_seed,
            scene=self.scene,
            episode=self.episode,
            plan_index=plan_index,
        )
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "plan_index": plan_index,
            "random_key_sha256": randomization["key_sha256"],
            "raw_proposal_available": bool(factual.takeover),
            "factual_raw_pointgoal": list(raw) if raw is not None else None,
            "factual_unit_bearing": (
                list(factual.unit_bearing)
                if factual.unit_bearing is not None else None
            ),
            "randomized_angle_rad": None,
            "randomized_unit_bearing": None,
            "randomized_controller_pointgoal": None,
            "factual_takeover": bool(factual.takeover),
            "randomized_takeover": bool(factual.takeover),
            "fixed_radius_m": FIXED_RADIUS_M,
        }
        if not factual.takeover:
            self.ledger.append(record)
            return factual

        require(raw is not None, "takeover without a finite raw proposal")
        raw_norm = math.hypot(*raw)
        require(raw_norm > 0.0, "takeover without a non-zero raw proposal")
        random_unit = randomization["unit_bearing"]
        randomized_raw = [
            raw_norm * float(random_unit[0]),
            raw_norm * float(random_unit[1]),
        ]
        transformed_kwargs = dict(kwargs)
        transformed_kwargs["pointgoal"] = randomized_raw
        transformed_kwargs["source"] = (
            f"{kwargs.get('source', 'raw_history')}|deterministic_random_angle"
        )
        randomized = self.original_adapter(**transformed_kwargs)
        require(randomized.takeover, "angle replacement changed takeover availability")
        require(
            randomized.controller_distance_m is not None
            and math.isclose(
                float(randomized.controller_distance_m),
                FIXED_RADIUS_M,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "angle replacement changed the fixed controller radius",
        )
        record.update(
            randomized_angle_rad=float(randomization["angle_rad"]),
            randomized_unit_bearing=list(randomized.unit_bearing),
            randomized_controller_pointgoal=list(
                randomized.controller_pointgoal
            ),
            randomized_takeover=bool(randomized.takeover),
        )
        self.ledger.append(record)
        return randomized


def load_online_source(record: Mapping[str, Any]) -> dict[str, Any]:
    """Load one frozen online-A source and verify its immutable receipts."""

    source = Path(str(record["online_a_episode"]))
    require(source.is_dir(), f"online-A source is missing: {source}")
    receipt_path = source / "receipt.json"
    trace_path = source / "online_a_trace.json"
    require(
        sha256_file(receipt_path) == str(record["online_a_receipt_sha256"]),
        "online-A receipt hash changed",
    )
    require(
        sha256_file(trace_path) == str(record["online_a_trace_sha256"]),
        "online-A trace hash changed",
    )
    receipt = json.loads(receipt_path.read_text())
    trace = json.loads(trace_path.read_text())
    require(trace.get("reached") is True, "frozen online-A did not succeed")
    require(
        len(trace.get("poses") or []) == int(record["online_a_steps"]),
        "online-A frame count changed",
    )
    return {
        "source": source,
        "receipt": receipt,
        "trace": trace,
        "scene": str(record["scene"]),
        "episode": str(record["episode"]),
    }


def replay_deranged_sidecar(
    factual: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    memory_step: Callable[[bytes], Mapping[str, Any]],
    navdp_replay_step: Callable[[bytes], Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay donor RGB to MemNav and factual decision RGB to NavDP.

    This deliberately does not execute either policy.  Every source image is
    verified against the SHA stored by the original online rollout.
    """

    factual_source = Path(factual["source"])
    factual_trace = factual["trace"]
    donor_source = Path(donor["source"])
    donor_trace = donor["trace"]

    factual_poses = list(factual_trace["poses"])
    donor_poses = list(donor_trace["poses"])
    require(
        [int(pose["step"]) for pose in factual_poses]
        == list(range(len(factual_poses))),
        "factual online-A poses are not contiguous",
    )
    require(
        [int(pose["step"]) for pose in donor_poses]
        == list(range(len(donor_poses))),
        "donor online-A poses are not contiguous",
    )

    sidecar_trace = []
    sidecar_hashes = []
    for pose in donor_poses:
        step = int(pose["step"])
        image = donor_source / "rgb" / f"{step:06d}.jpg"
        digest = sha256_file(image)
        require(digest == pose["jpg_sha256"], "donor RGB hash changed")
        receipt = memory_step(image.read_bytes())
        frame_idx = receipt.get("frame_idx")
        if frame_idx is not None:
            require(int(frame_idx) == step, "MemNav donor index changed")
            sidecar_trace.append(
                {
                    "frame_idx": int(frame_idx),
                    "step": step,
                    "x": float(pose["x"]),
                    "z": float(pose["z"]),
                    "yaw": float(pose["yaw"]),
                }
            )
        sidecar_hashes.append(digest)
    require(
        not sidecar_trace or len(sidecar_trace) == len(donor_poses),
        "MemNav did not index every donor frame",
    )

    plan_steps = [int(plan["step"]) for plan in factual_trace["plans"]]
    require(plan_steps, "factual online-A contains no decision frame")
    require(len(plan_steps) == len(set(plan_steps)), "duplicate factual plan step")
    factual_by_step = {int(pose["step"]): pose for pose in factual_poses}
    require(
        set(plan_steps).issubset(factual_by_step),
        "factual decision lies outside online-A trace",
    )
    fifo_hashes = []
    final_queue_lengths = None
    final_memory_size = None
    for step in plan_steps:
        pose = factual_by_step[step]
        image = factual_source / "rgb" / f"{step:06d}.jpg"
        digest = sha256_file(image)
        require(digest == pose["jpg_sha256"], "factual FIFO RGB hash changed")
        receipt = navdp_replay_step(image.read_bytes())
        require(
            receipt.get("diffusion_sampled") is False,
            "NavDP replay unexpectedly sampled diffusion",
        )
        final_queue_lengths = receipt.get("queue_lengths")
        final_memory_size = int(receipt.get("memory_size", -1))
        fifo_hashes.append(digest)

    require(
        final_memory_size is not None and final_memory_size > 0,
        "NavDP replay omitted memory size",
    )
    expected_queue = min(len(plan_steps), final_memory_size)
    require(
        final_queue_lengths == [expected_queue],
        "NavDP queue length differs from factual decision count",
    )
    return {
        # These legacy fields deliberately describe the factual policy prefix,
        # so paired identity checks remain meaningful.
        "online_frames": len(factual_poses),
        "decision_frames": len(plan_steps),
        "decision_steps": plan_steps,
        "navdp_memory_size": final_memory_size,
        "navdp_queue_lengths": final_queue_lengths,
        "memory_trace": sidecar_trace,
        "all_rgb_hashes_verified": True,
        "diffusion_samples_during_replay": 0,
        # The split receipts make the intervention explicit.
        "causal_control_schema_version": SCHEMA_VERSION,
        "factual_fifo_scene": str(factual["scene"]),
        "factual_fifo_episode": str(factual["episode"]),
        "factual_fifo_frames": len(factual_poses),
        "factual_fifo_decision_sha256": aggregate_hash(fifo_hashes),
        "sidecar_scene": str(donor["scene"]),
        "sidecar_episode": str(donor["episode"]),
        "sidecar_memory_frames": len(donor_poses),
        "sidecar_memory_sha256": aggregate_hash(sidecar_hashes),
        "sidecar_is_deranged": True,
    }


def validate_control_manifest(payload: Mapping[str, Any]) -> None:
    require(payload.get("schema_version") == SCHEMA_VERSION, "control schema changed")
    require(payload.get("confirmation_claim_allowed") is False, "mechanism manifest may not claim confirmation")
    require(payload.get("query_role") == "novel", "control must contain only Novel queries")
    rows = list(payload.get("episodes") or [])
    require(bool(rows), "control manifest is empty")
    identities = {(str(row["scene"]), str(row["episode"])) for row in rows}
    require(len(identities) == len(rows), "duplicate factual identity")
    donors = []
    for row in rows:
        donor = row["donor"]
        donor_identity = (str(donor["scene"]), str(donor["episode"]))
        require(donor_identity in identities, "donor lies outside frozen population")
        require(
            donor_identity != (str(row["scene"]), str(row["episode"])),
            "derangement contains a fixed point",
        )
        require(tuple(row["arm_order"]) in _latin_rotations(), "arm order is not a frozen Latin rotation")
        donors.append(donor_identity)
    require(len(set(donors)) == len(donors), "donor mapping is not a permutation")


def _latin_rotations() -> set[tuple[str, ...]]:
    return {
        tuple(ARMS[offset:] + ARMS[:offset])
        for offset in range(len(ARMS))
    }


__all__ = [
    "ARMS",
    "FIXED_RADIUS_M",
    "RandomizedBearingAdapter",
    "SCHEMA_VERSION",
    "aggregate_hash",
    "deterministic_random_bearing",
    "load_online_source",
    "replay_deranged_sidecar",
    "sha256_file",
    "validate_control_manifest",
]
