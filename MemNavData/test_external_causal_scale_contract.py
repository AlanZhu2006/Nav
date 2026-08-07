import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from MemNavData.build_lingbot_native_causal_seed_envelope import (
    build_seed_envelope,
)
from MemNavData.build_causal_ground_scale import (
    canonical_json_bytes as producer_canonical_json_bytes,
)

from MemNavData.external_causal_scale_contract import (
    CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
    EXTERNAL_CAUSAL_SCALE_SOURCE,
    ExternalCausalScaleContract,
    ExternalCausalScaleError,
    ExternalCausalScalePins,
    canonical_json_bytes,
    ndarray_sha256,
    sha256_bytes,
    sha256_file,
    validate_external_causal_frame,
)


def file_record(path: Path, root: Path) -> dict:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "content_sha256": sha256_file(path),
    }


def rgb_prefix_record(root: Path, scene: str, episode: str,
                      frame_count: int) -> dict:
    rows = []
    rgb = root / scene / episode / "videos/chunk-000/observation.images.rgb"
    for frame in range(frame_count):
        path = rgb / f"{frame}.jpg"
        rows.append(file_record(path, root))
    return {
        "frame_count": frame_count,
        "path_sequence_sha256": sha256_bytes(canonical_json_bytes(
            [row["path"] for row in rows])),
        "content_sequence_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


class ExternalCausalScaleContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "episodes"
        self.scene = "scene_a"
        self.episode = "episode_0000"
        self.episode_root = self.root / self.scene / self.episode
        self.rgb = (self.episode_root / "videos/chunk-000"
                    / "observation.images.rgb")
        self.rgb.mkdir(parents=True)
        self.n_frames = 48
        for frame in range(self.n_frames):
            (self.rgb / f"{frame}.jpg").write_bytes(
                f"rgb-{frame:03d}".encode())
        self.goal_b = self.episode_root / "goal_1.jpg"
        self.goal_c = self.episode_root / "goal_2.jpg"
        self.goal_b.write_bytes(b"goal-b")
        self.goal_c.write_bytes(b"goal-c")
        self.metadata = self.episode_root / "gen_meta.json"
        self.metadata.write_text(json.dumps({
            "n_frames": self.n_frames,
            "camera_height_m": 0.5,
        }))
        self.cam_pose = np.zeros((self.n_frames, 9), dtype=np.float32)
        self.config = {
            "prefix_frame_cap": 64,
            "num_scale_frames": 8,
            "bias_correction": 1.15,
            "scale_min": 0.8,
            "scale_max": 6.0,
        }
        self.producer_sha = "1" * 64
        self.commit = "2" * 40
        self.weights_sha = "3" * 64
        self.stream_sha = "4" * 64
        self.manifest = self.make_manifest()
        self.manifest_path = Path(self.temporary.name) / "manifest.json"
        self.write_manifest(self.manifest)
        self.artifact = self.make_artifact(self.manifest)
        self.artifact_path = Path(self.temporary.name) / "scale.json"
        self.write_artifact(self.artifact)

    def tearDown(self):
        self.temporary.cleanup()

    def test_consumer_uses_exact_producer_canonical_byte_contract(self):
        witness = {"unicode": "尺度", "nested": {"value": 1}}
        self.assertEqual(canonical_json_bytes(witness),
                         producer_canonical_json_bytes(witness))
        self.assertTrue(canonical_json_bytes(witness).endswith(b"\n"))

    def sample(self, sample_id: str, state: str, decision: int,
               goal_role: str) -> dict:
        goal_path = self.goal_b if goal_role == "B" else self.goal_c
        return {
            "sample_id": sample_id,
            "split_role": "train",
            "scene": self.scene,
            "state_source": "expert",
            "source_episode": self.episode,
            "source_episode_id": f"{self.scene}/{self.episode}",
            "goal_episode": self.episode,
            "goal_source_episode_id": f"{self.scene}/{self.episode}",
            "goal_variant": "factual",
            "goal_role": goal_role,
            "state_name": state,
            "decision_frame": decision,
            "causal_prefix": {
                "frame_count": decision,
                "causal_prefix_sha256": hashlib.sha256(
                    f"prefix-{decision}".encode()).hexdigest(),
            },
            "navdp_fifo": {
                "fifo_sha256": hashlib.sha256(
                    f"fifo-{decision}".encode()).hexdigest(),
            },
            "goal": file_record(goal_path, self.root),
        }

    def make_manifest(self) -> dict:
        return {
            "schema_version": (
                "nlsr_v2_multistage_expert_candidate_manifest_v1"),
            "input_roots": {"episode_root": str(self.root.resolve())},
            "scenes": [{
                "scene": self.scene,
                "split_role": "train",
                "selected_episodes": [{
                    "episode": self.episode,
                    "n_frames": self.n_frames,
                    "metadata": file_record(self.metadata, self.root),
                }],
            }],
            "samples": [
                self.sample("sample_b_t0", "goal_b_t0", 16, "B"),
                self.sample("sample_b_mid", "goal_b_midpoint_t1", 32, "B"),
                self.sample("sample_c", "goal_c_t2", 40, "C"),
            ],
        }

    def write_manifest(self, manifest: dict):
        payload = canonical_json_bytes(manifest)
        self.manifest_path.write_bytes(payload)
        self.manifest_sha = sha256_bytes(payload)

    def make_artifact(self, manifest: dict, *, valid=True) -> dict:
        decisions = sorted({
            int(row["decision_frame"]) for row in manifest["samples"]})
        sample_ids = sorted(row["sample_id"] for row in manifest["samples"])
        prefix = min(self.config["prefix_frame_cap"], decisions[0])
        scale = 1.15 * 0.5 / 0.25
        return {
            "schema_version": CAUSAL_SCALE_ARTIFACT_SCHEMA_VERSION,
            "purpose": "test causal first-prefix scale",
            "provenance": {
                "input_manifest_sha256": self.manifest_sha,
                "input_manifest_schema_version": manifest["schema_version"],
                "producer_source_sha256": self.producer_sha,
                "configuration_sha256": sha256_bytes(
                    canonical_json_bytes(self.config)),
                "estimator": {
                    "kind": "frozen_lingbot_compute_metric_scale_prefix",
                    "lingbot_commit": self.commit,
                    "weights_sha256": self.weights_sha,
                    "lingbot_stream_source_sha256": self.stream_sha,
                },
            },
            "configuration": self.config,
            "records": [{
                "scene": self.scene,
                "episode": self.episode,
                "split_role": "train",
                "sample_ids": sample_ids,
                "decision_frames": decisions,
                "earliest_decision_frame": decisions[0],
                "prefix_end_frame_exclusive": prefix,
                "camera_height_m": 0.5,
                "episode_frame_count": self.n_frames,
                "rgb_prefix": rgb_prefix_record(
                    self.root, self.scene, self.episode, prefix),
                "cam_pose_prefix_sha256": ndarray_sha256(
                    self.cam_pose[:prefix]),
                "cam_pose_prefix_dtype": self.cam_pose.dtype.str,
                "cache_schema_version": 2,
                "precompute_signature": "pinned-precompute-v1",
                "valid": valid,
                "invalid_reason": None if valid else "no_floor",
                "ground_h_est_raw": 0.25 if valid else None,
                "metric_scale_m_per_raw": scale if valid else None,
                "debug": {
                    "n_points": 128,
                    "n_frames": 8,
                    "n_valid": 8 if valid else 0,
                    "h_est": 0.25 if valid else None,
                    "h_iqr": 0.025 if valid else None,
                },
            }],
            "summary": {"future_frames_consumed": 0},
        }

    def write_artifact(self, artifact: dict):
        payload = canonical_json_bytes(artifact)
        self.artifact_path.write_bytes(payload)
        self.artifact_sha = sha256_bytes(payload)

    def pins(self) -> ExternalCausalScalePins:
        return ExternalCausalScalePins(
            manifest_sha256=self.manifest_sha,
            artifact_sha256=self.artifact_sha,
            producer_sha256=self.producer_sha,
            configuration_sha256=sha256_bytes(canonical_json_bytes(self.config)),
            lingbot_commit=self.commit,
            weights_sha256=self.weights_sha,
            stream_source_sha256=self.stream_sha,
        )

    def contract(self) -> ExternalCausalScaleContract:
        return ExternalCausalScaleContract(
            manifest_path=self.manifest_path,
            artifact_path=self.artifact_path,
            pins=self.pins(),
        )

    def bind(self, contract, sample_id="sample_b_t0", frame=8,
             offsets=(0,), query=None, candidate=None):
        return contract.bind_seed(
            manifest_sample_id=sample_id,
            scene=self.scene,
            episode=self.episode,
            query_path=query or self.goal_b,
            candidate_path=candidate or self.rgb / f"{frame}.jpg",
            candidate_frame=frame,
            neighbor_offsets=offsets,
            expected_split_role="train",
        )

    def test_one_episode_estimate_binds_distinct_multistage_samples(self):
        contract = self.contract()
        contract.validate_runtime_episode(
            scene=self.scene,
            episode=self.episode,
            cam_pose_enc=self.cam_pose,
            cache_schema_version=2,
            precompute_signature="pinned-precompute-v1",
        )
        t0 = self.bind(contract, "sample_b_t0", frame=8)
        midpoint = self.bind(contract, "sample_b_mid", frame=20)
        self.assertNotEqual(t0.sample_id, midpoint.sample_id)
        self.assertEqual(t0.goal_sha256, midpoint.goal_sha256)
        self.assertEqual(t0.decision_frame, 16)
        self.assertEqual(midpoint.decision_frame, 32)
        self.assertEqual(t0.metric_scale_m_per_raw, 2.3)
        self.assertEqual(t0.valid_frame_ratio, 1.0)
        self.assertAlmostEqual(t0.relative_h_iqr, 0.1)
        self.assertEqual(t0.clamped, 0)
        self.assertEqual(EXTERNAL_CAUSAL_SCALE_SOURCE,
                         "external_causal_first_prefix_v1")

    def test_unknown_sample_and_future_neighbor_fail_closed(self):
        contract = self.contract()
        with self.assertRaisesRegex(ExternalCausalScaleError, "known causal"):
            self.bind(contract, "missing")
        with self.assertRaisesRegex(ExternalCausalScaleError, "future decision"):
            self.bind(contract, "sample_b_t0", frame=14, offsets=(0, 2))

    def test_query_bytes_and_cross_episode_candidate_are_rebound(self):
        contract = self.contract()
        self.goal_b.write_bytes(b"changed-after-manifest")
        with self.assertRaisesRegex(ExternalCausalScaleError, "query bytes"):
            self.bind(contract)
        self.goal_b.write_bytes(b"goal-b")
        outside = Path(self.temporary.name) / "other" / "8.jpg"
        outside.parent.mkdir()
        outside.write_bytes(b"rgb-008")
        with self.assertRaisesRegex(ExternalCausalScaleError, "crosses"):
            self.bind(contract, candidate=outside)

    def test_runtime_pose_prefix_change_fails(self):
        contract = self.contract()
        changed = self.cam_pose.copy()
        changed[0, 0] = 1.0
        with self.assertRaisesRegex(ExternalCausalScaleError, "prefix changed"):
            contract.validate_runtime_episode(
                scene=self.scene,
                episode=self.episode,
                cam_pose_enc=changed,
                cache_schema_version=2,
                precompute_signature="pinned-precompute-v1",
            )

    def test_invalid_external_estimate_never_uses_pooled_fallback(self):
        self.artifact = self.make_artifact(self.manifest, valid=False)
        self.write_artifact(self.artifact)
        with self.assertRaisesRegex(ExternalCausalScaleError, "is invalid"):
            self.contract()

    def test_artifact_sha_pin_and_scale_formula_are_enforced(self):
        wrong_pins = copy.copy(self.pins())
        wrong_pins = ExternalCausalScalePins(
            **{**wrong_pins.__dict__, "artifact_sha256": "f" * 64})
        with self.assertRaisesRegex(ExternalCausalScaleError, "SHA changed"):
            ExternalCausalScaleContract(
                manifest_path=self.manifest_path,
                artifact_path=self.artifact_path,
                pins=wrong_pins,
            )
        changed = copy.deepcopy(self.artifact)
        changed["records"][0]["metric_scale_m_per_raw"] = 5.0
        self.write_artifact(changed)
        with self.assertRaisesRegex(ExternalCausalScaleError, "pinned formula"):
            self.contract()

    def test_ambiguous_manifest_join_key_is_rejected(self):
        duplicate = copy.deepcopy(self.manifest["samples"][0])
        duplicate["sample_id"] = "duplicate_join_key"
        self.manifest["samples"].append(duplicate)
        self.write_manifest(self.manifest)
        self.artifact = self.make_artifact(self.manifest)
        self.write_artifact(self.artifact)
        with self.assertRaisesRegex(ExternalCausalScaleError, "join key is ambiguous"):
            self.contract()

    def test_missing_episode_scale_record_fails_exact_cover(self):
        self.artifact["records"] = []
        self.write_artifact(self.artifact)
        with self.assertRaisesRegex(ExternalCausalScaleError,
                                    "no records|exactly cover"):
            self.contract()

    def teacher_rows(self, *, include_c=True):
        rows = []
        for goal, frames, session in (
                (self.goal_b, (8, 20), "legacy_b"),
                (self.goal_c, (8, 20, 35), "legacy_c")):
            if goal == self.goal_c and not include_c:
                continue
            for frame in frames:
                rows.append({
                    "session_id": session,
                    "scene": self.scene,
                    "episode": self.episode,
                    "kind": "revisit",
                    "query_path": str(goal),
                    "candidate_path": str(self.rgb / f"{frame}.jpg"),
                    "candidate_frame": frame,
                    "dino_cosine": 1.0 - frame / 100.0,
                    "teacher_covis": 0.8 if frame == 8 else 0.1,
                })
        return pd.DataFrame(rows)

    def test_seed_envelope_expands_same_goal_per_manifest_decision(self):
        rows, report = build_seed_envelope(
            teacher=self.teacher_rows(),
            contract=self.contract(),
            split_role="train",
            kind="revisit",
            goal_roles=("B", "C"),
            neighbor_offsets=(0,),
        )
        by_sample = rows.groupby("causal_manifest_sample_id")[
            "candidate_frame"].apply(list).to_dict()
        self.assertEqual(by_sample["sample_b_t0"], [8])
        self.assertEqual(by_sample["sample_b_mid"], [8, 20])
        self.assertEqual(by_sample["sample_c"], [8, 20, 35])
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["session_count"], 3)
        self.assertTrue(report["binding_approved"])

    def test_seed_envelope_requires_exact_selected_sample_coverage(self):
        with self.assertRaisesRegex(RuntimeError, "exact-cover"):
            build_seed_envelope(
                teacher=self.teacher_rows(include_c=False),
                contract=self.contract(),
                split_role="train",
                kind="revisit",
                goal_roles=("B", "C"),
                neighbor_offsets=(0,),
            )

    def test_row_validator_rejects_partial_and_nan_sample_coverage(self):
        rows, _ = build_seed_envelope(
            teacher=self.teacher_rows(),
            contract=self.contract(),
            split_role="train",
            kind="revisit",
            goal_roles=("B", "C"),
            neighbor_offsets=(0,),
        )
        contract = self.contract()
        for index, row in rows.iterrows():
            binding = self.bind(
                contract,
                sample_id=str(row["causal_manifest_sample_id"]),
                frame=int(row["candidate_frame"]),
                query=Path(str(row["query_path"])),
                candidate=Path(str(row["candidate_path"])),
            )
            rows.loc[index, "metric_scale_source"] = (
                EXTERNAL_CAUSAL_SCALE_SOURCE)
            rows.loc[index, "metric_scale_m_per_raw"] = (
                binding.metric_scale_m_per_raw)
            for column, value in binding.row_fields().items():
                rows.loc[index, column] = value
        approved = validate_external_causal_frame(
            rows,
            expected_sample_ids=("sample_b_t0", "sample_b_mid", "sample_c"),
            expected_split_role="train",
            expected_row_bindings={
                sample_id: contract.expected_row_binding(sample_id)
                for sample_id in ("sample_b_t0", "sample_b_mid", "sample_c")
            },
        )
        self.assertTrue(approved["exact_manifest_sample_coverage_approved"])
        partial = rows.loc[
            rows["causal_manifest_sample_id"] != "sample_c"].copy()
        with self.assertRaisesRegex(ExternalCausalScaleError, "exactly cover"):
            validate_external_causal_frame(
                partial,
                expected_sample_ids=(
                    "sample_b_t0", "sample_b_mid", "sample_c"),
                expected_split_role="train",
            )
        malformed = rows.copy()
        malformed.loc[malformed.index[0], "causal_manifest_sample_id"] = np.nan
        with self.assertRaisesRegex(ExternalCausalScaleError, "missing"):
            validate_external_causal_frame(malformed)
        rebound = rows.copy()
        rebound.loc[
            rebound["causal_manifest_sample_id"].eq("sample_b_t0"),
            "causal_goal_sha256",
        ] = "f" * 64
        with self.assertRaisesRegex(ExternalCausalScaleError,
                                    "changed causal_goal_sha256"):
            validate_external_causal_frame(
                rebound,
                expected_sample_ids=(
                    "sample_b_t0", "sample_b_mid", "sample_c"),
                expected_split_role="train",
                expected_row_bindings={
                    sample_id: contract.expected_row_binding(sample_id)
                    for sample_id in (
                        "sample_b_t0", "sample_b_mid", "sample_c")
                },
            )


if __name__ == "__main__":
    unittest.main()
