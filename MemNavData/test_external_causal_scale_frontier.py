import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from MemNavData.build_causal_ground_scale import (
    GroundScaleConfiguration,
    GroundScaleEstimate,
    build_scale_artifact,
)
from MemNavData.build_novel_candidate_manifest import (
    build_manifest,
    canonical_json_bytes as manifest_json_bytes,
)
from MemNavData.build_novel_frontier_candidates import (
    CandidateBuildError,
    DEPLOYMENT_ARM,
    DeploymentPoseError,
    build_artifact,
    canonical_json_bytes,
    lingbot_deployment_se2_rows,
    load_external_ground_scale_bindings,
    sha256_bytes,
)


M_W = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

LINGBOT_COMMIT = "1" * 40
WEIGHTS_SHA256 = "2" * 64
STREAM_SHA256 = "3" * 64


class FrozenFakeScaleEstimator:
    def provenance(self):
        return {
            "kind": "frozen_lingbot_compute_metric_scale_prefix",
            "lingbot_commit": LINGBOT_COMMIT,
            "weights_path": "/frozen/fake/model.pt",
            "weights_sha256": WEIGHTS_SHA256,
            "lingbot_stream_source_sha256": STREAM_SHA256,
            "device": "cuda:unit-test",
        }

    def estimate(self, *, rgb_paths, cam_pose_prefix, camera_height_m,
                 configuration):
        del rgb_paths, cam_pose_prefix
        ground_h = 0.25
        raw_scale = (
            configuration.bias_correction * camera_height_m / ground_h)
        scale = min(max(raw_scale, configuration.scale_min),
                    configuration.scale_max)
        return GroundScaleEstimate(
            ground_h_est_raw=ground_h,
            metric_scale_m_per_raw=scale,
            debug={"kind": "deterministic_fake_floor"},
        )


class ExternalCausalScaleFrontierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.episodes = cls.root / "episodes"
        cls.flow = cls.root / "flow"
        cls.environments = cls.root / "environments"
        cls.navmeshes = cls.root / "navmeshes"
        for root in (
                cls.episodes, cls.flow, cls.environments, cls.navmeshes):
            root.mkdir()
        cls.scene = "scene_external_scale"
        cls.split_path = cls.root / "split.json"
        cls.split_path.write_text(json.dumps({
            "version": "external_scale_fixture_v1",
            "train": [cls.scene],
            "development": [],
            "final_reserved": [],
        }), encoding="utf-8")
        (cls.environments / f"{cls.scene}.glb").write_bytes(b"fake glb")
        (cls.navmeshes / f"{cls.scene}.navmesh").write_bytes(
            b"fake navmesh")
        cls._make_episode("episode_0000", dense_prefix=False)
        cls._make_episode("episode_0001", dense_prefix=True)

        preliminary = cls._new_manifest()
        cls._bind_dense_prefix(preliminary)
        cls.manifest = cls._new_manifest()
        cls.manifest_path = cls.root / "manifest.json"
        cls.manifest_bytes = manifest_json_bytes(cls.manifest)
        cls.manifest_path.write_bytes(cls.manifest_bytes)
        cls.manifest_sha = sha256_bytes(cls.manifest_bytes)

        cls.scale_configuration = GroundScaleConfiguration(
            prefix_frame_cap=8,
            num_scale_frames=8,
            window=32,
            max_frame_num=64,
        )
        cls.scale_artifact = build_scale_artifact(
            manifest=cls.manifest,
            manifest_path=cls.manifest_path,
            expected_manifest_sha256=cls.manifest_sha,
            estimator=FrozenFakeScaleEstimator(),
            configuration=cls.scale_configuration,
            cache_pair_validator=lambda aggregator, camera, count: None,
        )
        cls.scale_path = cls.root / "causal_scale.json"
        cls.scale_path.write_bytes(canonical_json_bytes(cls.scale_artifact))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _new_manifest(cls):
        return build_manifest(
            split_path=cls.split_path,
            episode_root=cls.episodes,
            flow_cache_root=cls.flow,
            environment_root=cls.environments,
            navmesh_root=cls.navmeshes,
            roles=("train",),
        )

    @classmethod
    def _make_episode(cls, name, *, dense_prefix):
        episode = cls.episodes / cls.scene / name
        rgb = episode / "videos/chunk-000/observation.images.rgb"
        depth = episode / "videos/chunk-000/observation.images.depth"
        parquet_dir = episode / "data/chunk-000"
        metadata_dir = episode / "meta"
        for path in (rgb, depth, parquet_dir, metadata_dir):
            path.mkdir(parents=True)
        metadata = {
            "scene": f"{cls.scene}.glb",
            "ep_idx": int(name.rsplit("_", 1)[1]),
            "n_frames": 48,
            "n_legs": 3,
            "switches": [8, 40],
            "camera_height_m": 0.5,
            "frame_convention": "positions+parquet in data(Zup,M_W)",
            "goals": [
                {"name": "B", "kind": "novel", "pos": [1.0, 0.0, 0.0]},
                {"name": "C", "kind": "revisit", "pos": [0.0, 0.0, 0.0]},
            ],
        }
        (metadata_dir / "gen_meta.json").write_text(
            json.dumps(metadata), encoding="utf-8")
        Image.new("RGB", (32, 24), (120, 80, 40)).save(
            episode / "goal_1.jpg")
        Image.new("RGB", (32, 24), (40, 80, 120)).save(
            episode / "goal_2.jpg")
        intrinsic = np.asarray([
            [25.0, 0.0, 15.5],
            [0.0, 25.0, 11.5],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = M_W.astype(np.float32)
        extrinsic[:3, 3] = M_W @ np.asarray([0.0, 0.5, 0.0])
        rows = []
        for frame in range(48):
            Image.new("RGB", (32, 24), (frame, 20, 30)).save(
                rgb / f"{frame}.jpg")
            depth_array = np.full((24, 32), 60000, dtype=np.uint16)
            depth_array[:, 12:20] = 25000
            Image.fromarray(depth_array).save(depth / f"{frame}.png")
            action = np.eye(4, dtype=np.float32)
            action[:3, :3] = M_W.astype(np.float32)
            action[1, 3] = frame * 0.05
            rows.append({
                "index": frame,
                "observation.camera_intrinsic": intrinsic.tolist(),
                "observation.camera_extrinsic": extrinsic.tolist(),
                "action": action.tolist(),
            })
        pq.write_table(
            pa.Table.from_pylist(rows),
            parquet_dir / "episode_000000.parquet",
        )

        chunk = cls.flow / cls.scene / name / "videos/chunk-000"
        chunk.mkdir(parents=True)
        num_frames, num_scale, interval = 48, 8, 4
        anchor_indices = np.arange(
            num_scale, num_frames, interval, dtype=np.int64)
        cam_indices = np.concatenate((
            np.arange(num_scale, dtype=np.int64), anchor_indices))
        shared = {
            "cache_schema_version": np.asarray([2], dtype=np.int64),
            "keyframe_policy": np.asarray(["post_scale_mod_v1"]),
            "num_frames": np.asarray([num_frames], dtype=np.int64),
            "num_scale_frames": np.asarray([num_scale], dtype=np.int64),
            "keyframe_interval": np.asarray([interval], dtype=np.int64),
            "precompute_signature": np.asarray(["external_scale_fixture"]),
        }
        np.savez(
            chunk / "lingbot_cache.npz",
            dino_cls=np.zeros((num_frames, 1), dtype=np.float16),
            anchor_k=np.zeros((len(anchor_indices), 1), dtype=np.float16),
            anchor_v=np.zeros((len(anchor_indices), 1), dtype=np.float16),
            anchor_frame_indices=anchor_indices,
            **shared,
        )
        raw_pose = np.zeros((num_frames, 9), dtype=np.float32)
        raw_pose[:, 2] = np.arange(num_frames, dtype=np.float32) * 0.025
        raw_pose[:, 6] = 1.0
        camera_payload = {
            "cam_pose_enc": raw_pose,
            "cam_k": np.zeros((len(cam_indices), 1), dtype=np.float16),
            "cam_v": np.zeros((len(cam_indices), 1), dtype=np.float16),
            "cam_frame_indices": cam_indices,
            "ground_h_est": np.asarray(0.25, dtype=np.float32),
            **shared,
        }
        if dense_prefix:
            camera_payload.update({
                "ground_h_est_prefix": np.full(
                    num_frames, 0.25, dtype=np.float32),
                "ground_h_est_prefix_frame_indices": np.arange(
                    num_frames, dtype=np.int64),
                "ground_h_est_prefix_causal_prefix_sha256": np.full(
                    num_frames, "", dtype="<U64"),
                "ground_h_est_prefix_semantics": np.asarray(
                    "causal_prefix_floor_hist_v1"),
                "ground_h_est_prefix_builder_sha256": np.asarray("a" * 64),
                "ground_h_est_prefix_configuration_sha256": np.asarray(
                    "b" * 64),
            })
        np.savez(chunk / "lingbot_cam_cache.npz", **camera_payload)

    @classmethod
    def _bind_dense_prefix(cls, manifest):
        path = cls._camera_path("episode_0001")
        with np.load(path, allow_pickle=False) as source:
            payload = {name: source[name] for name in source.files}
        hashes = payload["ground_h_est_prefix_causal_prefix_sha256"].copy()
        for sample in manifest["samples"]:
            if sample["source_episode"] == "episode_0001":
                hashes[sample["decision_frame"] - 1] = sample[
                    "causal_prefix"]["causal_prefix_sha256"]
        payload["ground_h_est_prefix_causal_prefix_sha256"] = hashes
        np.savez(path, **payload)

    @classmethod
    def _camera_path(cls, episode="episode_0000"):
        return (
            cls.flow / cls.scene / episode /
            "videos/chunk-000/lingbot_cam_cache.npz"
        )

    @classmethod
    def _aggregator_path(cls, episode="episode_0000"):
        return (
            cls.flow / cls.scene / episode /
            "videos/chunk-000/lingbot_cache.npz"
        )

    @classmethod
    def _pins_for(cls, path=None, artifact=None):
        path = path or cls.scale_path
        artifact = artifact or cls.scale_artifact
        provenance = artifact["provenance"]
        estimator = provenance["estimator"]
        return {
            "causal_ground_scale_path": path,
            "expected_causal_ground_scale_sha256": sha256_bytes(
                path.read_bytes()),
            "expected_ground_scale_producer_sha256": provenance[
                "producer_source_sha256"],
            "expected_ground_scale_configuration_sha256": provenance[
                "configuration_sha256"],
            "expected_ground_scale_lingbot_commit": estimator[
                "lingbot_commit"],
            "expected_ground_scale_weights_sha256": estimator[
                "weights_sha256"],
            "expected_ground_scale_stream_source_sha256": estimator[
                "lingbot_stream_source_sha256"],
        }

    @classmethod
    def _build(cls, **kwargs):
        kwargs = {**cls._pins_for(), **kwargs}
        return build_artifact(
            manifest=cls.manifest,
            manifest_path=cls.manifest_path,
            manifest_sha256=cls.manifest_sha,
            depth_column_stride=4,
            scan_stride=4,
            **kwargs,
        )

    @classmethod
    def _write_variant(cls, mutate, *, canonical=True):
        artifact = copy.deepcopy(cls.scale_artifact)
        mutate(artifact)
        path = cls.root / f"variant_{len(list(cls.root.glob('variant_*')))}.json"
        if canonical:
            path.write_bytes(canonical_json_bytes(artifact))
        else:
            path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return artifact, path

    @classmethod
    def _load_bindings(cls, *, path=None, artifact=None, **pin_overrides):
        path = path or cls.scale_path
        artifact = artifact or cls.scale_artifact
        pins = cls._pins_for(path=path, artifact=artifact)
        loader_names = {
            "causal_ground_scale_path": "path",
            "expected_causal_ground_scale_sha256": "expected_artifact_sha256",
            "expected_ground_scale_producer_sha256": "expected_producer_sha256",
            "expected_ground_scale_configuration_sha256": (
                "expected_configuration_sha256"),
            "expected_ground_scale_lingbot_commit": "expected_lingbot_commit",
            "expected_ground_scale_weights_sha256": "expected_weights_sha256",
            "expected_ground_scale_stream_source_sha256": (
                "expected_stream_source_sha256"),
        }
        loader_pins = {
            loader_names[key]: value for key, value in pins.items()
        }
        loader_pins.update(pin_overrides)
        return load_external_ground_scale_bindings(
            **loader_pins,
            manifest=cls.manifest,
            manifest_sha256=cls.manifest_sha,
            allowed_scenes={cls.scene},
            legacy_builder_pin=None,
            legacy_configuration_pin=None,
        )[0]

    @classmethod
    def _direct_rows(cls, binding, *, episode="episode_0000", decision=None):
        decisions = [
            sample["decision_frame"] for sample in cls.manifest["samples"]
            if sample["source_episode"] == episode
        ]
        decision = decision or max(decisions)
        return lingbot_deployment_se2_rows(
            cls._aggregator_path(episode),
            cls._camera_path(episode),
            decision_frame=decision,
            episode_frame_count=48,
            camera_height_m=0.5,
            expected_causal_prefix_sha256="",
            expected_prefix_builder_sha256="",
            expected_prefix_configuration_sha256="",
            external_ground_scale=binding,
            episode_dir=cls.episodes / cls.scene / episode,
            episode_root=cls.episodes,
        )

    def test_external_sidecar_enables_cache_without_dense_prefix_and_v2(self):
        with np.load(self._camera_path(), allow_pickle=False) as cache:
            self.assertNotIn("ground_h_est_prefix", cache.files)
        artifact = self._build()
        self.assertEqual(
            artifact["schema_version"],
            "nlsr_v2_frontier_proposal_artifact_v2",
        )
        self.assertEqual(
            artifact["provenance"]["causal_ground_scale"]["mode"],
            "external_causal_first_prefix_v1",
        )
        rows = [
            row for row in artifact["records"]
            if row["source_episode"] == "episode_0000"
        ]
        self.assertTrue(rows)
        self.assertTrue(all(
            row["arms"][DEPLOYMENT_ARM]["proposal"]["valid"]
            for row in rows
        ))
        provenance = rows[0]["arms"][DEPLOYMENT_ARM]["pose_provenance"]
        self.assertEqual(
            provenance["metric_scale_source"],
            "external_causal_first_prefix_v1",
        )
        self.assertEqual(
            provenance["external_scale_artifact_sha256"],
            sha256_bytes(self.scale_path.read_bytes()),
        )
        self.assertFalse(provenance["future_cache_payload_hashed"])

    def test_explicit_invalid_external_never_falls_back_to_dense_prefix(self):
        legacy = build_artifact(
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha,
            expected_ground_prefix_builder_sha256="a" * 64,
            expected_ground_prefix_configuration_sha256="b" * 64,
            depth_column_stride=4,
            scan_stride=4,
        )
        dense_rows = [
            row for row in legacy["records"]
            if row["source_episode"] == "episode_0001"
        ]
        self.assertTrue(all(
            row["arms"][DEPLOYMENT_ARM]["proposal"]["valid"]
            for row in dense_rows
        ))

        def invalidate(artifact):
            record = next(
                row for row in artifact["records"]
                if row["episode"] == "episode_0001")
            record["valid"] = False
            record["invalid_reason"] = "unit_test_no_floor"
            record["ground_h_est_raw"] = None
            record["metric_scale_m_per_raw"] = None

        variant, path = self._write_variant(invalidate)
        artifact = self._build(**self._pins_for(path, variant))
        external_rows = [
            row for row in artifact["records"]
            if row["source_episode"] == "episode_0001"
        ]
        self.assertTrue(all(
            not row["arms"][DEPLOYMENT_ARM]["proposal"]["valid"]
            and "external causal ground scale is unavailable" in
            row["arms"][DEPLOYMENT_ARM]["proposal"]["invalid_reason"]
            for row in external_rows
        ))

    def test_external_arguments_are_all_or_none_and_pins_are_exact(self):
        with self.assertRaisesRegex(CandidateBuildError, "every exact pin"):
            load_external_ground_scale_bindings(
                path=self.scale_path,
                expected_artifact_sha256=None,
                expected_producer_sha256=None,
                expected_configuration_sha256=None,
                expected_lingbot_commit=None,
                expected_weights_sha256=None,
                expected_stream_source_sha256=None,
                manifest=self.manifest,
                manifest_sha256=self.manifest_sha,
                allowed_scenes={self.scene},
                legacy_builder_pin=None,
                legacy_configuration_pin=None,
            )
        wrong = {
            "expected_artifact_sha256": "f" * 64,
            "expected_producer_sha256": "f" * 64,
            "expected_configuration_sha256": "f" * 64,
            "expected_lingbot_commit": "f" * 40,
            "expected_weights_sha256": "f" * 64,
            "expected_stream_source_sha256": "f" * 64,
        }
        for field, value in wrong.items():
            with self.subTest(field=field):
                with self.assertRaises(CandidateBuildError):
                    self._load_bindings(**{field: value})
        with self.assertRaisesRegex(CandidateBuildError, "mutually exclusive"):
            pins = self._pins_for()
            load_external_ground_scale_bindings(
                path=pins["causal_ground_scale_path"],
                expected_artifact_sha256=pins[
                    "expected_causal_ground_scale_sha256"],
                expected_producer_sha256=pins[
                    "expected_ground_scale_producer_sha256"],
                expected_configuration_sha256=pins[
                    "expected_ground_scale_configuration_sha256"],
                expected_lingbot_commit=pins[
                    "expected_ground_scale_lingbot_commit"],
                expected_weights_sha256=pins[
                    "expected_ground_scale_weights_sha256"],
                expected_stream_source_sha256=pins[
                    "expected_ground_scale_stream_source_sha256"],
                manifest=self.manifest,
                manifest_sha256=self.manifest_sha,
                allowed_scenes={self.scene},
                legacy_builder_pin="a" * 64,
                legacy_configuration_pin=None,
            )

    def test_wrong_schema_and_noncanonical_json_fail_closed(self):
        schema_artifact, schema_path = self._write_variant(
            lambda artifact: artifact.__setitem__("schema_version", "wrong"))
        with self.assertRaisesRegex(CandidateBuildError, "schema"):
            self._load_bindings(path=schema_path, artifact=schema_artifact)
        pretty_artifact, pretty_path = self._write_variant(
            lambda artifact: None, canonical=False)
        with self.assertRaisesRegex(CandidateBuildError, "not canonical"):
            self._load_bindings(path=pretty_path, artifact=pretty_artifact)

    def test_prefix_rgb_pose_and_dtype_changes_invalidate_binding(self):
        binding = self._load_bindings()[(self.scene, "episode_0000")]
        rgb_path = (
            self.episodes / self.scene / "episode_0000" /
            "videos/chunk-000/observation.images.rgb/2.jpg"
        )
        original_rgb = rgb_path.read_bytes()
        try:
            Image.new("RGB", (32, 24), (255, 0, 0)).save(rgb_path)
            with self.assertRaisesRegex(DeploymentPoseError, "RGB prefix"):
                self._direct_rows(binding)
        finally:
            rgb_path.write_bytes(original_rgb)

        path = self._camera_path()
        original_cache = path.read_bytes()
        try:
            payload = self._camera_payload(path)
            pose = payload["cam_pose_enc"].copy()
            pose[2, 2] += 1.0
            payload["cam_pose_enc"] = pose
            np.savez(path, **payload)
            with self.assertRaisesRegex(DeploymentPoseError, "prefix changed"):
                self._direct_rows(binding)
        finally:
            path.write_bytes(original_cache)
        try:
            payload = self._camera_payload(path)
            payload["cam_pose_enc"] = payload["cam_pose_enc"].astype(np.float64)
            np.savez(path, **payload)
            with self.assertRaisesRegex(DeploymentPoseError, "dtype changed"):
                self._direct_rows(binding)
        finally:
            path.write_bytes(original_cache)

    @staticmethod
    def _camera_payload(path):
        with np.load(path, allow_pickle=False) as source:
            return {name: source[name] for name in source.files}

    def test_pose_causality_before_and_after_decision(self):
        binding = self._load_bindings()[(self.scene, "episode_0000")]
        decisions = [
            sample["decision_frame"] for sample in self.manifest["samples"]
            if sample["source_episode"] == "episode_0000"
        ]
        decision = max(decisions)
        prefix_end = int(binding.record["prefix_end_frame_exclusive"])
        baseline_rows, baseline_provenance = self._direct_rows(
            binding, decision=decision)
        path = self._camera_path()
        original_cache = path.read_bytes()
        try:
            payload = self._camera_payload(path)
            pose = payload["cam_pose_enc"].copy()
            pose[decision + 1, 2] += 10.0
            payload["cam_pose_enc"] = pose
            np.savez(path, **payload)
            rows, provenance = self._direct_rows(binding, decision=decision)
            self.assertEqual(rows, baseline_rows)
            self.assertEqual(provenance, baseline_provenance)
        finally:
            path.write_bytes(original_cache)

        try:
            payload = self._camera_payload(path)
            pose = payload["cam_pose_enc"].copy()
            changed_frame = prefix_end + 1
            self.assertLess(changed_frame, decision)
            pose[changed_frame, 2] += 1.0
            payload["cam_pose_enc"] = pose
            np.savez(path, **payload)
            rows, _ = self._direct_rows(binding, decision=decision)
            self.assertNotEqual(rows[changed_frame], baseline_rows[changed_frame])
        finally:
            path.write_bytes(original_cache)

    def test_formula_camera_height_and_cache_generation_are_rechecked(self):
        mutations = {
            "formula": lambda record: record.__setitem__(
                "metric_scale_m_per_raw",
                float(record["metric_scale_m_per_raw"]) + 0.5),
            "camera height": lambda record: record.__setitem__(
                "camera_height_m", 0.6),
            "cache generation": lambda record: record.__setitem__(
                "precompute_signature", "different_generation"),
        }
        expected = {
            "formula": "pinned formula",
            "camera height": "camera height changed",
            "cache generation": "different cache generation",
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                def mutate(artifact, mutation=mutation):
                    record = next(
                        row for row in artifact["records"]
                        if row["episode"] == "episode_0000")
                    mutation(record)
                variant, path = self._write_variant(mutate)
                binding = self._load_bindings(
                    path=path, artifact=variant
                )[(self.scene, "episode_0000")]
                with self.assertRaisesRegex(
                        DeploymentPoseError, expected[label]):
                    self._direct_rows(binding)

    def test_valid_false_is_an_invalid_deployment_arm(self):
        def invalidate(artifact):
            record = next(
                row for row in artifact["records"]
                if row["episode"] == "episode_0000")
            record["valid"] = False
            record["invalid_reason"] = "no_floor"
            record["ground_h_est_raw"] = None
            record["metric_scale_m_per_raw"] = None

        variant, path = self._write_variant(invalidate)
        artifact = self._build(**self._pins_for(path, variant))
        rows = [
            row for row in artifact["records"]
            if row["source_episode"] == "episode_0000"
        ]
        self.assertTrue(all(
            not row["arms"][DEPLOYMENT_ARM]["proposal"]["valid"]
            for row in rows
        ))


if __name__ == "__main__":
    unittest.main()
