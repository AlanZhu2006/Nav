import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from MemNavData.build_multistage_candidate_manifest import (
    EXPECTED_AUDIT_SCHEMA,
    EXPECTED_AUDIT_STATUS,
    GOAL_C_STATE_NAME,
    SCHEMA_VERSION,
    MultistageManifestError,
    build_multistage_manifest,
    load_pinned_canonical_artifact,
)
from MemNavData.build_novel_candidate_manifest import (
    build_manifest as build_source_manifest,
    canonical_json_bytes,
    sha256_bytes,
    write_artifact,
)
from MemNavData.flow_cache_routing import ROUTE_SCHEMA_VERSION, ROUTE_STATUS


class MultistageCandidateManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episodes = self.root / "episodes"
        self.flow = self.root / "flow"
        self.routes = self.root / "routes"
        self.environments = self.root / "environments"
        self.navmeshes = self.root / "navmeshes"
        for root in (
                self.episodes, self.flow, self.routes,
                self.environments, self.navmeshes):
            root.mkdir()
        self.scene = "scene_train"
        self.split_path = self.root / "split.json"
        self.split = {
            "version": "fixture_multistage_v1",
            "train": [self.scene],
            "development": [],
            "final_reserved": ["scene_final"],
        }
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")
        (self.environments / f"{self.scene}.glb").write_bytes(b"environment")
        (self.navmeshes / f"{self.scene}.navmesh").write_bytes(b"navmesh")
        for number in range(2):
            episode_name = f"episode_{number:04d}"
            self._episode(episode_name)
            self._flow_pair(episode_name)
        self.route_path, self.route_sha = self._route_artifact()

    def tearDown(self):
        self.temporary.cleanup()

    def _episode(
        self,
        name: str,
        *,
        camera_height_m: float | None = None,
    ) -> Path:
        episode = self.episodes / self.scene / name
        (episode / "meta").mkdir(parents=True, exist_ok=True)
        (episode / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        rgb = episode / "videos/chunk-000/observation.images.rgb"
        depth = episode / "videos/chunk-000/observation.images.depth"
        rgb.mkdir(parents=True, exist_ok=True)
        depth.mkdir(parents=True, exist_ok=True)
        metadata = {
            "scene": f"{self.scene}.glb",
            "n_frames": 48,
            "n_legs": 3,
            "switches": [8, 40],
            "goals": [
                {"kind": "novel", "pos": [1.0, 0.0, 1.0]},
                {"kind": "revisit", "pos": [0.0, 0.0, 0.0]},
            ],
        }
        if camera_height_m is not None:
            metadata["camera_height_m"] = camera_height_m
        (episode / "meta/gen_meta.json").write_text(
            json.dumps(metadata), encoding="utf-8")
        (episode / "goal_1.jpg").write_bytes(f"goal-b:{name}".encode())
        (episode / "goal_2.jpg").write_bytes(f"goal-c:{name}".encode())
        for frame in range(48):
            (rgb / f"{frame}.jpg").write_bytes(f"rgb:{name}:{frame}".encode())
            (depth / f"{frame}.png").write_bytes(
                f"depth:{name}:{frame}".encode())
        self._write_parquet(episode)
        return episode

    @staticmethod
    def _write_parquet(
        episode: Path,
        *,
        changed_row: int | None = None,
        delta: float = 0.0,
    ) -> None:
        intrinsic = [
            [355.0, 0.0, 240.0],
            [0.0, 351.0, 135.0],
            [0.0, 0.0, 1.0],
        ]
        extrinsic = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
        rows = []
        for frame in range(48):
            action = [
                [1.0, 0.0, 0.0, float(frame)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            if frame == changed_row:
                action[0][3] += delta
            rows.append({
                "index": frame,
                "observation.camera_intrinsic": intrinsic,
                "observation.camera_extrinsic": extrinsic,
                "action": action,
            })
        pq.write_table(
            pa.Table.from_pylist(rows),
            episode / "data/chunk-000/episode_000000.parquet",
        )

    def _flow_pair(self, episode: str) -> None:
        chunk = self.flow / self.scene / episode / "videos/chunk-000"
        chunk.mkdir(parents=True)
        (chunk / "lingbot_cache.npz").write_bytes(b"aggregate")
        (chunk / "lingbot_cam_cache.npz").write_bytes(b"camera")

    def _route_artifact(self) -> tuple[Path, str]:
        pairs = []
        for episode in ("episode_0000", "episode_0001"):
            relative_chunk = f"{self.scene}/{episode}/videos/chunk-000"
            files = []
            for name in ("lingbot_cache.npz", "lingbot_cam_cache.npz"):
                path = self.flow / relative_chunk / name
                files.append({
                    "name": name,
                    "bytes": path.stat().st_size,
                    "content_sha256": None,
                })
            pairs.append({
                "episode": f"{self.scene}/{episode}",
                "source_id": "official_base",
                "source_relative_chunk": relative_chunk,
                "validation": {"files": files},
            })
        route = {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "status": ROUTE_STATUS,
            "split_sha256": sha256_bytes(self.split_path.read_bytes()),
            "raw_audit_sha256": "4" * 64,
            "route_root": str(self.routes.resolve()),
            "source_roots": {"official_base": str(self.flow.resolve())},
            "official_snapshot_semantics": "unit-test snapshot",
            "official_snapshot_sha256": "5" * 64,
            "patch_payloads_fully_sha256": True,
            "counts": {"scenes": 1, "pairs": 2, "official_base": 2},
            "pairs": pairs,
        }
        path = self.routes / "flow_routes.json"
        payload = canonical_json_bytes(route)
        path.write_bytes(payload)
        digest = sha256_bytes(payload)
        Path(f"{path}.sha256").write_text(
            f"{digest}  {path.name}\n", encoding="ascii")
        return path, digest

    def _source(self) -> tuple[dict, Path, str]:
        source = build_source_manifest(
            split_path=self.split_path,
            episode_root=self.episodes,
            flow_cache_root=None,
            flow_route_provenance=self.route_path,
            expected_flow_route_sha256=self.route_sha,
            environment_root=self.environments,
            navmesh_root=self.navmeshes,
            roles=("train",),
        )
        path = self.root / "routed_manifest.json"
        payload = canonical_json_bytes(source)
        path.write_bytes(payload)
        digest = sha256_bytes(payload)
        return source, path, digest

    def _audit(self, source: dict, source_sha: str) -> tuple[dict, Path, str]:
        audit = {
            "schema_version": EXPECTED_AUDIT_SCHEMA,
            "status": EXPECTED_AUDIT_STATUS,
            "downstream_ready": True,
            "manifest_sha256": source_sha,
            "split_sha256": source["split"]["sha256"],
            "summary": source["summary"],
        }
        path = self.root / "routed_manifest_audit.json"
        payload = canonical_json_bytes(audit)
        path.write_bytes(payload)
        digest = sha256_bytes(payload)
        return audit, path, digest

    def _build(
        self,
        *,
        legacy_camera_height_m: float | None = 0.5,
    ) -> tuple[dict, dict]:
        source, source_path, source_sha = self._source()
        audit, audit_path, audit_sha = self._audit(source, source_sha)
        output = build_multistage_manifest(
            routed_manifest=source,
            routed_manifest_path=source_path,
            routed_manifest_sha256=source_sha,
            routed_manifest_audit=audit,
            routed_manifest_audit_path=audit_path,
            routed_manifest_audit_sha256=audit_sha,
            legacy_camera_height_m=legacy_camera_height_m,
        )
        return source, output

    def test_goal_b_rows_are_byte_preserved_and_goal_c_rows_are_causal(self):
        source, output = self._build()
        self.assertEqual(output["schema_version"], SCHEMA_VERSION)
        source_b = source["samples"]
        output_b = output["samples"][:len(source_b)]
        self.assertEqual(
            [canonical_json_bytes(row) for row in output_b],
            [canonical_json_bytes(row) for row in source_b],
        )
        self.assertEqual(output["summary"]["goal_b_sample_count"], 8)
        self.assertEqual(output["summary"]["goal_c_sample_count"], 4)
        self.assertEqual(output["summary"]["sample_count"], 12)

        c_rows = output["samples"][8:]
        self.assertEqual({row["goal_role"] for row in c_rows}, {"C"})
        self.assertEqual({row["state_name"] for row in c_rows}, {
            GOAL_C_STATE_NAME})
        self.assertEqual({row["decision_frame"] for row in c_rows}, {40})
        for row in c_rows:
            self.assertEqual(row["causal_prefix"]["frame_count"], 40)
            self.assertEqual(row["causal_prefix"]["parquet_row_count"], 40)
            self.assertEqual(row["navdp_fifo"]["current_frame_index"], 39)
            self.assertEqual(
                row["navdp_fifo"]["replay_frame_indices"],
                [7, 15, 23, 31],
            )
            self.assertEqual(
                row["navdp_fifo"]["after_append_frame_indices"],
                [7, 15, 23, 31, 39],
            )
            self.assertTrue(row["goal"]["path"].endswith("goal_2.jpg"))
            self.assertTrue(row["state_frame"]["path"].endswith("/39.jpg"))
        by_source = {}
        for row in c_rows:
            by_source.setdefault(row["source_episode"], {})[
                row["goal_variant"]] = row
        for source_episode, variants in by_source.items():
            self.assertEqual(set(variants), {"factual", "counterfactual"})
            self.assertEqual(variants["factual"]["goal_episode"], source_episode)
            self.assertNotEqual(
                variants["counterfactual"]["goal_episode"], source_episode)
            self.assertEqual(
                variants["factual"]["causal_prefix"],
                variants["counterfactual"]["causal_prefix"],
            )
            self.assertEqual(
                variants["factual"]["navdp_fifo"],
                variants["counterfactual"]["navdp_fifo"],
            )

    def test_groups_are_split_scene_episode_and_state_qualified(self):
        _, output = self._build()
        bindings = output["sample_group_bindings"]
        self.assertEqual(len(bindings), len(output["samples"]))
        pairs = {}
        for binding in bindings:
            self.assertEqual(binding["split_role"], "train")
            self.assertEqual(binding["scene"], self.scene)
            self.assertTrue(binding["episode_group_id"].startswith(
                f"train/{self.scene}/"))
            pairs.setdefault(
                binding["counterfactual_pair_group_id"], []).append(
                    binding["sample_id"])
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(len(members) == 2 for members in pairs.values()))

    def test_legacy_camera_height_is_explicit_bound_and_has_no_default(self):
        with self.assertRaisesRegex(
                MultistageManifestError, "--legacy-camera-height-m explicitly"):
            self._build(legacy_camera_height_m=None)
        _, output = self._build(legacy_camera_height_m=0.5)
        policy = output["configuration"]["camera_height_policy"]
        self.assertEqual(policy["legacy_camera_height_m"], 0.5)
        self.assertEqual(
            policy["legacy_value_source"],
            "explicit_cli:--legacy-camera-height-m",
        )
        bindings = output["provenance"]["camera_height_bindings"]
        self.assertEqual(len(bindings), 2)
        self.assertEqual({row["camera_height_m"] for row in bindings}, {0.5})
        self.assertEqual(
            {row["value_source"] for row in bindings},
            {"explicit_cli:--legacy-camera-height-m"},
        )
        self.assertEqual(
            output["summary"]["legacy_camera_height_episode_count"], 2)

    def test_metadata_height_is_authoritative_and_conflict_is_rejected(self):
        for number in range(2):
            self._episode(f"episode_{number:04d}", camera_height_m=0.6)
        with self.assertRaisesRegex(
                MultistageManifestError, "conflicts with the explicit legacy"):
            self._build(legacy_camera_height_m=0.5)
        _, output = self._build(legacy_camera_height_m=None)
        self.assertEqual(
            {row["camera_height_m"] for row in
             output["provenance"]["camera_height_bindings"]},
            {0.6},
        )
        self.assertEqual(
            output["summary"]["legacy_camera_height_episode_count"], 0)

    def test_present_but_invalid_metadata_height_cannot_use_legacy_fallback(self):
        metadata_path = (
            self.episodes / self.scene /
            "episode_0000/meta/gen_meta.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["camera_height_m"] = None
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(
                MultistageManifestError, "metadata.camera_height_m must be numeric"):
            self._build(legacy_camera_height_m=0.5)

    def test_future_change_is_not_in_c_prefix_but_prefix_change_is(self):
        _, initial = self._build()
        initial_c = {
            (row["source_episode"], row["goal_variant"]):
                row["causal_prefix"]["causal_prefix_sha256"]
            for row in initial["samples"]
            if row["goal_role"] == "C"
        }
        episode = self.episodes / self.scene / "episode_0000"
        self._write_parquet(episode, changed_row=44, delta=0.75)
        _, future_changed = self._build()
        future_c = {
            (row["source_episode"], row["goal_variant"]):
                row["causal_prefix"]["causal_prefix_sha256"]
            for row in future_changed["samples"]
            if row["goal_role"] == "C"
        }
        self.assertEqual(initial_c, future_c)

        self._write_parquet(episode, changed_row=30, delta=0.5)
        _, prefix_changed = self._build()
        prefix_c = {
            (row["source_episode"], row["goal_variant"]):
                row["causal_prefix"]["causal_prefix_sha256"]
            for row in prefix_changed["samples"]
            if row["goal_role"] == "C"
        }
        for variant in ("factual", "counterfactual"):
            self.assertNotEqual(
                future_c[("episode_0000", variant)],
                prefix_c[("episode_0000", variant)],
            )
            self.assertEqual(
                future_c[("episode_0001", variant)],
                prefix_c[("episode_0001", variant)],
            )

    def test_tampered_goal_b_source_row_is_rejected_not_normalized(self):
        source, source_path, source_sha = self._source()
        source["samples"][0]["decision_frame"] += 1
        source_payload = canonical_json_bytes(source)
        source_path.write_bytes(source_payload)
        source_sha = sha256_bytes(source_payload)
        audit, audit_path, audit_sha = self._audit(source, source_sha)
        with self.assertRaisesRegex(
                MultistageManifestError, "Goal-B sample 0 differs"):
            build_multistage_manifest(
                routed_manifest=source,
                routed_manifest_path=source_path,
                routed_manifest_sha256=source_sha,
                routed_manifest_audit=audit,
                routed_manifest_audit_path=audit_path,
                routed_manifest_audit_sha256=audit_sha,
                legacy_camera_height_m=0.5,
            )

    def test_goal_c_must_be_metadata_revisit(self):
        source, source_path, source_sha = self._source()
        audit, audit_path, audit_sha = self._audit(source, source_sha)
        metadata_path = (
            self.episodes / self.scene /
            "episode_0000/meta/gen_meta.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["goals"][1]["kind"] = "novel"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(
                MultistageManifestError, "source episode no longer validates"):
            build_multistage_manifest(
                routed_manifest=source,
                routed_manifest_path=source_path,
                routed_manifest_sha256=source_sha,
                routed_manifest_audit=audit,
                routed_manifest_audit_path=audit_path,
                routed_manifest_audit_sha256=audit_sha,
                legacy_camera_height_m=0.5,
            )

    def test_audit_must_bind_exact_manifest_split_and_summary(self):
        source, source_path, source_sha = self._source()
        audit, audit_path, audit_sha = self._audit(source, source_sha)
        audit["manifest_sha256"] = "a" * 64
        audit_payload = canonical_json_bytes(audit)
        audit_path.write_bytes(audit_payload)
        audit_sha = sha256_bytes(audit_payload)
        with self.assertRaisesRegex(
                MultistageManifestError, "does not bind the exact"):
            build_multistage_manifest(
                routed_manifest=source,
                routed_manifest_path=source_path,
                routed_manifest_sha256=source_sha,
                routed_manifest_audit=audit,
                routed_manifest_audit_path=audit_path,
                routed_manifest_audit_sha256=audit_sha,
                legacy_camera_height_m=0.5,
            )

    def test_pinned_loader_and_resume_reject_noncanonical_or_drift(self):
        source, source_path, source_sha = self._source()
        loaded, digest = load_pinned_canonical_artifact(
            source_path, source_sha, label="fixture")
        self.assertEqual((loaded, digest), (source, source_sha))
        with self.assertRaisesRegex(MultistageManifestError, "SHA256 mismatch"):
            load_pinned_canonical_artifact(
                source_path, "a" * 64, label="fixture")

        pretty_path = self.root / "pretty.json"
        pretty_payload = json.dumps(source, indent=2).encode("utf-8")
        pretty_path.write_bytes(pretty_payload)
        with self.assertRaisesRegex(MultistageManifestError, "not canonical"):
            load_pinned_canonical_artifact(
                pretty_path,
                sha256_bytes(pretty_payload),
                label="pretty fixture",
            )

        _, output = self._build()
        out_path = self.root / "multistage.json"
        sha_path = self.root / "multistage.json.sha256"
        status, output_sha = write_artifact(output, out_path, sha_path)
        self.assertEqual(status, "written")
        self.assertEqual(
            write_artifact(output, out_path, sha_path, resume=True),
            ("resumed", output_sha),
        )
        drifted = dict(output)
        drifted["purpose"] = "drifted"
        with self.assertRaisesRegex(Exception, "differs"):
            write_artifact(drifted, out_path, sha_path, resume=True)

    def test_feature_boundary_contains_no_gt_or_future_feature(self):
        _, output = self._build()
        boundary = output["configuration"]["feature_boundary"]
        self.assertFalse(boundary["future_observation_as_feature"])
        self.assertFalse(boundary["future_parquet_as_feature"])
        self.assertFalse(boundary["goal_pose_as_feature"])
        self.assertFalse(boundary["geodesic_as_feature"])
        forbidden = {"goal_pos", "goal_yaw", "geodesic", "label", "success"}
        for sample in output["samples"]:
            self.assertTrue(forbidden.isdisjoint(sample))


if __name__ == "__main__":
    unittest.main()
