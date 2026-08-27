import copy
import tempfile
import unittest
from pathlib import Path

from MemNavData.flow_cache_routing import (
    FLOW_FILE_NAMES,
    FlowRoutingError,
    ROUTE_SCHEMA_VERSION,
    ROUTE_STATUS,
    canonical_json_bytes,
    load_route_registry,
    registry_from_manifest,
    sha256_bytes,
    sha256_file,
)


class FlowCacheRoutingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.official = self.root / "official"
        self.patch = self.root / "patch"
        self.route_root = self.root / "routes"
        for root in (self.official, self.patch, self.route_root):
            root.mkdir()
        self.split_sha = "1" * 64
        self.raw_sha = "2" * 64
        self.episodes = (
            ("scene_a/episode_0000", "official_base", self.official),
            ("scene_b/episode_0000", "flow4096_patch", self.patch),
        )
        self.rows = []
        for episode, source_id, root in self.episodes:
            chunk_relative = f"{episode}/videos/chunk-000"
            chunk = root / chunk_relative
            chunk.mkdir(parents=True)
            files = []
            for name in FLOW_FILE_NAMES:
                path = chunk / name
                path.write_bytes(f"{source_id}:{episode}:{name}".encode())
                files.append({
                    "name": name,
                    "bytes": path.stat().st_size,
                    "content_sha256": (
                        sha256_file(path)
                        if source_id == "flow4096_patch" else None
                    ),
                })
            self.rows.append({
                "episode": episode,
                "source_id": source_id,
                "source_relative_chunk": chunk_relative,
                "validation": {"files": files},
            })
        self.record = {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "status": ROUTE_STATUS,
            "split_sha256": self.split_sha,
            "raw_audit_sha256": self.raw_sha,
            "route_root": str(self.route_root.resolve()),
            "source_roots": {
                "official_base": str(self.official.resolve()),
                "flow4096_patch": str(self.patch.resolve()),
            },
            "official_snapshot_semantics": "fixture metadata pin",
            "official_snapshot_sha256": "3" * 64,
            "patch_payloads_fully_sha256": True,
            "counts": {
                "scenes": 2,
                "pairs": 2,
                "official_base": 1,
                "flow4096_patch": 1,
            },
            "pairs": self.rows,
        }
        self.artifact = self.route_root / "FLOW_ROUTE_PROVENANCE.json"
        self.digest = self._write(self.record)

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, record, *, canonical=True):
        if canonical:
            payload = canonical_json_bytes(record)
        else:
            payload = ("{\n  \"status\": \"flow_routes_audited\"\n}\n").encode()
        digest = sha256_bytes(payload)
        self.artifact.write_bytes(payload)
        Path(f"{self.artifact}.sha256").write_text(
            f"{digest}  {self.artifact.name}\n", encoding="ascii")
        return digest

    def _manifest(self, registry):
        return {
            "schema_version": "nlsr_v2_expert_candidate_manifest_v2",
            "split": {"sha256": self.split_sha},
            "flow_cache_routing": registry.manifest_record(),
        }

    def test_zero_copy_registry_freezes_and_resolves_an_atomic_pair(self):
        registry = load_route_registry(self.artifact, self.digest)
        records = registry.episode_file_records("scene_a", "episode_0000")
        episode_record = {"flow_cache": {"complete": True, "files": records}}
        aggregator, camera = registry.resolve_manifest_pair(
            episode_record, "scene_a", "episode_0000")
        self.assertEqual(aggregator, (
            self.official / "scene_a/episode_0000/videos/chunk-000/"
            "lingbot_cache.npz").resolve())
        self.assertEqual(camera, (
            self.official / "scene_a/episode_0000/videos/chunk-000/"
            "lingbot_cam_cache.npz").resolve())
        self.assertFalse(aggregator.is_symlink())
        self.assertEqual(
            registry_from_manifest(self._manifest(registry)).artifact_sha256,
            self.digest,
        )

    def test_wrong_sha_sidecar_and_noncanonical_json_fail_closed(self):
        with self.assertRaisesRegex(FlowRoutingError, "SHA256 mismatch"):
            load_route_registry(self.artifact, "f" * 64)
        sidecar = Path(f"{self.artifact}.sha256")
        sidecar.write_text(f"{self.digest} malformed\n", encoding="ascii")
        with self.assertRaisesRegex(FlowRoutingError, "sidecar"):
            load_route_registry(self.artifact, self.digest)

        self.digest = self._write(self.record)
        parsed_payload = (
            __import__("json").dumps(self.record, indent=2, sort_keys=True)
            + "\n"
        ).encode()
        self.artifact.write_bytes(parsed_payload)
        noncanonical_sha = sha256_bytes(parsed_payload)
        sidecar.write_text(
            f"{noncanonical_sha}  {self.artifact.name}\n", encoding="ascii")
        with self.assertRaisesRegex(FlowRoutingError, "not canonical"):
            load_route_registry(self.artifact, noncanonical_sha)

    def test_route_artifact_itself_cannot_be_a_symlink(self):
        physical = self.route_root / "physical_route.json"
        self.artifact.replace(physical)
        self.artifact.symlink_to(physical)
        with self.assertRaisesRegex(FlowRoutingError, "not physical"):
            load_route_registry(self.artifact, self.digest)

    def test_traversal_duplicate_and_unknown_source_are_rejected(self):
        traversal = copy.deepcopy(self.record)
        traversal["pairs"][0]["source_relative_chunk"] = "../escape"
        digest = self._write(traversal)
        with self.assertRaisesRegex(FlowRoutingError, "root-relative"):
            load_route_registry(self.artifact, digest)

        duplicate = copy.deepcopy(self.record)
        duplicate["pairs"][1]["episode"] = duplicate["pairs"][0]["episode"]
        duplicate["pairs"][1]["source_relative_chunk"] = (
            duplicate["pairs"][0]["source_relative_chunk"])
        digest = self._write(duplicate)
        with self.assertRaisesRegex(FlowRoutingError, "duplicate route episode"):
            load_route_registry(self.artifact, digest)

        unknown = copy.deepcopy(self.record)
        unknown["pairs"][0]["source_id"] = "untrusted"
        digest = self._write(unknown)
        with self.assertRaisesRegex(FlowRoutingError, "unknown route source"):
            load_route_registry(self.artifact, digest)

    def test_symlink_escape_and_patch_payload_mutation_are_rejected(self):
        official_file = (
            self.official / "scene_a/episode_0000/videos/chunk-000/"
            "lingbot_cache.npz")
        external = self.root / "external.npz"
        external.write_bytes(official_file.read_bytes())
        official_file.unlink()
        official_file.symlink_to(external)
        with self.assertRaisesRegex(FlowRoutingError, "escapes|physical"):
            load_route_registry(self.artifact, self.digest)

        official_file.unlink()
        official_file.write_bytes(external.read_bytes())
        patch_file = (
            self.patch / "scene_b/episode_0000/videos/chunk-000/"
            "lingbot_cam_cache.npz")
        original_size = patch_file.stat().st_size
        patch_file.write_bytes(b"x" * original_size)
        with self.assertRaisesRegex(FlowRoutingError, "content SHA256 differs"):
            load_route_registry(self.artifact, self.digest)

    def test_manifest_binding_split_and_file_mutation_are_rejected(self):
        registry = load_route_registry(self.artifact, self.digest)
        manifest = self._manifest(registry)
        manifest["split"] = {"sha256": "9" * 64}
        with self.assertRaisesRegex(FlowRoutingError, "split differs"):
            registry_from_manifest(manifest)

        manifest = self._manifest(registry)
        manifest["flow_cache_routing"]["source_roots"]["official_base"] = (
            str(self.patch.resolve()))
        with self.assertRaisesRegex(FlowRoutingError, "differs"):
            registry_from_manifest(manifest)

        records = registry.episode_file_records("scene_a", "episode_0000")
        records[0]["bytes"] += 1
        episode_record = {"flow_cache": {"complete": True, "files": records}}
        with self.assertRaisesRegex(FlowRoutingError, "binding differs"):
            registry.resolve_manifest_pair(
                episode_record, "scene_a", "episode_0000")


if __name__ == "__main__":
    unittest.main()
