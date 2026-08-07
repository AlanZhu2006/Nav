"""Adversarial contract tests for the provenance-backed NavMesh bake stage.

These tests intentionally use a fake runtime.  They prove orchestration,
content pinning, double-bake admission, publication, and resume behavior; they
do not claim to test Habitat-Sim's C++ Recast implementation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from MemNavData import bake_pinned_navmeshes as baker
from MemNavData.bake_pinned_navmeshes import (
    BAKE_INDEX_FILE,
    BAKE_STATUS,
    DERIVED_MANIFEST_FILE,
    EXPECTED_HABITAT_VERSION,
    PUBLISHED_DIRECTORY,
    SCENES_DIRECTORY,
    BakeResult,
    NavmeshBakeError,
    NavmeshObservation,
    bake_geometry_bundle,
)
from MemNavData.build_frozen_geometry_map import (
    MAP_FILENAME,
    build_geometry_map_bundle,
    canonical_json_bytes,
    load_pinned_manifest,
    sha256_bytes,
)
from MemNavData.habitat_rollout_primitives import NAVMESH_SETTING_FIELDS


BINDINGS_SHA256 = "b" * 64
INIT_SHA256 = "a" * 64


def navmesh_settings() -> dict[str, object]:
    values: dict[str, object] = {
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


def file_record(path: Path, root: Path) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    payload = path.read_bytes()
    return {
        "path": relative,
        "path_sha256": sha256_bytes(relative.encode("utf-8")),
        "bytes": len(payload),
        "content_sha256": sha256_bytes(payload),
    }


def runtime_identity(
    *,
    version: str = EXPECTED_HABITAT_VERSION,
    bindings_sha256: str = BINDINGS_SHA256,
) -> dict[str, object]:
    files = {
        "habitat_sim_init": {
            "name": "__init__.py",
            "bytes": 101,
            "content_sha256": INIT_SHA256,
        },
        "habitat_sim_bindings": {
            "name": "habitat_sim_bindings.so",
            "bytes": 202,
            "content_sha256": bindings_sha256,
        },
    }
    fingerprint_input = {
        "habitat_sim_version": version,
        "python_version": "3.9.19",
        "runtime_files": files,
    }
    return {
        **fingerprint_input,
        "runtime_fingerprint_sha256": sha256_bytes(
            canonical_json_bytes(fingerprint_input)
        ),
    }


class FakeBakeRuntime:
    def __init__(
        self,
        *,
        nondeterministic_scene: str | None = None,
        fail_scene: str | None = None,
        mutate_scene: str | None = None,
        identity: dict[str, object] | None = None,
    ) -> None:
        self._identity = runtime_identity() if identity is None else identity
        self.nondeterministic_scene = nondeterministic_scene
        self.fail_scene = fail_scene
        self.mutate_scene = mutate_scene
        self.bake_calls: list[str] = []
        self.validate_calls: list[Path] = []
        self._per_scene_calls: dict[str, int] = {}

    @property
    def identity(self) -> dict[str, object]:
        return copy.deepcopy(self._identity)

    def assert_settings_contract(
        self, requested: dict[str, object]
    ) -> dict[str, object]:
        return copy.deepcopy(requested)

    @staticmethod
    def _observation(glb_or_navmesh: Path) -> NavmeshObservation:
        offset = float(len(glb_or_navmesh.name)) / 100.0
        return NavmeshObservation(
            navigable_area_m2=10.0 + offset,
            bounds_min_xyz=(-1.0, 0.0, -2.0),
            bounds_max_xyz=(3.0, 2.0, 4.0),
            vertex_count=12,
            index_count=36,
        )

    @classmethod
    def _serialized_observation(cls, glb_or_navmesh: Path) -> NavmeshObservation:
        observation = cls._observation(glb_or_navmesh)
        return NavmeshObservation(
            navigable_area_m2=observation.navigable_area_m2,
            bounds_min_xyz=observation.bounds_min_xyz,
            bounds_max_xyz=(
                observation.bounds_max_xyz[0],
                observation.bounds_max_xyz[1] + 1.6,
                observation.bounds_max_xyz[2],
            ),
            vertex_count=observation.vertex_count,
            index_count=observation.index_count,
        )

    def bake_once(
        self,
        glb_path: Path,
        navmesh_output: Path,
        requested_settings: dict[str, object],
    ) -> BakeResult:
        scene = glb_path.stem
        self.bake_calls.append(scene)
        self._per_scene_calls[scene] = self._per_scene_calls.get(scene, 0) + 1
        if scene == self.fail_scene:
            raise RuntimeError(f"injected bake failure for {scene}")
        payload = b"FAKE_NAVMESH_V1\x00" + sha256_bytes(
            glb_path.read_bytes() + canonical_json_bytes(requested_settings)
        ).encode("ascii")
        if scene == self.nondeterministic_scene:
            payload += f"::{self._per_scene_calls[scene]}".encode("ascii")
        navmesh_output.write_bytes(payload)
        observation = self._observation(glb_path)
        if (
            scene == self.mutate_scene
            and self._per_scene_calls[scene] == baker.REPETITIONS
        ):
            glb_path.write_bytes(glb_path.read_bytes() + b"DRIFT")
        return BakeResult(
            copy.deepcopy(requested_settings),
            observation,
            self._serialized_observation(glb_path),
        )

    def validate_output(
        self,
        navmesh_path: Path,
        requested_settings: dict[str, object],
    ) -> BakeResult:
        self.validate_calls.append(navmesh_path)
        if not navmesh_path.read_bytes().startswith(b"FAKE_NAVMESH_V1\x00"):
            raise NavmeshBakeError("fake runtime rejected navmesh bytes")
        observation = self._serialized_observation(Path("scene-a.glb"))
        return BakeResult(copy.deepcopy(requested_settings), observation, observation)


class NavmeshBakeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environments = self.root / "environments"
        self.old_navmeshes = self.root / "old-navmeshes"
        self.environments.mkdir()
        self.old_navmeshes.mkdir()
        self.scene_files: dict[str, tuple[Path, Path]] = {}
        for scene in ("scene-a", "scene-b"):
            glb = self.environments / f"{scene}.glb"
            old_navmesh = self.old_navmeshes / f"{scene}.navmesh"
            glb.write_bytes(f"PINNED_GLB::{scene}\x00".encode())
            old_navmesh.write_bytes(f"UNTRUSTED_OLD_NAV::{scene}\x00".encode())
            self.scene_files[scene] = (glb, old_navmesh)
        self.manifest_path = self.root / "expert-manifest.json"
        self.settings_path = self.root / "navmesh-settings.json"
        self.output = self.root / "baked-geometry"
        self.manifest = self._manifest_for(("scene-a", "scene-b"))
        self._write_manifest(self.manifest)
        self._write_settings(navmesh_settings())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest_for(self, scenes: tuple[str, ...]) -> dict[str, object]:
        rows = []
        for index, scene in enumerate(scenes):
            glb, old_navmesh = self.scene_files[scene]
            rows.append(
                {
                    "scene": scene,
                    "split_role": "train" if index == 0 else "development",
                    "environment": file_record(glb, self.environments),
                    "navmesh": file_record(old_navmesh, self.old_navmeshes),
                    "selected_episodes": [],
                }
            )
        return {
            "schema_version": "nlsr_v2_expert_candidate_manifest_v2",
            "input_roots": {
                "episode_root": str(self.root / "unused-episodes"),
                "environment_root": str(self.environments),
                "navmesh_root": str(self.old_navmeshes),
            },
            "scenes": rows,
            "samples": [],
            "summary": {"scene_count": len(rows)},
        }

    def _write_manifest(self, value: object) -> str:
        payload = canonical_json_bytes(value)
        self.manifest_path.write_bytes(payload)
        self.manifest_sha = sha256_bytes(payload)
        return self.manifest_sha

    def _write_settings(self, value: object) -> str:
        payload = canonical_json_bytes(value)
        self.settings_path.write_bytes(payload)
        self.settings_sha = sha256_bytes(payload)
        return self.settings_sha

    def _bake(
        self,
        runtime: FakeBakeRuntime | None = None,
        **overrides: object,
    ) -> dict[str, object]:
        arguments: dict[str, object] = {
            "manifest_path": self.manifest_path,
            "expected_manifest_sha256": self.manifest_sha,
            "settings_path": self.settings_path,
            "expected_settings_sha256": self.settings_sha,
            "expected_habitat_version": EXPECTED_HABITAT_VERSION,
            "expected_habitat_bindings_sha256": BINDINGS_SHA256,
            "output_root": self.output,
            "runtime": FakeBakeRuntime() if runtime is None else runtime,
        }
        arguments.update(overrides)
        return bake_geometry_bundle(**arguments)  # type: ignore[arg-type]

    def test_build_publishes_receipts_and_derived_manifest_is_map_compatible(
        self,
    ) -> None:
        runtime = FakeBakeRuntime()
        result = self._bake(runtime)
        self.assertEqual(result["status"], "written")
        self.assertEqual(result["baked_scene_count"], 2)
        self.assertEqual(result["resumed_scene_count"], 0)
        self.assertEqual(
            runtime.bake_calls, ["scene-a", "scene-a", "scene-b", "scene-b"]
        )
        self.assertEqual(len(runtime.validate_calls), 2)

        published = self.output / PUBLISHED_DIRECTORY
        self.assertEqual(
            {path.name for path in published.iterdir()},
            {
                BAKE_INDEX_FILE,
                f"{BAKE_INDEX_FILE}.sha256",
                DERIVED_MANIFEST_FILE,
                f"{DERIVED_MANIFEST_FILE}.sha256",
            },
        )
        derived_path = published / DERIVED_MANIFEST_FILE
        derived_raw = derived_path.read_bytes()
        derived = json.loads(derived_raw)
        self.assertEqual(derived_raw, canonical_json_bytes(derived))
        self.assertEqual(derived["geometry_bake_derivation"]["status"], BAKE_STATUS)
        self.assertEqual(
            {scene["split_role"] for scene in derived["scenes"]},
            {"train", "development"},
        )
        navmesh_root = Path(derived["input_roots"]["navmesh_root"])
        self.assertEqual(navmesh_root, (self.output / SCENES_DIRECTORY).resolve())
        self.assertEqual(
            Path(derived["input_roots"]["geometry_bake_root"]),
            self.output.resolve(),
        )
        for provenance_name in ("run_contract", "bake_index"):
            record = derived["geometry_bake_derivation"][provenance_name]
            provenance_path = self.output / record["path"]
            self.assertTrue(provenance_path.is_file())
            self.assertEqual(
                sha256_bytes(provenance_path.read_bytes()), record["content_sha256"]
            )
        for scene in derived["scenes"]:
            self.assertIn("geometry_bake_receipt", scene)
            self.assertTrue((navmesh_root / scene["navmesh"]["path"]).is_file())
            self.assertNotEqual(
                scene["navmesh"]["content_sha256"],
                self.manifest["scenes"][0]["navmesh"]["content_sha256"],
            )

        loaded, snapshot = load_pinned_manifest(
            derived_path, result["derived_manifest_sha256"]
        )
        self.assertEqual(loaded, derived)
        self.assertEqual(snapshot.content_sha256, result["derived_manifest_sha256"])

        map_output = self.root / "frozen-map"
        map_result = build_geometry_map_bundle(
            manifest_path=derived_path,
            expected_manifest_sha256=result["derived_manifest_sha256"],
            navmesh_settings_path=self.settings_path,
            expected_navmesh_settings_sha256=self.settings_sha,
            habitat_sim_version=EXPECTED_HABITAT_VERSION,
            agent_radius_m=0.3,
            agent_height_m=1.5,
            output_directory=map_output,
        )
        self.assertEqual(map_result["scene_count"], 2)
        self.assertTrue((map_output / MAP_FILENAME).is_file())

    def test_exact_resume_validates_without_rebaking(self) -> None:
        first_runtime = FakeBakeRuntime()
        first = self._bake(first_runtime)
        with self.assertRaisesRegex(NavmeshBakeError, "--resume is required"):
            self._bake(FakeBakeRuntime())

        second_runtime = FakeBakeRuntime()
        second = self._bake(second_runtime, resume=True)
        self.assertEqual(second["status"], "resumed")
        self.assertEqual(second["baked_scene_count"], 0)
        self.assertEqual(second["resumed_scene_count"], 2)
        self.assertEqual(second_runtime.bake_calls, [])
        self.assertEqual(len(second_runtime.validate_calls), 2)
        self.assertEqual(
            first["derived_manifest_sha256"], second["derived_manifest_sha256"]
        )

    def test_double_bake_byte_mismatch_is_rejected_without_scene_publication(
        self,
    ) -> None:
        runtime = FakeBakeRuntime(nondeterministic_scene="scene-a")
        with self.assertRaisesRegex(NavmeshBakeError, "not repeatable"):
            self._bake(runtime)
        scenes = self.output / SCENES_DIRECTORY
        self.assertEqual(list(scenes.glob("[!.]*")), [])
        self.assertEqual(list(scenes.glob(f"{baker.SCENE_STAGING_PREFIX}*")), [])
        self.assertFalse((self.output / PUBLISHED_DIRECTORY).exists())

    def test_completed_scene_survives_failure_and_is_resumed(self) -> None:
        failing = FakeBakeRuntime(fail_scene="scene-b")
        with self.assertRaisesRegex(RuntimeError, "injected bake failure"):
            self._bake(failing)
        scene_a_key = sha256_bytes(b"scene-a")
        self.assertTrue((self.output / SCENES_DIRECTORY / scene_a_key).is_dir())
        self.assertFalse((self.output / PUBLISHED_DIRECTORY).exists())

        resumed = FakeBakeRuntime()
        result = self._bake(resumed, resume=True)
        self.assertEqual(result["resumed_scene_count"], 1)
        self.assertEqual(result["baked_scene_count"], 1)
        self.assertEqual(resumed.bake_calls, ["scene-b", "scene-b"])

    def test_glb_drift_during_bake_is_rejected(self) -> None:
        runtime = FakeBakeRuntime(mutate_scene="scene-a")
        with self.assertRaisesRegex(NavmeshBakeError, "source GLB drifted"):
            self._bake(runtime)
        self.assertFalse((self.output / PUBLISHED_DIRECTORY).exists())

    def test_runtime_version_binding_and_fingerprint_are_exact(self) -> None:
        wrong_version = FakeBakeRuntime(identity=runtime_identity(version="0.3.3"))
        with self.assertRaisesRegex(NavmeshBakeError, "Habitat version mismatch"):
            self._bake(wrong_version)
        self.assertFalse(self.output.exists())

        wrong_binding = FakeBakeRuntime(
            identity=runtime_identity(bindings_sha256="c" * 64)
        )
        with self.assertRaisesRegex(NavmeshBakeError, "bindings SHA256 mismatch"):
            self._bake(wrong_binding)
        self.assertFalse(self.output.exists())

        bad_identity = runtime_identity()
        bad_identity["runtime_fingerprint_sha256"] = "d" * 64
        with self.assertRaisesRegex(NavmeshBakeError, "fingerprint mismatch"):
            self._bake(FakeBakeRuntime(identity=bad_identity))
        self.assertFalse(self.output.exists())

    def test_final_reserved_and_duplicate_geometry_inputs_are_rejected(self) -> None:
        forbidden = copy.deepcopy(self.manifest)
        forbidden["scenes"][0]["split_role"] = "final_reserved"  # type: ignore[index]
        self._write_manifest(forbidden)
        with self.assertRaisesRegex(NavmeshBakeError, "forbidden/final"):
            self._bake()
        self.assertFalse(self.output.exists())

        duplicate = copy.deepcopy(self.manifest)
        duplicate["scenes"][1]["environment"] = copy.deepcopy(  # type: ignore[index]
            duplicate["scenes"][0]["environment"]  # type: ignore[index]
        )
        self._write_manifest(duplicate)
        with self.assertRaisesRegex(NavmeshBakeError, "duplicate scene GLB"):
            self._bake()
        self.assertFalse(self.output.exists())

    def test_tampered_scene_or_published_artifact_is_never_overwritten(self) -> None:
        self._bake()
        scene_key = sha256_bytes(b"scene-a")
        navmesh = self.output / SCENES_DIRECTORY / scene_key / "scene.navmesh"
        original = navmesh.read_bytes()
        navmesh.write_bytes(original + b"TAMPER")
        with self.assertRaisesRegex(NavmeshBakeError, "baked navmesh bytes changed"):
            self._bake(resume=True)
        navmesh.write_bytes(original)

        derived = self.output / PUBLISHED_DIRECTORY / DERIVED_MANIFEST_FILE
        derived.write_bytes(derived.read_bytes() + b"TAMPER")
        with self.assertRaisesRegex(NavmeshBakeError, "published.*changed"):
            self._bake(resume=True)

    def test_stale_staging_cleanup_is_signature_bound(self) -> None:
        failing = FakeBakeRuntime(fail_scene="scene-a")
        with self.assertRaisesRegex(RuntimeError, "injected bake failure"):
            self._bake(failing)
        contract_raw = (self.output / baker.RUN_CONTRACT_FILE).read_bytes()
        run_signature = sha256_bytes(contract_raw)
        scene_key = sha256_bytes(b"scene-a")
        expected_name = f"{baker.SCENE_STAGING_PREFIX}{scene_key}-{run_signature}"
        exact_stale = self.output / SCENES_DIRECTORY / expected_name
        exact_stale.mkdir()
        result = self._bake(FakeBakeRuntime(), resume=True)
        self.assertEqual(result["status"], "written")

        published = self.output / PUBLISHED_DIRECTORY
        for path in published.iterdir():
            path.unlink()
        published.rmdir()
        unknown = self.output / f"{baker.PUBLISH_STAGING_PREFIX}{'e' * 64}"
        unknown.mkdir()
        with self.assertRaisesRegex(NavmeshBakeError, "ownership mismatch"):
            self._bake(FakeBakeRuntime(), resume=True)
        self.assertTrue(unknown.is_dir())

    def test_observation_parser_rejects_bool_counts_and_non_xyz_bounds(self) -> None:
        valid = FakeBakeRuntime._observation(Path("scene.glb")).to_dict()
        invalid_count = copy.deepcopy(valid)
        invalid_count["vertex_count"] = True
        with self.assertRaisesRegex(NavmeshBakeError, "topology"):
            NavmeshObservation.from_dict(invalid_count)
        invalid_bounds = copy.deepcopy(valid)
        invalid_bounds["bounds_min_xyz"] = [0.0, 1.0]
        with self.assertRaisesRegex(NavmeshBakeError, "minimum bounds"):
            NavmeshObservation.from_dict(invalid_bounds)

    def test_serialization_equivalence_uses_serialized_vertical_bounds(self) -> None:
        in_simulator = NavmeshObservation(
            navigable_area_m2=26.0,
            bounds_min_xyz=(-11.0, -0.1, -5.0),
            bounds_max_xyz=(4.0, 2.7, 3.0),
            vertex_count=522,
            index_count=522,
        )
        serialized = NavmeshObservation(
            navigable_area_m2=26.0,
            bounds_min_xyz=(-11.0, -0.1, -5.0),
            bounds_max_xyz=(4.0, 4.3, 3.0),
            vertex_count=522,
            index_count=522,
        )
        settings = {"agent_height": 1.5, "cell_height": 0.2}
        baker._observations_serialization_equivalent(
            in_simulator, serialized, settings, "fixture"
        )

        horizontal_drift = copy.deepcopy(serialized.to_dict())
        horizontal_drift["bounds_max_xyz"][0] += 0.01
        with self.assertRaisesRegex(NavmeshBakeError, "horizontal maximum"):
            baker._observations_serialization_equivalent(
                in_simulator,
                NavmeshObservation.from_dict(horizontal_drift),
                settings,
                "fixture",
            )

        wrong_vertical_expansion = copy.deepcopy(serialized.to_dict())
        wrong_vertical_expansion["bounds_max_xyz"][1] = 4.4
        with self.assertRaisesRegex(NavmeshBakeError, "quantized agent-height"):
            baker._observations_serialization_equivalent(
                in_simulator,
                NavmeshObservation.from_dict(wrong_vertical_expansion),
                settings,
                "fixture",
            )

        shifted_vertical_minimum = copy.deepcopy(serialized.to_dict())
        shifted_vertical_minimum["bounds_min_xyz"][1] += 0.01
        with self.assertRaisesRegex(NavmeshBakeError, "quantized agent-height"):
            baker._observations_serialization_equivalent(
                in_simulator,
                NavmeshObservation.from_dict(shifted_vertical_minimum),
                settings,
                "fixture",
            )


if __name__ == "__main__":
    unittest.main()
