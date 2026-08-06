"""Adversarial tests for the atomic frozen-geometry map builder."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from MemNavData import build_frozen_geometry_map as builder
from MemNavData.build_frozen_geometry_map import (
    GEOMETRY_MAP_SCHEMA,
    IDENTITY_DIRECTORY,
    MAP_FILENAME,
    GeometryMapBuildError,
    build_geometry_map_bundle,
    canonical_json_bytes,
    parse_root_overrides,
    sha256_bytes,
)
from MemNavData.collect_real_h24_rollouts import load_geometry_map
from MemNavData.habitat_rollout_primitives import (
    FrozenGeometryIdentity,
    NAVMESH_SETTING_FIELDS,
)


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


class FrozenGeometryMapBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environments = self.root / "environments"
        self.navmeshes = self.root / "navmeshes"
        self.environments.mkdir()
        self.navmeshes.mkdir()
        self.scene_files: dict[str, tuple[Path, Path]] = {}
        for scene in ("scene-a", "scene-b"):
            glb = self.environments / f"{scene}.glb"
            navmesh = self.navmeshes / f"{scene}.navmesh"
            glb.write_bytes(f"GLB::{scene}\x00".encode())
            navmesh.write_bytes(f"NAVMESH::{scene}\x00".encode())
            self.scene_files[scene] = (glb, navmesh)
        self.manifest = self._manifest_for(("scene-a", "scene-b"))
        self.manifest_path = self.root / "expert-manifest.json"
        self.settings_path = self.root / "navmesh-settings.json"
        self._write_manifest(self.manifest)
        self._write_settings(navmesh_settings())
        self.output = self.root / "geometry-bundle"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest_for(self, scenes: tuple[str, ...]) -> dict[str, object]:
        rows = []
        for scene in scenes:
            glb, navmesh = self.scene_files[scene]
            rows.append(
                {
                    "scene": scene,
                    "split_role": "train",
                    "environment": file_record(glb, self.environments),
                    "navmesh": file_record(navmesh, self.navmeshes),
                    "selected_episodes": [],
                }
            )
        return {
            "schema_version": "nlsr_v2_expert_candidate_manifest_v2",
            "input_roots": {
                "episode_root": str(self.root / "unused-episodes"),
                "environment_root": str(self.environments),
                "navmesh_root": str(self.navmeshes),
            },
            "scenes": rows,
            "samples": [],
            "summary": {"scene_count": len(rows)},
        }

    def _write_manifest(self, value: object) -> str:
        raw = canonical_json_bytes(value)
        self.manifest_path.write_bytes(raw)
        self.manifest_sha = sha256_bytes(raw)
        return self.manifest_sha

    def _write_settings(self, value: object, *, raw: bytes | None = None) -> str:
        payload = canonical_json_bytes(value) if raw is None else raw
        self.settings_path.write_bytes(payload)
        self.settings_sha = sha256_bytes(payload)
        return self.settings_sha

    def _build(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "manifest_path": self.manifest_path,
            "expected_manifest_sha256": self.manifest_sha,
            "navmesh_settings_path": self.settings_path,
            "expected_navmesh_settings_sha256": self.settings_sha,
            "habitat_sim_version": "0.3.3",
            "agent_radius_m": 0.3,
            "agent_height_m": 1.5,
            "output_directory": self.output,
        }
        arguments.update(overrides)
        return build_geometry_map_bundle(**arguments)  # type: ignore[arg-type]

    def test_build_is_canonical_relocatable_and_collector_loadable(self) -> None:
        result = self._build()
        self.assertEqual(result["status"], "written")
        self.assertIn("not proof", result["navmesh_settings_semantics"])
        map_path = self.output / MAP_FILENAME
        map_raw = map_path.read_bytes()
        value = json.loads(map_raw)
        self.assertEqual(map_raw, canonical_json_bytes(value))
        self.assertEqual(value["schema_version"], GEOMETRY_MAP_SCHEMA)
        self.assertEqual(set(value["scenes"]), {"scene-a", "scene-b"})
        map_sha = sha256_bytes(map_raw)
        self.assertEqual(result["geometry_map_sha256"], map_sha)
        self.assertEqual(
            (self.output / f"{MAP_FILENAME}.sha256").read_text(),
            f"{map_sha}  {MAP_FILENAME}\n",
        )
        for scene, entry in value["scenes"].items():
            identity_relative = Path(entry["identity_path"])
            self.assertFalse(identity_relative.is_absolute())
            self.assertEqual(identity_relative.parts[0], IDENTITY_DIRECTORY)
            identity = FrozenGeometryIdentity.load_json(self.output / identity_relative)
            self.assertEqual(identity.identity_sha256, entry["identity_sha256"])
            self.assertEqual(identity.habitat_sim_version, "0.3.3")
            self.assertEqual(
                identity.glb_sha256,
                sha256_bytes(self.scene_files[scene][0].read_bytes()),
            )

        loaded = load_geometry_map(map_path, map_sha)
        self.assertEqual(set(loaded), {"scene-a", "scene-b"})
        self.assertTrue(
            all(
                entry.identity_path.is_relative_to(self.output)
                for entry in loaded.values()
            )
        )

    def test_cpu_cli_requires_and_reports_all_explicit_contract_inputs(self) -> None:
        cli_output = self.root / "cli-geometry-bundle"
        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            builder.main(
                [
                    "--expert-manifest",
                    str(self.manifest_path),
                    "--expected-manifest-sha256",
                    self.manifest_sha,
                    "--navmesh-settings-json",
                    str(self.settings_path),
                    "--expected-navmesh-settings-sha256",
                    self.settings_sha,
                    "--habitat-sim-version",
                    "0.3.3",
                    "--agent-radius-m",
                    "0.3",
                    "--agent-height-m",
                    "1.5",
                    "--output-directory",
                    str(cli_output),
                ]
            )
        receipt = json.loads(standard_output.getvalue())
        self.assertEqual(receipt["status"], "written")
        self.assertEqual(receipt["scene_count"], 2)
        self.assertEqual(receipt["manifest_sha256"], self.manifest_sha)
        self.assertEqual(receipt["navmesh_settings_sha256"], self.settings_sha)
        self.assertIn("not proof", receipt["navmesh_settings_semantics"])
        self.assertTrue((cli_output / MAP_FILENAME).is_file())

    def test_exact_resume_accepts_and_default_rejects_existing_output(self) -> None:
        first = self._build()
        with self.assertRaisesRegex(GeometryMapBuildError, "already exists"):
            self._build()
        second = self._build(resume=True)
        self.assertEqual(first["geometry_map_sha256"], second["geometry_map_sha256"])
        self.assertEqual(second["status"], "resumed")

    def test_resume_rejects_changed_extra_or_incomplete_bundle(self) -> None:
        self._build()
        map_value = json.loads((self.output / MAP_FILENAME).read_bytes())
        identity = self.output / map_value["scenes"]["scene-a"]["identity_path"]
        original = identity.read_bytes()
        identity.write_bytes(original + b"drift")
        with self.assertRaisesRegex(GeometryMapBuildError, "differs"):
            self._build(resume=True)
        identity.write_bytes(original)
        (self.output / "unexpected.txt").write_text("extra")
        with self.assertRaisesRegex(GeometryMapBuildError, "unexpected"):
            self._build(resume=True)
        (self.output / "unexpected.txt").unlink()
        (self.output / f"{MAP_FILENAME}.sha256").unlink()
        with self.assertRaisesRegex(GeometryMapBuildError, "incomplete"):
            self._build(resume=True)

    def test_resume_requires_existing_complete_directory(self) -> None:
        with self.assertRaisesRegex(GeometryMapBuildError, "requires an existing"):
            self._build(resume=True)
        self.output.write_text("not a directory")
        with self.assertRaisesRegex(GeometryMapBuildError, "not a directory"):
            self._build(resume=True)

    def test_input_content_pins_and_canonical_encodings_are_mandatory(self) -> None:
        with self.assertRaisesRegex(GeometryMapBuildError, "manifest SHA256 mismatch"):
            self._build(expected_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(GeometryMapBuildError, "Settings SHA256 mismatch"):
            self._build(expected_navmesh_settings_sha256="f" * 64)

        pretty_manifest = json.dumps(self.manifest, indent=2).encode() + b"\n"
        self.manifest_path.write_bytes(pretty_manifest)
        self.manifest_sha = sha256_bytes(pretty_manifest)
        with self.assertRaisesRegex(GeometryMapBuildError, "manifest is not canonical"):
            self._build()

        self._write_manifest(self.manifest)
        pretty_settings = json.dumps(navmesh_settings(), indent=2).encode() + b"\n"
        self._write_settings(navmesh_settings(), raw=pretty_settings)
        with self.assertRaisesRegex(GeometryMapBuildError, "canonical compact"):
            self._build()

    def test_duplicate_json_keys_and_nonfinite_values_are_rejected(self) -> None:
        duplicate = b'{"agent_height":1.5,"agent_height":1.5}\n'
        self._write_settings({}, raw=duplicate)
        with self.assertRaisesRegex(GeometryMapBuildError, "duplicate key"):
            self._build()
        nonfinite = canonical_json_bytes(navmesh_settings()).replace(
            b'"cell_size":0.05', b'"cell_size":NaN'
        )
        self._write_settings({}, raw=nonfinite)
        with self.assertRaisesRegex(GeometryMapBuildError, "non-finite"):
            self._build()

    def test_settings_are_complete_and_match_explicit_agent_geometry(self) -> None:
        missing = navmesh_settings()
        missing.pop("edge_max_len")
        self._write_settings(missing)
        with self.assertRaisesRegex(GeometryMapBuildError, "fields changed"):
            self._build()
        extra = navmesh_settings()
        extra["new_field"] = 1.0
        self._write_settings(extra)
        with self.assertRaisesRegex(GeometryMapBuildError, "fields changed"):
            self._build()

        self._write_settings(navmesh_settings())
        with self.assertRaisesRegex(GeometryMapBuildError, "agent_radius_m disagrees"):
            self._build(agent_radius_m=0.31)
        with self.assertRaisesRegex(GeometryMapBuildError, "agent_height_m disagrees"):
            self._build(agent_height_m=1.6)
        with self.assertRaisesRegex(GeometryMapBuildError, "finite and positive"):
            self._build(agent_height_m=float("nan"))
        with self.assertRaisesRegex(GeometryMapBuildError, "version is invalid"):
            self._build(habitat_sim_version=" 0.3.3")

    def test_duplicate_scene_and_duplicate_geometry_path_are_rejected(self) -> None:
        duplicate_scene = copy.deepcopy(self.manifest)
        duplicate_scene["scenes"].append(
            copy.deepcopy(  # type: ignore[index,union-attr]
                duplicate_scene["scenes"][0]
            )
        )  # type: ignore[index]
        duplicate_scene["summary"]["scene_count"] = 3  # type: ignore[index]
        self._write_manifest(duplicate_scene)
        with self.assertRaisesRegex(GeometryMapBuildError, "duplicate scene id"):
            self._build()

        duplicate_path = copy.deepcopy(self.manifest)
        duplicate_path["scenes"][1]["environment"] = copy.deepcopy(  # type: ignore[index]
            duplicate_path["scenes"][0]["environment"]
        )  # type: ignore[index]
        self._write_manifest(duplicate_path)
        with self.assertRaisesRegex(GeometryMapBuildError, "duplicate geometry path"):
            self._build()

    def test_missing_directory_symlink_and_path_escape_are_rejected(self) -> None:
        glb, _navmesh = self.scene_files["scene-a"]
        glb.unlink()
        with self.assertRaisesRegex(GeometryMapBuildError, "missing"):
            self._build()

        glb.write_bytes(b"replacement")
        self.manifest = self._manifest_for(("scene-a", "scene-b"))
        self._write_manifest(self.manifest)
        target = self.root / "target.glb"
        target.write_bytes(glb.read_bytes())
        glb.unlink()
        glb.symlink_to(target)
        with self.assertRaisesRegex(GeometryMapBuildError, "symlink"):
            self._build()

        glb.unlink()
        glb.mkdir()
        self.manifest["scenes"][0]["environment"]["bytes"] = 0  # type: ignore[index]
        self.manifest["scenes"][0]["environment"]["content_sha256"] = (  # type: ignore[index]
            sha256_bytes(b"")
        )
        self._write_manifest(self.manifest)
        with self.assertRaisesRegex(GeometryMapBuildError, "not a regular file"):
            self._build()

        glb.rmdir()
        glb.write_bytes(b"restored")
        self.manifest = self._manifest_for(("scene-a", "scene-b"))
        record = self.manifest["scenes"][0]["environment"]  # type: ignore[index]
        record["path"] = "../escape.glb"  # type: ignore[index]
        record["path_sha256"] = sha256_bytes(b"../escape.glb")  # type: ignore[index]
        self._write_manifest(self.manifest)
        with self.assertRaisesRegex(GeometryMapBuildError, "not canonical"):
            self._build()

    def test_intermediate_symlink_and_symlinked_manifest_are_rejected(self) -> None:
        nested = self.environments / "nested"
        actual = self.root / "actual"
        actual.mkdir()
        source = actual / "linked.glb"
        source.write_bytes(b"linked")
        nested.symlink_to(actual, target_is_directory=True)
        manifest = self._manifest_for(("scene-a", "scene-b"))
        relative = "nested/linked.glb"
        manifest["scenes"][0]["environment"] = {  # type: ignore[index]
            "path": relative,
            "path_sha256": sha256_bytes(relative.encode()),
            "bytes": len(source.read_bytes()),
            "content_sha256": sha256_bytes(source.read_bytes()),
        }
        self._write_manifest(manifest)
        with self.assertRaisesRegex(GeometryMapBuildError, "uses a symlink"):
            self._build()

        self._write_manifest(self.manifest)
        real_manifest = self.root / "real-manifest.json"
        self.manifest_path.rename(real_manifest)
        self.manifest_path.symlink_to(real_manifest)
        with self.assertRaisesRegex(GeometryMapBuildError, "manifest is a symlink"):
            self._build()

    def test_explicit_root_override_relocates_without_guessing(self) -> None:
        relocated_env = self.root / "relocated-environments"
        relocated_nav = self.root / "relocated-navmeshes"
        relocated_env.mkdir()
        relocated_nav.mkdir()
        for scene, (glb, navmesh) in self.scene_files.items():
            (relocated_env / glb.name).write_bytes(glb.read_bytes())
            (relocated_nav / navmesh.name).write_bytes(navmesh.read_bytes())
        for path in self.environments.iterdir():
            if path.is_file():
                path.unlink()
        for path in self.navmeshes.iterdir():
            path.unlink()
        result = self._build(
            root_overrides={
                "environment_root": relocated_env,
                "navmesh_root": relocated_nav,
            }
        )
        self.assertEqual(result["scene_count"], 2)

        parsed = parse_root_overrides(
            [
                f"environment_root={relocated_env}",
                f"navmesh_root={relocated_nav}",
            ]
        )
        self.assertEqual(parsed["environment_root"], relocated_env.resolve())
        with self.assertRaisesRegex(GeometryMapBuildError, "duplicate root"):
            parse_root_overrides(
                [
                    f"environment_root={relocated_env}",
                    f"environment_root={relocated_env}",
                ]
            )
        with self.assertRaisesRegex(GeometryMapBuildError, "unsupported root"):
            parse_root_overrides([f"episode_root={self.root}"])

    def test_source_drift_before_publication_leaves_no_final_or_staging(self) -> None:
        original_snapshot = builder.snapshot_regular_file
        changed = False
        glb = self.scene_files["scene-a"][0]

        def drifting(path: Path | str, label: str):
            nonlocal changed
            if label.startswith("geometry recheck") and not changed:
                glb.write_bytes(glb.read_bytes() + b"drift")
                changed = True
            return original_snapshot(path, label)

        with mock.patch.object(builder, "snapshot_regular_file", side_effect=drifting):
            with self.assertRaisesRegex(GeometryMapBuildError, "drifted during build"):
                self._build()
        self.assertFalse(os.path.lexists(self.output))
        self.assertFalse((self.root / f".{self.output.name}.lock").exists())
        self.assertEqual(list(self.root.glob(f".{self.output.name}.staging-*")), [])

    def test_staging_failure_never_publishes_partial_output(self) -> None:
        original_write = builder._write_file_fsync
        calls = 0

        def fail_second(path: Path, payload: bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected staging failure")
            original_write(path, payload)

        with mock.patch.object(builder, "_write_file_fsync", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected"):
                self._build()
        self.assertFalse(os.path.lexists(self.output))
        self.assertFalse((self.root / f".{self.output.name}.lock").exists())
        self.assertEqual(list(self.root.glob(f".{self.output.name}.staging-*")), [])

    def test_output_symlink_and_builder_lock_are_fail_closed(self) -> None:
        target = self.root / "target-output"
        target.mkdir()
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(GeometryMapBuildError, "output is a symlink"):
            self._build(resume=True)
        self.output.unlink()
        lock = self.root / f".{self.output.name}.lock"
        lock.write_text("owned")
        with self.assertRaisesRegex(GeometryMapBuildError, "owns lock"):
            self._build()


if __name__ == "__main__":
    unittest.main()
