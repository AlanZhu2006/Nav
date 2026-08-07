"""Adversarial tests for the reusable NavMesh bake cross-auditor."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from MemNavData import audit_pinned_navmesh_bake as auditor
from MemNavData import bake_pinned_navmeshes as baker
from MemNavData.audit_pinned_navmesh_bake import (
    AuditContract,
    NavmeshBakeAuditError,
    audit_navmesh_bake,
)
from MemNavData.build_frozen_geometry_map import canonical_json_bytes, sha256_bytes
from MemNavData.test_bake_pinned_navmeshes import (
    BINDINGS_SHA256,
    FakeBakeRuntime,
    file_record,
    navmesh_settings,
)


class NavmeshBakeAuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_root = self.root / "run"
        self.output_root = self.run_root / "navmesh-bake"
        self.environments = self.root / "environments"
        self.old_navmeshes = self.root / "old-navmeshes"
        self.run_root.mkdir()
        self.environments.mkdir()
        self.old_navmeshes.mkdir()
        rows = []
        for index, scene in enumerate(("scene-a", "scene-b")):
            glb = self.environments / f"{scene}.glb"
            old_navmesh = self.old_navmeshes / f"{scene}.navmesh"
            glb.write_bytes(f"GLB::{scene}\x00".encode())
            old_navmesh.write_bytes(f"OLD::{scene}\x00".encode())
            rows.append(
                {
                    "scene": scene,
                    "split_role": "train" if index == 0 else "development",
                    "environment": file_record(glb, self.environments),
                    "navmesh": file_record(old_navmesh, self.old_navmeshes),
                    "selected_episodes": [],
                }
            )
        manifest = {
            "schema_version": "nlsr_v2_expert_candidate_manifest_v2",
            "input_roots": {
                "episode_root": str(self.root / "unused-episodes"),
                "environment_root": str(self.environments),
                "navmesh_root": str(self.old_navmeshes),
            },
            "scenes": rows,
            "samples": [],
            "summary": {"scene_count": 2},
        }
        self.manifest_path = self.root / "manifest.json"
        self.settings_path = self.root / "settings.json"
        self.manifest_sha = self._write_canonical(self.manifest_path, manifest)
        self.settings_sha = self._write_canonical(
            self.settings_path, navmesh_settings()
        )
        self.producer_path = Path(baker.__file__).resolve()
        self.auditor_path = Path(auditor.__file__).resolve()
        self.launcher_path = self.root / "slurm_nlsr_navmesh_bake.sbatch"
        self.habitat_python_path = self.root / "habitat-python"
        self.base_sif_path = self.root / "base.sif"
        self.launcher_path.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
        self.habitat_python_path.write_bytes(b"FAKE_PYTHON_BINARY")
        self.base_sif_path.write_bytes(b"FAKE_EXACT_SIF_BYTES")
        self.producer_sha = sha256_bytes(self.producer_path.read_bytes())
        self.auditor_sha = sha256_bytes(self.auditor_path.read_bytes())
        self.launcher_sha = sha256_bytes(self.launcher_path.read_bytes())
        self.habitat_python_sha = sha256_bytes(self.habitat_python_path.read_bytes())
        self.base_sif_sha = sha256_bytes(self.base_sif_path.read_bytes())
        baker.bake_geometry_bundle(
            manifest_path=self.manifest_path,
            expected_manifest_sha256=self.manifest_sha,
            settings_path=self.settings_path,
            expected_settings_sha256=self.settings_sha,
            expected_habitat_version=baker.EXPECTED_HABITAT_VERSION,
            expected_habitat_bindings_sha256=BINDINGS_SHA256,
            output_root=self.output_root,
            runtime=FakeBakeRuntime(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_canonical(path: Path, value: object) -> str:
        payload = canonical_json_bytes(value)
        path.write_bytes(payload)
        return sha256_bytes(payload)

    @staticmethod
    def _write_artifact_with_sidecar(path: Path, value: object) -> str:
        payload = canonical_json_bytes(value)
        path.write_bytes(payload)
        digest = sha256_bytes(payload)
        Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n")
        return digest

    def _contract(self, **overrides: object) -> AuditContract:
        values: dict[str, object] = {
            "output_root": self.output_root,
            "run_root": self.run_root,
            "expected_parent_manifest_sha256": self.manifest_sha,
            "expected_settings_sha256": self.settings_sha,
            "expected_bindings_sha256": BINDINGS_SHA256,
            "expected_producer_sha256": self.producer_sha,
            "expected_auditor_sha256": self.auditor_sha,
            "expected_habitat_version": baker.EXPECTED_HABITAT_VERSION,
            "expected_producer_commit": "c" * 40,
            "expected_launcher_sha256": self.launcher_sha,
            "producer_path": self.producer_path,
            "auditor_path": self.auditor_path,
            "launcher_path": self.launcher_path,
            "habitat_python_path": self.habitat_python_path,
            "expected_habitat_python_sha256": self.habitat_python_sha,
            "base_sif_path": self.base_sif_path,
            "expected_base_sif_sha256": self.base_sif_sha,
            "expected_base_sif_bytes": self.base_sif_path.stat().st_size,
            "expected_scene_count": 2,
            "expected_fresh_simulator_repetitions": 2,
        }
        values.update(overrides)
        return AuditContract(**values)  # type: ignore[arg-type]

    def _load_derived_and_index(self) -> tuple[Path, dict, Path, dict]:
        published = self.output_root / "published"
        derived_path = published / auditor.DERIVED_MANIFEST_FILE
        index_path = published / auditor.BAKE_INDEX_FILE
        return (
            derived_path,
            json.loads(derived_path.read_bytes()),
            index_path,
            json.loads(index_path.read_bytes()),
        )

    def _cli_args(self) -> list[str]:
        return [
            "--output-root",
            str(self.output_root),
            "--run-root",
            str(self.run_root),
            "--expected-parent-manifest-sha256",
            self.manifest_sha,
            "--expected-settings-sha256",
            self.settings_sha,
            "--expected-bindings-sha256",
            BINDINGS_SHA256,
            "--expected-producer-sha256",
            self.producer_sha,
            "--expected-auditor-sha256",
            self.auditor_sha,
            "--expected-habitat-version",
            baker.EXPECTED_HABITAT_VERSION,
            "--expected-producer-commit",
            "c" * 40,
            "--expected-launcher-sha256",
            self.launcher_sha,
            "--producer-path",
            str(self.producer_path),
            "--auditor-path",
            str(self.auditor_path),
            "--launcher-path",
            str(self.launcher_path),
            "--habitat-python-path",
            str(self.habitat_python_path),
            "--expected-habitat-python-sha256",
            self.habitat_python_sha,
            "--base-sif-path",
            str(self.base_sif_path),
            "--expected-base-sif-sha256",
            self.base_sif_sha,
            "--expected-base-sif-bytes",
            str(self.base_sif_path.stat().st_size),
            "--expected-scene-count",
            "2",
            "--expected-fresh-simulator-repetitions",
            "2",
        ]

    def _coherently_rewrite_scene_receipt(
        self,
        scene_id: str,
        mutate,
    ) -> None:
        derived_path, derived, index_path, index = self._load_derived_and_index()
        derived_by_scene = {row["scene"]: row for row in derived["scenes"]}
        receipt_record = index["scenes"][scene_id]["bake_receipt"]
        receipt_path = self.output_root / receipt_record["path"]
        receipt = json.loads(receipt_path.read_bytes())
        mutate(receipt)
        receipt_payload = canonical_json_bytes(receipt)
        receipt_path.write_bytes(receipt_payload)
        replacement = {
            "path": receipt_record["path"],
            "path_sha256": sha256_bytes(receipt_record["path"].encode()),
            "bytes": len(receipt_payload),
            "content_sha256": sha256_bytes(receipt_payload),
        }
        index["scenes"][scene_id]["bake_receipt"] = replacement
        derived_by_scene[scene_id]["geometry_bake_receipt"] = replacement
        index_sha = self._write_artifact_with_sidecar(index_path, index)
        index_payload = index_path.read_bytes()
        index_record = derived["geometry_bake_derivation"]["bake_index"]
        index_record.update(
            {
                "bytes": len(index_payload),
                "content_sha256": index_sha,
            }
        )
        self._write_artifact_with_sidecar(derived_path, derived)

    def test_success_is_idempotent_and_receipt_pins_auditor(self) -> None:
        first = audit_navmesh_bake(self._contract())
        second = audit_navmesh_bake(self._contract())
        self.assertEqual(first, second)
        self.assertEqual(first["navmesh_bake_cross_audit"], "passed")
        self.assertEqual(first["scene_count"], 2)
        receipt_path = self.run_root / auditor.LAUNCHER_RECEIPT_FILE
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        self.assertEqual(receipt_raw, canonical_json_bytes(receipt))
        self.assertEqual(receipt["auditor"]["content_sha256"], self.auditor_sha)
        self.assertEqual(
            Path(f"{receipt_path}.sha256").read_text(),
            f"{sha256_bytes(receipt_raw)}  {receipt_path.name}\n",
        )

    def test_cli_contract_runs_the_same_cross_audit(self) -> None:
        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            auditor.main(self._cli_args())
        result = json.loads(standard_output.getvalue())
        self.assertEqual(result["navmesh_bake_cross_audit"], "passed")
        self.assertEqual(result["scene_count"], 2)

    def test_exact_sidecar_tamper_is_rejected(self) -> None:
        derived_path, _derived, _index_path, _index = self._load_derived_and_index()
        sidecar = Path(f"{derived_path}.sha256")
        digest = sha256_bytes(derived_path.read_bytes())
        sidecar.write_text(f"{digest}  wrong-name.json\n")
        with self.assertRaisesRegex(NavmeshBakeAuditError, "exact sidecar mismatch"):
            audit_navmesh_bake(self._contract())

    def test_navmesh_byte_tamper_is_rejected(self) -> None:
        _derived_path, _derived, _index_path, index = self._load_derived_and_index()
        record = index["scenes"]["scene-a"]["navmesh"]
        navmesh = self.output_root / record["path"]
        navmesh.write_bytes(navmesh.read_bytes() + b"TAMPER")
        with self.assertRaisesRegex(
            NavmeshBakeAuditError, "byte length changed|content changed"
        ):
            audit_navmesh_bake(self._contract())

    def test_coherent_scene_receipt_status_tamper_is_rejected(self) -> None:
        self._coherently_rewrite_scene_receipt(
            "scene-a", lambda receipt: receipt.update(status="forged")
        )
        with self.assertRaisesRegex(NavmeshBakeAuditError, "receipt identity changed"):
            audit_navmesh_bake(self._contract())

    def test_runtime_binding_tamper_is_rejected_even_with_new_sidecar(self) -> None:
        derived_path, derived, _index_path, _index = self._load_derived_and_index()
        runtime = derived["geometry_bake_derivation"]["runtime"]
        runtime["runtime_files"]["habitat_sim_bindings"]["content_sha256"] = "d" * 64
        fingerprint_input = {
            "habitat_sim_version": runtime["habitat_sim_version"],
            "python_version": runtime["python_version"],
            "runtime_files": runtime["runtime_files"],
        }
        runtime["runtime_fingerprint_sha256"] = sha256_bytes(
            canonical_json_bytes(fingerprint_input)
        )
        self._write_artifact_with_sidecar(derived_path, derived)
        with self.assertRaisesRegex(NavmeshBakeAuditError, "runtime binding changed"):
            audit_navmesh_bake(self._contract())

    def test_launcher_receipt_or_its_sidecar_cannot_be_replaced(self) -> None:
        audit_navmesh_bake(self._contract())
        receipt_path = self.run_root / auditor.LAUNCHER_RECEIPT_FILE
        receipt = json.loads(receipt_path.read_bytes())
        receipt["producer_commit"] = "e" * 40
        receipt_path.write_bytes(canonical_json_bytes(receipt))
        with self.assertRaisesRegex(
            NavmeshBakeAuditError, "existing launcher receipt differs"
        ):
            audit_navmesh_bake(self._contract())

        receipt_path.unlink()
        Path(f"{receipt_path}.sha256").write_text("0" * 64 + "  forged.json\n")
        with self.assertRaisesRegex(
            NavmeshBakeAuditError, "existing launcher receipt sidecar differs"
        ):
            audit_navmesh_bake(self._contract())

    def test_outer_runtime_and_auditor_pins_are_rechecked(self) -> None:
        self.base_sif_path.write_bytes(self.base_sif_path.read_bytes() + b"DRIFT")
        with self.assertRaisesRegex(NavmeshBakeAuditError, "base SIF changed"):
            audit_navmesh_bake(self._contract())
        self.base_sif_path.write_bytes(b"FAKE_EXACT_SIF_BYTES")
        with self.assertRaisesRegex(
            NavmeshBakeAuditError, "bake auditor source changed"
        ):
            audit_navmesh_bake(self._contract(expected_auditor_sha256="f" * 64))

    def test_auditor_matches_habitat_vertical_bounds_semantics(self) -> None:
        in_simulator = {
            "navigable_area_m2": 26.0,
            "bounds_min_xyz": [-11.0, -0.1, -5.0],
            "bounds_max_xyz": [4.0, 4.3, 3.0],
            "vertex_count": 522,
            "index_count": 522,
        }
        base = dict(in_simulator)
        serialized = {
            **base,
            "bounds_min_xyz": list(base["bounds_min_xyz"]),
            "bounds_max_xyz": [4.0, 2.7, 3.0],
        }
        self.assertTrue(
            auditor._observations_serialization_equivalent(
                in_simulator, serialized
            )
        )
        serialized["bounds_max_xyz"][2] += 0.01
        self.assertFalse(
            auditor._observations_serialization_equivalent(
                in_simulator, serialized
            )
        )

    def test_slurm_launcher_pins_the_exact_auditor_source(self) -> None:
        launcher = Path(auditor.__file__).with_name("slurm_nlsr_navmesh_bake.sbatch")
        match = re.search(
            r'^EXPECTED_AUDITOR_SHA="([0-9a-f]{64})"$',
            launcher.read_text(),
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), sha256_bytes(self.auditor_path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
