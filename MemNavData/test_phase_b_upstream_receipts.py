import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from MemNavData.phase_b_upstream_receipts import (
    PhaseBUpstreamPins,
    PhaseBUpstreamReceiptError,
    RECEIPT_BINDING_SCHEMA,
    validate_phase_b_upstream_receipts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_receipt(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    digest = _sha(path)
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def _bundle(root: Path):
    teacher_csv = root / "teacher.csv"
    manifest = root / "manifest.json"
    scale = root / "scale.json"
    teacher_csv.write_text("session_id,teacher_covis\ns,0.8\n")
    manifest.write_text("{}\n")
    scale.write_text("{}\n")
    teacher_sha = _sha(teacher_csv)
    manifest_sha = _sha(manifest)
    scale_sha = _sha(scale)

    teacher_audit = root / "teacher_audit.json"
    teacher_audit_sha = _publish_receipt(
        teacher_audit,
        {
            "schema_version": ("manifest_native_causal_covisibility_teacher_audit_v1"),
            "status": "audited_not_deployment_approved",
            "deployment_approved": False,
            "producer_content_sha256": "1" * 64,
            "configuration_sha256": "2" * 64,
            "runtime_identity_sha256": "3" * 64,
            "counts": {"samples": 600, "candidates": 17_845},
            "manifest": {
                "content_sha256": manifest_sha,
                "sample_count": 600,
            },
            "csv": {
                "content_sha256": teacher_sha,
                "candidate_rows": 17_845,
                "phase_b_contract": {
                    "allowed_kinds": [
                        "manifest_causal_goal_localization_train",
                        "manifest_causal_goal_localization_development",
                    ],
                    "causal_manifest_sample_id": "sample_id",
                    "episode": "source_episode",
                    "exact_candidate_cover": True,
                    "session_id": "sample_id",
                },
            },
            "invariants": {
                "all_candidate_frames_strictly_before_decision": True,
                "exact_manifest_sample_cover": True,
                "phase_b_csv_exact_candidate_cover": True,
                "phase_b_csv_exact_manifest_session_cover": True,
            },
        },
    )

    acceptance_commit = "4" * 40
    producer = {
        "source_bundle_sha256": "5" * 64,
        "configuration_sha256": "6" * 64,
        "lingbot_commit": "7" * 40,
        "weights_sha256": "8" * 64,
        "stream_source_sha256": "9" * 64,
    }
    scale_acceptance = root / "scale_acceptance.json"
    scale_acceptance_sha = _publish_receipt(
        scale_acceptance,
        {
            "schema_version": "nlsr_causal_ground_scale_acceptance_v1",
            "status": "causal_prefixes_physically_rebound",
            "acceptance_commit": acceptance_commit,
            "inputs": {
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": manifest_sha,
                "scale_artifact_path": str(scale.resolve()),
                "scale_artifact_sha256": scale_sha,
            },
            "coverage": {
                "scene_count": 50,
                "episode_count": 100,
                "sample_count": 600,
                "future_frames_consumed": 0,
                "all_episode_estimates_valid": True,
            },
            "physical_rebinding": {
                "independent_prefix_validation_passes": 2,
                "routed_cache_pairs_reopened": 100,
                "camera_pose_prefix_hash_checks": 200,
                "rgb_prefix_hash_checks": 200,
            },
            "producer": producer,
        },
    )
    pins = PhaseBUpstreamPins(
        teacher_csv_sha256=teacher_sha,
        teacher_audit_sha256=teacher_audit_sha,
        manifest_sha256=manifest_sha,
        scale_artifact_sha256=scale_sha,
        scale_acceptance_sha256=scale_acceptance_sha,
        scale_acceptance_commit=acceptance_commit,
        scale_producer_sha256=producer["source_bundle_sha256"],
        scale_configuration_sha256=producer["configuration_sha256"],
        scale_lingbot_commit=producer["lingbot_commit"],
        scale_weights_sha256=producer["weights_sha256"],
        scale_stream_source_sha256=producer["stream_source_sha256"],
    )
    paths = {
        "teacher_csv_path": teacher_csv,
        "teacher_audit_path": teacher_audit,
        "manifest_path": manifest,
        "scale_artifact_path": scale,
        "scale_acceptance_path": scale_acceptance,
    }
    return paths, pins


class PhaseBUpstreamReceiptTest(unittest.TestCase):
    def test_accepts_exact_formal_upstream_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, pins = _bundle(Path(temporary))
            summary = validate_phase_b_upstream_receipts(**paths, pins=pins)
            self.assertEqual(summary["schema_version"], RECEIPT_BINDING_SCHEMA)
            self.assertEqual(summary["teacher"]["samples"], 600)
            self.assertEqual(summary["teacher"]["candidates"], 17_845)
            self.assertEqual(summary["scale"]["episode_count"], 100)
            self.assertEqual(summary["scale"]["future_frames_consumed"], 0)

    def test_rejects_teacher_payload_changed_after_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, pins = _bundle(Path(temporary))
            paths["teacher_csv_path"].write_text("changed\n")
            with self.assertRaisesRegex(
                PhaseBUpstreamReceiptError, "teacher CSV changed"
            ):
                validate_phase_b_upstream_receipts(**paths, pins=pins)

    def test_rejects_cross_run_scale_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, pins = _bundle(root)
            receipt = json.loads(paths["scale_acceptance_path"].read_text())
            receipt["inputs"]["manifest_path"] = str(root / "other_manifest.json")
            changed_sha = _publish_receipt(paths["scale_acceptance_path"], receipt)
            changed_pins = PhaseBUpstreamPins(
                **{
                    **pins.__dict__,
                    "scale_acceptance_sha256": changed_sha,
                }
            )
            with self.assertRaisesRegex(PhaseBUpstreamReceiptError, "physical paths"):
                validate_phase_b_upstream_receipts(**paths, pins=changed_pins)

    def test_rejects_receipt_sidecar_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, pins = _bundle(Path(temporary))
            sidecar = Path(f"{paths['teacher_audit_path']}.sha256")
            sidecar.write_text("0" * 64 + f"  {paths['teacher_audit_path'].name}\n")
            with self.assertRaisesRegex(PhaseBUpstreamReceiptError, "sidecar mismatch"):
                validate_phase_b_upstream_receipts(**paths, pins=pins)


if __name__ == "__main__":
    unittest.main()
