import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.prepare_cdec_consumed_closed_loop import prepare


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareCDECConsumedClosedLoopTest(unittest.TestCase):
    def _fixture(self, root: Path):
        scenes = [f"scene_{index:02d}" for index in range(20)]
        episodes = {
            scene: [
                {"episode": f"episode_{index:04d}"}
                for index in range(8)
            ] for scene in scenes
        }
        manifest = {
            "audit": {
                "status": "ok", "development_read": False,
                "blind_read": False,
            },
            "data_role_guards": {"blind_allowed": False},
            "scenes": scenes,
            "episodes": episodes,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        dependency_path = root / "dependency.json"
        dependency_path.write_text(json.dumps({
            "manifest_sha256": sha(manifest_path),
            "dependencies": {
                name: {"path": f"/{name}", "bytes": 1, "sha256": "0" * 64}
                for name in ("gatecurr600", "navdp_checkpoint", "lingbot_map_long")
            },
        }))
        trace_run = root / "trace_run"
        trace_run.mkdir()
        (trace_run / "data_manifest.json").write_bytes(manifest_path.read_bytes())
        report = trace_run / "report.json"
        report.write_bytes(b"opaque consumed report; never decoded")
        for index, scene in enumerate(scenes):
            trace_root = trace_run / "scenes" / f"{index:02d}_{scene}" / "trace_source"
            trace_root.mkdir(parents=True)
            (trace_root / "summary.json").write_bytes(
                b"opaque summary; never decoded")
            for row in episodes[scene]:
                # Deliberately invalid JSON.  Preparation succeeding proves
                # that trace outcomes were not deserialized or inspected.
                (trace_root / f"{row['episode']}_leg1_trace.json").write_bytes(
                    f"opaque-{scene}-{row['episode']}".encode())
        return manifest_path, dependency_path, trace_run, report

    def test_freezes_opaque_trace_hashes_without_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, dependency, trace_run, report = self._fixture(root)
            out = root / "out"
            result = prepare(
                source_manifest=manifest,
                expected_manifest_sha256=sha(manifest),
                source_dependency_receipt=dependency,
                expected_dependency_receipt_sha256=sha(dependency),
                trace_run_root=trace_run,
                trace_run_report=report,
                expected_trace_run_report_sha256=sha(report),
                run_root=out,
            )
            self.assertEqual(result["episodes"], 160)
            receipt = json.loads((out / "trace_receipt.json").read_text())
            self.assertTrue(receipt["trace_bytes_hashed_only"])
            self.assertFalse(receipt["trace_payload_decoded"])
            self.assertFalse(receipt["episode_target_or_outcome_fields_accessed"])
            for name in (
                "data_manifest.json", "dependency_receipt.json",
                "trace_receipt.json", "preparation.json",
            ):
                self.assertTrue((out / name).is_file())
                self.assertTrue((out / f"{name}.sha256").is_file())

    def test_fails_closed_on_missing_trace_and_removes_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, dependency, trace_run, report = self._fixture(root)
            missing = (
                trace_run / "scenes" / "00_scene_00" / "trace_source"
                / "episode_0000_leg1_trace.json")
            missing.unlink()
            out = root / "out"
            with self.assertRaisesRegex(RuntimeError, "identity/count"):
                prepare(
                    source_manifest=manifest,
                    expected_manifest_sha256=sha(manifest),
                    source_dependency_receipt=dependency,
                    expected_dependency_receipt_sha256=sha(dependency),
                    trace_run_root=trace_run,
                    trace_run_report=report,
                    expected_trace_run_report_sha256=sha(report),
                    run_root=out,
                )
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
