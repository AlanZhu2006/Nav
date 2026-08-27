import hashlib
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from MemNavData.flow_cache_routing import (
    ROUTE_SCHEMA_VERSION,
    ROUTE_STATUS,
    canonical_json_bytes,
    load_route_registry,
    sha256_bytes,
)
import MemNavData.build_nlsr_merged_flow as route_builder


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_pin(script: Path, variable: str) -> str:
    pattern = rf'^{re.escape(variable)}="([0-9a-f]{{64}})"$'
    matches = re.findall(pattern, script.read_text(encoding="utf-8"), re.MULTILINE)
    if len(matches) != 1:
        raise AssertionError(f"expected one exact {variable} pin in {script}")
    return matches[0]


class NlsrGapfillContractTest(unittest.TestCase):
    def test_patch_admission_uses_content_pinned_schema_budget(self):
        schema = SimpleNamespace(DEFAULT_KEYFRAME_BUDGET=320)
        self.assertEqual(route_builder._schema_keyframe_budget(schema), 320)
        self.assertTrue(route_builder._patch_budget_compliant(301, 8, 320))
        self.assertTrue(route_builder._patch_budget_compliant(312, 8, 320))
        self.assertFalse(route_builder._patch_budget_compliant(313, 8, 320))
        for invalid in (None, True, 0, 320.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    route_builder.FlowAuditError, "keyframe budget"
                ):
                    route_builder._schema_keyframe_budget(
                        SimpleNamespace(DEFAULT_KEYFRAME_BUDGET=invalid)
                    )
        self.assertEqual(
            route_builder.FLOW_THRESHOLD_TIERS,
            (20.0, 25.0, 30.0, 40.0, 50.0, 60.0),
        )
        self.assertEqual(
            {
                key: value["minimum_threshold"]
                for key, value in route_builder.PATCH_EPISODES.items()
            },
            {
                "B6ByNegPMKs/episode_0001": 60.0,
                "YmJkqBEsHnH/episode_0000": 25.0,
                "YmJkqBEsHnH/episode_0001": 20.0,
            },
        )

    def test_flow_stage_adapts_monotonically_and_binds_admission_logs(self):
        script = (ROOT / "MemNavData/slurm_nlsr_flow4096_gapfill.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("nlsr_flow_threshold_admission.py", script)
        self.assertIn("--minimum-threshold", script)
        self.assertIn("--threshold-admission-root", script)
        self.assertIn("overwrite_args+=(--overwrite)", script)
        self.assertIn("flow_threshold_selection.log", script)
        self.assertIn("no label or downstream metric is read", script)
        self.assertNotIn('run_precompute ymj_ep0 25 "${LIST_ROOT}/ymj_ep0.txt"', script)

    def test_route_schema_is_shared_and_builder_never_materializes_links(self):
        self.assertEqual(route_builder.SCHEMA_VERSION, ROUTE_SCHEMA_VERSION)
        source = (ROOT / "MemNavData/build_nlsr_merged_flow.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"status": ROUTE_STATUS', source)
        self.assertNotIn("os.symlink(", source)
        self.assertNotIn("os.link(", source)
        self.assertIn('"threshold_admission": admission', source)
        self.assertNotIn('**({"threshold_admission"', source)
        self.assertEqual(ROUTE_STATUS, "flow_routes_audited")

    def test_admission_trajectory_stays_inside_loader_compatible_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch = root / "patch"
            route = root / "route"
            chunk = patch / "scene/episode_0000/videos/chunk-000"
            chunk.mkdir(parents=True)
            route.mkdir()
            files = []
            for name, payload in (
                ("lingbot_cache.npz", b"aggregator"),
                ("lingbot_cam_cache.npz", b"camera"),
            ):
                path = chunk / name
                path.write_bytes(payload)
                files.append(
                    {
                        "name": name,
                        "bytes": len(payload),
                        "content_sha256": sha256_bytes(payload),
                    }
                )
            record = {
                "schema_version": ROUTE_SCHEMA_VERSION,
                "status": ROUTE_STATUS,
                "split_sha256": "1" * 64,
                "raw_audit_sha256": "2" * 64,
                "route_root": str(route.resolve()),
                "source_roots": {"flow4096_patch": str(patch.resolve())},
                "official_snapshot_semantics": "test",
                "official_snapshot_sha256": "3" * 64,
                "patch_payloads_fully_sha256": True,
                "counts": {
                    "scenes": 1,
                    "pairs": 1,
                    "flow4096_patch": 1,
                },
                "pairs": [
                    {
                        "episode": "scene/episode_0000",
                        "source_id": "flow4096_patch",
                        "source_relative_chunk": (
                            "scene/episode_0000/videos/chunk-000"
                        ),
                        "validation": {
                            "files": files,
                            "threshold_admission": {"trajectory": "pinned"},
                        },
                    }
                ],
            }
            payload = canonical_json_bytes(record)
            digest = sha256_bytes(payload)
            artifact = route / "FLOW_ROUTE_PROVENANCE.json"
            artifact.write_bytes(payload)
            Path(f"{artifact}.sha256").write_text(
                f"{digest}  {artifact.name}\n", encoding="ascii"
            )
            registry = load_route_registry(artifact, digest)
            self.assertEqual(set(registry.files_by_episode), {"scene/episode_0000"})

    def test_stage_source_sha_pins_match_exact_files(self):
        raw_stage = ROOT / "MemNavData/slurm_nlsr_historical_gapfill.sbatch"
        flow_stage = ROOT / "MemNavData/slurm_nlsr_flow4096_gapfill.sbatch"
        manifest_stage = ROOT / "MemNavData/slurm_nlsr_corrected_manifest.sbatch"
        checks = (
            (
                raw_stage,
                "EXPECTED_AUDITOR_SHA",
                ROOT / "MemNavData/audit_nlsr_historical_gapfill.py",
            ),
            (
                flow_stage,
                "EXPECTED_MERGER_SHA",
                ROOT / "MemNavData/build_nlsr_merged_flow.py",
            ),
            (
                flow_stage,
                "EXPECTED_ADMISSION_SHA",
                ROOT / "MemNavData/nlsr_flow_threshold_admission.py",
            ),
            (
                flow_stage,
                "EXPECTED_ROUTER_SHA",
                ROOT / "MemNavData/flow_cache_routing.py",
            ),
            (
                manifest_stage,
                "EXPECTED_BUILDER_SHA",
                ROOT / "MemNavData/build_novel_candidate_manifest.py",
            ),
            (
                manifest_stage,
                "EXPECTED_AUDITOR_SHA",
                ROOT / "MemNavData/audit_nlsr_corrected_manifest.py",
            ),
            (
                manifest_stage,
                "EXPECTED_ROUTER_SHA",
                ROOT / "MemNavData/flow_cache_routing.py",
            ),
        )
        for script, variable, source in checks:
            with self.subTest(script=script.name, variable=variable):
                self.assertEqual(shell_pin(script, variable), sha256(source))

    def test_manifest_stage_consumes_the_pinned_route_not_a_symlink_view(self):
        script = (ROOT / "MemNavData/slurm_nlsr_corrected_manifest.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("FLOW_ROUTE_PROVENANCE.json", script)
        self.assertIn("--flow-route-provenance", script)
        self.assertIn("--expected-flow-route-sha", script)
        self.assertIn("--expected-flow-route-root", script)
        self.assertNotIn("MERGED_FLOW_PROVENANCE.json", script)
        self.assertNotIn('--flow-cache-root "${MERGED_FLOW_ROOT}"', script)

    def test_multistage_stage_is_dependency_pinned_and_has_no_height_default_in_python(
        self,
    ):
        script_path = ROOT / "MemNavData/slurm_nlsr_multistage_manifest.sbatch"
        script = script_path.read_text(encoding="utf-8")
        self.assertEqual(
            shell_pin(script_path, "EXPECTED_BUILDER_SHA"),
            sha256(ROOT / "MemNavData/build_multistage_candidate_manifest.py"),
        )
        self.assertIn("--expected-routed-manifest-sha", script)
        self.assertIn("--expected-routed-manifest-audit-sha", script)
        self.assertIn("--legacy-camera-height-m", script)
        self.assertIn("multistage_dependency_preflight=passed", script)
        builder = (
            ROOT / "MemNavData/build_multistage_candidate_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("default=0.5", builder)


if __name__ == "__main__":
    unittest.main()
