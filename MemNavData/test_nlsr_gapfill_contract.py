import hashlib
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

from MemNavData.flow_cache_routing import (
    ROUTE_SCHEMA_VERSION,
    ROUTE_STATUS,
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
                        route_builder.FlowAuditError, "keyframe budget"):
                    route_builder._schema_keyframe_budget(
                        SimpleNamespace(DEFAULT_KEYFRAME_BUDGET=invalid))

    def test_route_schema_is_shared_and_builder_never_materializes_links(self):
        self.assertEqual(route_builder.SCHEMA_VERSION, ROUTE_SCHEMA_VERSION)
        source = (ROOT / "MemNavData/build_nlsr_merged_flow.py").read_text(
            encoding="utf-8")
        self.assertIn(f'"status": ROUTE_STATUS', source)
        self.assertNotIn("os.symlink(", source)
        self.assertNotIn("os.link(", source)
        self.assertEqual(ROUTE_STATUS, "flow_routes_audited")

    def test_stage_source_sha_pins_match_exact_files(self):
        raw_stage = ROOT / "MemNavData/slurm_nlsr_historical_gapfill.sbatch"
        flow_stage = ROOT / "MemNavData/slurm_nlsr_flow4096_gapfill.sbatch"
        manifest_stage = ROOT / "MemNavData/slurm_nlsr_corrected_manifest.sbatch"
        checks = (
            (raw_stage, "EXPECTED_AUDITOR_SHA",
             ROOT / "MemNavData/audit_nlsr_historical_gapfill.py"),
            (flow_stage, "EXPECTED_MERGER_SHA",
             ROOT / "MemNavData/build_nlsr_merged_flow.py"),
            (flow_stage, "EXPECTED_ROUTER_SHA",
             ROOT / "MemNavData/flow_cache_routing.py"),
            (manifest_stage, "EXPECTED_BUILDER_SHA",
             ROOT / "MemNavData/build_novel_candidate_manifest.py"),
            (manifest_stage, "EXPECTED_AUDITOR_SHA",
             ROOT / "MemNavData/audit_nlsr_corrected_manifest.py"),
            (manifest_stage, "EXPECTED_ROUTER_SHA",
             ROOT / "MemNavData/flow_cache_routing.py"),
        )
        for script, variable, source in checks:
            with self.subTest(script=script.name, variable=variable):
                self.assertEqual(shell_pin(script, variable), sha256(source))

    def test_manifest_stage_consumes_the_pinned_route_not_a_symlink_view(self):
        script = (ROOT / "MemNavData/slurm_nlsr_corrected_manifest.sbatch").read_text(
            encoding="utf-8")
        self.assertIn("FLOW_ROUTE_PROVENANCE.json", script)
        self.assertIn("--flow-route-provenance", script)
        self.assertIn("--expected-flow-route-sha", script)
        self.assertIn("--expected-flow-route-root", script)
        self.assertNotIn("MERGED_FLOW_PROVENANCE.json", script)
        self.assertNotIn('--flow-cache-root "${MERGED_FLOW_ROOT}"', script)


if __name__ == "__main__":
    unittest.main()
