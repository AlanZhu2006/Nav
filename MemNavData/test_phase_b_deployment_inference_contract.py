"""Tests for the Phase-B deployment inference contract.

The pinned trainer launcher (MemNavData/slurm_nlsr_phase_b_train.sbatch)
imports this module in its preflight, but it was never written, so the formal
trainer could not start.  These tests exercise the guarantees the contract
actually has to provide: canonical byte stability, SHA pinning that rejects
tampering, duplicate-key and non-finite rejection, the frozen ABI constants
that the deployment path depends on, and the privileged-input allowlist.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from MemNavData import phase_b_deployment_inference_contract as contract


class CanonicalBytesTest(unittest.TestCase):
    def test_key_order_does_not_change_bytes(self) -> None:
        left = contract.canonical_json_bytes({"b": 1, "a": [1, 2]})
        right = contract.canonical_json_bytes({"a": [1, 2], "b": 1})
        self.assertEqual(left, right)

    def test_trailing_newline_is_part_of_the_contract(self) -> None:
        self.assertTrue(contract.canonical_json_bytes({"a": 1}).endswith(b"\n"))

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            contract.canonical_json_bytes({"a": float("nan")})

    def test_digest_matches_canonical_bytes(self) -> None:
        value = {"a": 1, "b": "x"}
        self.assertEqual(
            contract.sha256_bytes(contract.canonical_json_bytes(value)),
            contract.sha256_bytes(contract.canonical_json_bytes(dict(value))))


class PinnedJsonLoadTest(unittest.TestCase):
    def _write(self, directory: str, payload: bytes) -> Path:
        path = Path(directory) / "pinned.json"
        path.write_bytes(payload)
        return path

    def test_exact_pin_loads(self) -> None:
        payload = contract.canonical_json_bytes({"a": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            loaded = contract.load_pinned_canonical_json(
                path, contract.sha256_bytes(payload))
            self.assertEqual(loaded["a"], 1)

    def test_tampered_content_is_rejected(self) -> None:
        payload = contract.canonical_json_bytes({"a": 1})
        digest = contract.sha256_bytes(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory, contract.canonical_json_bytes({"a": 2}))
            with self.assertRaises(contract.PhaseBInferenceContractError):
                contract.load_pinned_canonical_json(path, digest)

    def test_missing_file_is_rejected(self) -> None:
        payload = contract.canonical_json_bytes({"a": 1})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(contract.PhaseBInferenceContractError):
                contract.load_pinned_canonical_json(
                    Path(directory) / "absent.json",
                    contract.sha256_bytes(payload))

    def test_malformed_expected_digest_is_rejected(self) -> None:
        payload = contract.canonical_json_bytes({"a": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            with self.assertRaises(contract.PhaseBInferenceContractError):
                contract.load_pinned_canonical_json(path, "not-a-sha")

    def test_duplicate_keys_are_rejected(self) -> None:
        payload = b'{"a": 1, "a": 2}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            with self.assertRaises(contract.PhaseBInferenceContractError):
                contract.load_pinned_canonical_json(
                    path, contract.sha256_bytes(payload))

    def test_non_finite_constants_are_rejected(self) -> None:
        payload = b'{"a": NaN}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            with self.assertRaises(contract.PhaseBInferenceContractError):
                contract.load_pinned_canonical_json(
                    path, contract.sha256_bytes(payload))


class FrozenAbiTest(unittest.TestCase):
    """These constants are part of the deployment ABI: a silent edit would
    let an incompatible artifact load, so they are asserted explicitly."""

    def test_schema_and_semantics_are_pinned(self) -> None:
        self.assertEqual(contract.SCHEMA_VERSION,
                         "nlsr_phase_b_deployment_inference_v1")
        self.assertEqual(contract.POSE_CONVENTION,
                         "lingbot_native_xz_to_navdp_forward_left_v1")
        self.assertEqual(contract.CANDIDATE_SELECTION_SEMANTICS,
                         "causal_dino_topk_temporal_diverse_v1")
        self.assertEqual(
            contract.ENSEMBLE_SEMANTICS,
            "mean_member_probabilities_total_predictive_variance_v1")
        self.assertEqual(
            contract.PRIVILEGED_INPUT_POLICY,
            "strict_deployment_allowlist_no_teacher_gt_or_navmesh_v1")
        self.assertEqual(contract.DEPLOYMENT_APPROVAL_SCHEMA_VERSION,
                         "lingbot_native_phase_b_approval_v1")

    def test_required_key_sets_are_non_empty_and_frozen(self) -> None:
        for name in ("TOP_LEVEL_KEYS", "PROVENANCE_KEYS", "CONFIGURATION_KEYS",
                     "RECORD_KEYS", "CANDIDATE_KEYS", "SUMMARY_KEYS",
                     "DEPLOYMENT_APPROVAL_KEYS"):
            keys = getattr(contract, name)
            self.assertIsInstance(keys, frozenset, name)
            self.assertTrue(keys, name)

    def test_provenance_pins_the_frozen_upstream_identities(self) -> None:
        for key in ("lingbot_commit", "lingbot_weights_sha256",
                    "lingbot_stream_source_sha256"):
            self.assertIn(key, contract.PROVENANCE_KEYS)

    def test_supported_manifest_schemas_are_explicit(self) -> None:
        self.assertIsInstance(contract.SUPPORTED_MANIFEST_SCHEMAS, frozenset)
        self.assertTrue(contract.SUPPORTED_MANIFEST_SCHEMAS)


class ForbiddenKeyTest(unittest.TestCase):
    """Deployment inputs must never carry teacher/GT/navmesh signals."""

    def test_forbidden_fragments_cover_the_known_leak_sources(self) -> None:
        fragments = {item.lower() for item in contract._FORBIDDEN_KEY_FRAGMENTS}
        for expected in ("teacher", "oracle", "ground_truth", "covis",
                         "pathfinder", "navmesh", "habitat_pose", "target_",
                         "label"):
            self.assertIn(expected, fragments)

    def test_geodesic_is_not_covered_by_the_fragment_scan(self) -> None:
        """Documented gap, asserted so it cannot change unnoticed.

        The scan blocks ``pathfinder`` but not ``geodesic``, so a key such as
        ``geodesic_progress`` would pass this defence.  The deployment
        allowlist elsewhere still rejects it; if the fragment list is ever
        extended, this test should be updated together with it."""
        fragments = {item.lower() for item in contract._FORBIDDEN_KEY_FRAGMENTS}
        self.assertNotIn("geodesic", fragments)

    def test_scan_rejects_a_nested_forbidden_key(self) -> None:
        fragment = contract._FORBIDDEN_KEY_FRAGMENTS[0]
        with self.assertRaises(contract.PhaseBInferenceContractError):
            contract._scan_forbidden_keys({"outer": {f"x_{fragment}_y": 1}})

    def test_scan_accepts_a_clean_mapping(self) -> None:
        contract._scan_forbidden_keys(
            {"outer": {"dino_cosine": 0.5, "candidates": [{"rank": 1}]}})


class ValidateEntryPointTest(unittest.TestCase):
    def test_sha_mismatch_fails_closed(self) -> None:
        artifact = {"schema_version": contract.SCHEMA_VERSION}
        wrong = "0" * 64
        with self.assertRaises(contract.PhaseBInferenceContractError):
            contract.validate_phase_b_deployment_inference(
                artifact=artifact, artifact_sha256=wrong,
                manifest={}, manifest_sha256=wrong, pins=None)


if __name__ == "__main__":
    unittest.main()
