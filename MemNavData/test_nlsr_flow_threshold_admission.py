"""Tests for densest budget-compliant flow-threshold admission."""

from __future__ import annotations

import copy
import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import MemNavData.nlsr_flow_threshold_admission as admission
from MemNavData.build_nlsr_merged_flow import (
    FLOW_FILES,
    FLOW_THRESHOLD_TIERS,
    FlowAuditError,
    canonical_bytes,
    validate_threshold_admission_record,
)


def validation(threshold: float, anchors: int, salt: str = "a") -> dict:
    return {
        "flow_threshold": threshold,
        "anchor_count": anchors,
        "num_scale_frames": 8,
        "total_memory_frames": anchors + 8,
        "strict_patch_keyframe_budget_compliant": anchors + 8 <= 320,
        "precompute_signature": hashlib.sha256(
            f"signature:{salt}".encode()
        ).hexdigest(),
        "files": [
            {
                "name": name,
                "bytes": 10 + index,
                "content_sha256": hashlib.sha256(f"{salt}:{name}".encode()).hexdigest(),
            }
            for index, name in enumerate(FLOW_FILES)
        ],
    }


class FlowThresholdAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patch = self.root / "patch"
        self.logs = self.root / "admission"
        self.patch.mkdir()
        self.schema = SimpleNamespace(DEFAULT_KEYFRAME_BUDGET=320)
        self.arguments = {
            "episode": "YmJkqBEsHnH/episode_0000",
            "frames": 871,
            "minimum_threshold": 25.0,
            "patch_flow_root": self.patch,
            "admission_root": self.logs,
            "cache_schema": self.schema,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _plan(self, state: str, result: dict | None = None, reason: str | None = None):
        with mock.patch.object(
            admission,
            "_inspect_pair",
            return_value=(state, result, reason),
        ):
            return admission.plan_threshold_admission(**self.arguments)

    @property
    def journal(self) -> Path:
        return self.logs / "YmJkqBEsHnH__episode_0000.json"

    def _record(self) -> dict:
        raw = self.journal.read_bytes()
        value = json.loads(raw)
        self.assertEqual(raw, canonical_bytes(value))
        return value

    def test_starts_at_minimum_then_advances_exactly_one_tier_with_overwrite(self):
        first = self._plan("absent")
        self.assertEqual(first["action"], "compute")
        self.assertEqual(first["threshold"], 25.0)
        self.assertFalse(first["overwrite"])

        second = self._plan("valid", validation(25.0, 376, "a"))
        self.assertEqual(second["action"], "overwrite")
        self.assertEqual(second["threshold"], 30.0)
        self.assertTrue(second["overwrite"])

        # Before overwrite completes, the exact previous tier is neither
        # downgraded nor mistaken for the new tier.
        same = self._plan("valid", validation(25.0, 376, "a"))
        self.assertEqual(same["action"], "overwrite")
        self.assertEqual(same["threshold"], 30.0)

        final_validation = validation(30.0, 300, "c")
        accepted = self._plan("valid", final_validation)
        self.assertEqual(accepted["action"], "accept")
        self.assertEqual(accepted["threshold"], 30.0)
        record = self._record()
        self.assertEqual(
            [
                (row["threshold"], row["action"], row["outcome"])
                for row in record["attempts"]
            ],
            [
                (25.0, "compute", "over_budget"),
                (30.0, "overwrite", "accepted"),
            ],
        )
        self.assertIn("no evaluation label", record["selection_basis"])
        validate_threshold_admission_record(
            record,
            episode=self.arguments["episode"],
            minimum_threshold=25.0,
            keyframe_budget=320,
            final_validation=final_validation,
        )
        missing_overwrite = copy.deepcopy(record)
        missing_overwrite["attempts"][1]["action"] = "compute"
        with self.assertRaisesRegex(FlowAuditError, "explicit overwrite action"):
            validate_threshold_admission_record(
                missing_overwrite,
                episode=self.arguments["episode"],
                minimum_threshold=25.0,
                keyframe_budget=320,
            )

    def test_compliant_existing_minimum_is_exactly_resumed(self):
        cached = validation(25.0, 301, "d")
        decision = self._plan("valid", cached)
        self.assertEqual(decision["action"], "accept")
        record = self._record()
        self.assertEqual(record["attempts"][0]["action"], "resume")
        self.assertEqual(record["attempts"][0]["outcome"], "accepted")
        again = self._plan("valid", cached)
        self.assertEqual(again["action"], "accept")

        drifted = validation(25.0, 301, "e")
        with self.assertRaisesRegex(
            admission.ThresholdAdmissionError, "no longer exactly resumes"
        ):
            self._plan("valid", drifted)

    def test_existing_overbudget_minimum_is_inspected_then_overwritten(self):
        decision = self._plan("valid", validation(25.0, 376, "f"))
        self.assertEqual(decision["action"], "overwrite")
        self.assertEqual(decision["threshold"], 30.0)
        record = self._record()
        self.assertEqual(record["attempts"][0]["action"], "inspect_existing")
        self.assertEqual(record["attempts"][0]["outcome"], "over_budget")
        self.assertEqual(record["attempts"][1]["action"], "overwrite")

    def test_interrupted_or_mixed_pending_write_retries_same_tier_only(self):
        self._plan("absent")
        retry = self._plan("invalid", reason="pair provenance differs")
        self.assertEqual(retry["action"], "overwrite")
        self.assertEqual(retry["threshold"], 25.0)
        record = self._record()
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(record["attempts"][0]["action"], "overwrite")
        with (
            mock.patch.object(
                admission,
                "_inspect_pair",
                return_value=("invalid", None, "mixed signatures"),
            ),
            mock.patch.object(
                admission, "_declared_partial_thresholds", return_value=[30.0]
            ),
            self.assertRaisesRegex(
                admission.ThresholdAdmissionError, "refusing a downgrade overwrite"
            ),
        ):
            admission.plan_threshold_admission(**self.arguments)

    def test_untracked_partial_or_nonminimum_cache_is_never_overwritten(self):
        with self.assertRaisesRegex(
            admission.ThresholdAdmissionError, "untracked existing cache is invalid"
        ):
            self._plan("invalid", reason="partial cache pair")
        self.assertFalse(self.journal.exists())
        with self.assertRaisesRegex(
            admission.ThresholdAdmissionError, "not at the episode minimum"
        ):
            self._plan("valid", validation(30.0, 250, "g"))
        self.assertFalse(self.journal.exists())

    def test_thresholds_cannot_skip_downgrade_or_use_unapproved_values(self):
        self._plan("absent")
        with self.assertRaisesRegex(
            admission.ThresholdAdmissionError, "pending or immediately previous"
        ):
            self._plan("valid", validation(40.0, 250, "h"))
        record = self._record()
        record["attempts"][0]["threshold"] = 20.0
        self.journal.write_bytes(canonical_bytes(record))
        with self.assertRaisesRegex(
            admission.ThresholdAdmissionError, "did not advance"
        ):
            self._plan("absent")
        self.assertEqual(FLOW_THRESHOLD_TIERS, (20.0, 25.0, 30.0, 40.0, 50.0, 60.0))
        with self.assertRaisesRegex(FlowAuditError, "approved threshold"):
            admission._threshold_index(35.0, "test threshold")

    def test_maximum_tier_exhaustion_is_logged_and_fails_closed(self):
        self.arguments["episode"] = "B6ByNegPMKs/episode_0001"
        self.arguments["frames"] = 2456
        self.arguments["minimum_threshold"] = 60.0
        first = self._plan("absent")
        self.assertEqual(first["threshold"], 60.0)
        with self.assertRaisesRegex(
            admission.ThresholdAdmissionError, "all approved thresholds"
        ):
            self._plan("valid", validation(60.0, 313, "i"))
        journal = self.logs / "B6ByNegPMKs__episode_0001.json"
        record = json.loads(journal.read_bytes())
        self.assertEqual(record["status"], "exhausted")
        self.assertEqual(record["attempts"][-1]["outcome"], "over_budget")

    def test_atomic_journal_failure_does_not_publish_temporary_file(self):
        with mock.patch.object(os, "replace", side_effect=OSError("injected")):
            with self.assertRaisesRegex(OSError, "injected"):
                self._plan("absent")
        self.assertFalse(self.journal.exists())
        self.assertEqual(list(self.logs.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
