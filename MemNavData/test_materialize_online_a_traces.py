import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from materialize_online_a_traces import (
    best_separated_pair,
    discover_single_anchor_candidates,
)


class AnchorMarginTest(unittest.TestCase):
    @staticmethod
    def poses(count: int) -> list[dict]:
        return [
            {"x": float(index) * 0.05, "z": 0.0}
            for index in range(count)
        ]

    def test_default_end_margin_remains_symmetric(self) -> None:
        self.assertIsNone(
            best_separated_pair(self.poses(100), margin=39, min_gap=32)
        )

    def test_paper_asymmetric_margin_keeps_early_anchor_eligible(self) -> None:
        pair = best_separated_pair(
            self.poses(100), margin=39, min_gap=32, end_margin=16
        )
        self.assertIsNotNone(pair)
        _score, first, second, *_ = pair
        self.assertGreaterEqual(first, 39)
        self.assertGreaterEqual(second - first, 32)
        self.assertLess(second, 100 - 16)

    def test_single_query_history_does_not_require_second_anchor(self) -> None:
        payload = {
            "reached": True,
            "source_scene": "scene",
            "episode": "episode_0000",
            "poses": [
                {"x": float(index), "z": 0.0}
                for index in range(60)
            ],
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode_0000_leg1_trace.json"
            path.write_text(json.dumps(payload))
            with patch(
                "materialize_online_a_traces.validate_leg1_trace"
            ), patch(
                "materialize_online_a_traces.native_control_audit",
                return_value={"ok": True},
            ):
                rows = discover_single_anchor_candidates(
                    Path(temporary), minimum_frame=39, end_margin=16
                )
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0].anchor, 39)
        self.assertLess(rows[0].anchor, 60 - 16)


if __name__ == "__main__":
    unittest.main()
