import argparse
import csv
import json

from MemNavData.compare_pi3x_bridge_density import _paired_binary, run


def test_paired_binary_counts_gain_loss() -> None:
    result = _paired_binary([False, True, True], [True, True, False])
    assert result["gain"] == 1
    assert result["loss"] == 1
    assert result["first_success"] == result["second_success"] == 2
    assert result["exact_mcnemar_p"] == 1.0


def test_run_requires_and_compares_the_exact_row_universe(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    fields = [
        "session_id", "scene", "candidate_label", "session_label",
        "decision_frame", "candidate_frame", "dino_cosine", "candidate_rank",
    ]
    rows = [
        {"session_id": "s/positive", "scene": "s", "candidate_label": "1",
         "session_label": "1", "decision_frame": "20", "candidate_frame": "10",
         "dino_cosine": "0.9", "candidate_rank": "0"},
        {"session_id": "s/negative", "scene": "s", "candidate_label": "0",
         "session_label": "0", "decision_frame": "20", "candidate_frame": "5",
         "dino_cosine": "0.8", "candidate_rank": "0"},
    ]
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("\n".join([
        json.dumps({"row_index": 0, "scene": "s", "goal_bearing_error_deg_reporting_only": 40}),
        json.dumps({"row_index": 1, "scene": "s", "goal_bearing_error_deg_reporting_only": 90}),
    ]) + "\n")
    second.write_text("\n".join([
        json.dumps({"row_index": 0, "scene": "s", "goal_bearing_error_deg_reporting_only": 10}),
        json.dumps({"row_index": 1, "scene": "s", "goal_bearing_error_deg_reporting_only": 90}),
    ]) + "\n")
    result = run(argparse.Namespace(
        rows_csv=rows_path,
        first=first,
        second=second,
        output=tmp_path / "summary.json",
        expected_rows=2,
        expected_rows_sha256=None,
    ))
    assert result["positive_candidates"]["paired_within_30deg"]["gain"] == 1
    assert result["session_metrics"]["top8_ceiling"]["gain"] == 1
    assert result["positive_candidate_gap_bins"]["0_32"]["second"]["within_30deg"] == 1
