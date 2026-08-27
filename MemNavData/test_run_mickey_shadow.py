import csv

import pytest

from MemNavData.run_mickey_shadow import (
    read_rows,
    safe_image_path,
    selected_indices,
)


def test_selected_indices_are_ordered_bounded_and_unique():
    assert selected_indices(10, "4,1,9", 2) == [4, 1]
    with pytest.raises(ValueError):
        selected_indices(10, "1,1", 0)
    with pytest.raises(IndexError):
        selected_indices(10, "10", 0)


def test_safe_image_path_cannot_escape_bundle(tmp_path):
    image = tmp_path / "scene" / "image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"not-decoded-here")
    assert safe_image_path(tmp_path, "scene/image.jpg") == image
    with pytest.raises(ValueError):
        safe_image_path(tmp_path, "../image.jpg")


def test_read_rows_requires_label_blind_runtime_columns(tmp_path):
    path = tmp_path / "rows.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "session_id", "query_relative_path", "candidate_relative_path",
            "candidate_rank", "dino_cosine"))
        writer.writeheader()
        writer.writerow({
            "session_id": "session",
            "query_relative_path": "q.jpg",
            "candidate_relative_path": "r.jpg",
            "candidate_rank": 0,
            "dino_cosine": 0.9,
        })
    assert len(read_rows(path)) == 1
