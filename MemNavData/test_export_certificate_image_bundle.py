import csv
import io
import tarfile

import pytest

from MemNavData.export_certificate_image_bundle import (
    export_bundle,
    referenced_paths,
    validated_relative_path,
)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("query_relative_path", "candidate_relative_path"))
        writer.writeheader()
        writer.writerows(rows)


def test_export_is_unique_sorted_and_content_preserving(tmp_path):
    root = tmp_path / "episodes"
    for relative, content in (("scene/q.jpg", b"query"),
                              ("scene/c.jpg", b"candidate")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    rows = tmp_path / "rows.csv"
    write_rows(rows, [{
        "query_relative_path": "scene/q.jpg",
        "candidate_relative_path": "scene/c.jpg",
    }, {
        "query_relative_path": "scene/q.jpg",
        "candidate_relative_path": "scene/c.jpg",
    }])
    output = tmp_path / "bundle.tar"
    report = export_bundle(rows, root, output, expected_images=2)
    assert report["images"] == 2
    with tarfile.open(output) as archive:
        assert archive.getnames() == ["scene/c.jpg", "scene/q.jpg"]
        assert archive.extractfile("scene/q.jpg").read() == b"query"


@pytest.mark.parametrize("value", ("/absolute.jpg", "../escape.jpg", "a/../b.jpg", ""))
def test_rejects_unsafe_paths(value):
    with pytest.raises(ValueError):
        validated_relative_path(value)


def test_missing_path_column_fails(tmp_path):
    rows = tmp_path / "rows.csv"
    rows.write_text("query_relative_path\na.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError):
        referenced_paths(rows)
