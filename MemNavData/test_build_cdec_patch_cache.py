import numpy as np
import pandas as pd
import pytest

from MemNavData.build_cdec_patch_cache import (
    build_relations,
    pair_universe,
    safe_relative,
    validate_table,
)
from MemNavData.patch_temporal_router import directional_patch_feature_names


def table():
    rows = []
    for candidate in range(2):
        rows.append({
            "session_id": "train/scene/episode/state/variant",
            "scene": "scene",
            "query_relative_path": "scene/query.jpg",
            "candidate_relative_path": f"scene/c{candidate}.jpg",
            "candidate_rank": candidate,
            "dino_cosine": 0.8 + 0.01 * candidate,
            "no_future": True,
        })
    return pd.DataFrame(rows)


def test_table_and_pair_universe(tmp_path):
    frame = table()
    validate_table(frame, expected_rows=2, expected_sessions=1,
                   expected_scenes=1, expected_candidates=2)
    for relative in ("scene/query.jpg", "scene/c0.jpg", "scene/c1.jpg"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    raw, paths, query, candidate = pair_universe(frame, tmp_path)
    assert len(raw) == len(paths) == 3
    assert np.array_equal(query, [2, 2])
    assert query.dtype == candidate.dtype == np.int32


def test_relations_are_finite_and_aligned():
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(3, 4, 6))
    tokens /= np.linalg.norm(tokens, axis=-1, keepdims=True)
    result = build_relations(
        tokens, np.asarray([0, 0]), np.asarray([1, 2]),
        np.asarray([0.8, 0.9]))
    assert result.shape == (2, len(directional_patch_feature_names()))
    assert np.isfinite(result).all()


@pytest.mark.parametrize("value", ("/x.jpg", "../x.jpg", "a/../x.jpg", ""))
def test_rejects_unsafe_relative_path(value):
    with pytest.raises(ValueError):
        safe_relative(value)


def test_future_frame_fails():
    frame = table()
    frame.loc[0, "no_future"] = False
    with pytest.raises(RuntimeError):
        validate_table(frame, expected_rows=2, expected_sessions=1,
                       expected_scenes=1, expected_candidates=2)
