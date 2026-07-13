import pytest

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
from scripts.dataset_converters.precompute_lingbot_features import validate_frame_capacity


def _trajectory(path, n_frames):
    return str(path), str(path / "rgb"), [f"{i}.jpg" for i in range(n_frames)]


def test_precompute_rejects_trajectories_beyond_rope_capacity(tmp_path):
    trajectories = [
        _trajectory(tmp_path / "fits", 2048),
        _trajectory(tmp_path / "too-long", 2049),
    ]

    with pytest.raises(ValueError, match=r"1 selected trajectories.*longest=2049"):
        validate_frame_capacity(trajectories, max_frame_num=2048, root_dirs=str(tmp_path))


def test_precompute_accepts_longest_trajectory_at_capacity(tmp_path):
    validate_frame_capacity(
        [_trajectory(tmp_path / "fits", 4096)],
        max_frame_num=4096,
        root_dirs=str(tmp_path),
    )


def test_dataset_rejects_missing_feature_cache_by_default(tmp_path):
    root = tmp_path / "raw"
    episode = root / "vln_n1" / "scene" / "episode_0002"
    (episode / "data" / "chunk-000").mkdir(parents=True)
    (episode / "videos" / "chunk-000" / "observation.images.rgb").mkdir(parents=True)
    (episode / "meta").mkdir()
    (episode / "data" / "chunk-000" / "episode_000000.parquet").touch()
    (episode / "meta" / "gen_meta.json").write_text("{}")

    with pytest.raises(RuntimeError, match=r"Incomplete MemNav feature coverage: 1.*episode_0002"):
        MemNav_Dataset(root, feature_root=tmp_path / "features")
