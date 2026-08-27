from MemNavData.fit_pi3x_spatial_reliability_deployment import (
    deployment_scene_splits,
)


def test_deployment_splits_are_disjoint_and_cover_every_scene():
    scenes = [f"scene_{index}" for index in range(40)]
    splits = deployment_scene_splits(scenes, 4)
    assert len(splits) == 4
    assert all(len(fit) == 30 and len(calibration) == 10 for fit, calibration in splits)
    assert all(not (fit & calibration) for fit, calibration in splits)
    assert all(fit | calibration == set(scenes) for fit, calibration in splits)
    calibration_scenes = [
        scene for _, calibration in splits for scene in calibration
    ]
    assert set(calibration_scenes) == set(scenes)
    assert len(calibration_scenes) == len(set(calibration_scenes))
