import json
import math

import numpy as np
import pytest

from MemNavData.cdec_pairwise_runtime import (
    CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION,
    CDECPairwiseRanker,
    circular_direction_summary,
    pad_dino_image_batch,
    pool_dino_patch_tokens,
    relation_feature_matrix,
)
from MemNavData.patch_temporal_router import (
    directional_patch_feature_names,
)


def artifact_payload(*, approved=False):
    count = len(directional_patch_feature_names())
    return {
        "schema_version": CDEC_PAIRWISE_RUNTIME_SCHEMA_VERSION,
        "deployment_approved": approved,
        "runtime_semantics": {
            "authority": "rank_frozen_causal_shortlist_only",
            "activation_authority": "independent_atomic_pnp_certificate",
            "fallback": "native_imagegoal",
            "score_calibration": "uncalibrated_pairwise_utility",
        },
        "model": {
            "feature_names": list(directional_patch_feature_names()),
            "coefficient": [1.0] + [0.0] * (count - 1),
            "mean": [0.0] * count,
            "scale": [1.0] * count,
            "intercept": 0.0,
            "patch_grid_size": 8,
            "dino_inference_batch_size": 16,
            "relation_storage_dtype": "float32",
        },
    }


def write_artifact(tmp_path, *, approved=False):
    path = tmp_path / "ranker.json"
    path.write_text(json.dumps(artifact_payload(approved=approved)))
    return path


def test_unapproved_artifact_is_explicit_and_cannot_activate(tmp_path):
    path = write_artifact(tmp_path)
    with pytest.raises(PermissionError):
        CDECPairwiseRanker(path)
    ranker = CDECPairwiseRanker(path, allow_unapproved=True)
    features = np.zeros((2, len(directional_patch_feature_names())))
    features[:, 0] = [0.2, 0.8]
    result = ranker.rank_features(features, [7, 3])
    assert result["selected_anchor"] == 3
    assert result["activation_authorized"] is False
    assert result["activation_authority"] == (
        "independent_atomic_pnp_certificate")
    assert math.isclose(sum(result["within_shortlist_mass"]), 1.0)
    with pytest.raises(ValueError):
        ranker.rank_features(features, [7.5, 3])


def test_artifact_symlink_is_rejected(tmp_path):
    path = write_artifact(tmp_path, approved=True)
    link = tmp_path / "ranker-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        CDECPairwiseRanker(link)


def test_score_reproduces_zero_intercept_standardized_linear_fit(tmp_path):
    payload = artifact_payload(approved=True)
    count = len(directional_patch_feature_names())
    payload["model"]["coefficient"] = np.linspace(-1.0, 1.0, count).tolist()
    payload["model"]["mean"] = np.linspace(0.0, 0.2, count).tolist()
    payload["model"]["scale"] = np.linspace(0.5, 1.5, count).tolist()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload))
    ranker = CDECPairwiseRanker(path)
    rng = np.random.default_rng(9)
    features = rng.normal(size=(4, count))
    expected = ((features - ranker.mean) / ranker.scale) @ ranker.coefficient
    np.testing.assert_allclose(ranker.score_features(features), expected,
                               rtol=0, atol=1e-12)


def test_patch_pooling_and_relation_features_are_aligned():
    import torch

    torch.manual_seed(5)
    raw = torch.randn(3, 1369, 12)
    pooled = pool_dino_patch_tokens(raw)
    assert pooled.shape == (3, 64, 12)
    assert pooled.dtype == np.float16
    features = relation_feature_matrix(
        pooled[0], pooled[1:], [0.7, 0.9])
    assert features.shape == (2, len(directional_patch_feature_names()))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_dino_image_batch_is_padded_to_frozen_training_shape():
    import torch

    images = torch.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2)
    padded, count = pad_dino_image_batch(images)
    assert count == 3
    assert padded.shape == (16, 2, 2, 2)
    assert torch.equal(padded[:3], images)
    assert torch.equal(padded[3:], images[-1:].expand(13, -1, -1, -1))


def test_circular_summary_exposes_ambiguity_but_never_authorizes():
    concentrated = circular_direction_summary(
        [0.9, 0.1], [[1.0, 0.0], [0.0, 1.0]])
    assert concentrated["resultant_length"] > 0.7
    assert concentrated["execution_authorized"] is False
    cancelled = circular_direction_summary(
        [0.5, 0.5], [[1.0, 0.0], [-1.0, 0.0]])
    assert cancelled["mean_unit_bearing_forward_left"] is None
    assert cancelled["resultant_length"] == pytest.approx(0.0)
