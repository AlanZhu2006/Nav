from argparse import Namespace

from MemNavData.train_monocular_geometry_adapter import (
    fixed_scene_split,
    gate_decision,
)


def _metric(token, epsilon, spearman, top1):
    return {
        "token_cosine_error": token,
        "token_smooth_l1": token,
        "epsilon_mse": epsilon,
        "epsilon_cosine_error": epsilon,
        "critic_mse": 1.0,
        "critic_spearman": spearman,
        "critic_top1_agreement": top1,
        "critic_top2_overlap": top1,
    }


def test_scene_split_is_deterministic_and_disjoint():
    scenes = [f"scene_{index:02d}" for index in range(40)]
    first = fixed_scene_split(scenes, 0.2, "salt")
    second = fixed_scene_split(list(reversed(scenes)), 0.2, "salt")
    assert first == second
    assert len(first[0]) == 32
    assert len(first[1]) == 8
    assert not set(first[0]) & set(first[1])


def test_gate_refuses_underpowered_smoke():
    metrics = {
        "zero_depth_tokens": _metric(1.0, 1.0, 0.0, 0.0),
        "raw_depth_tokens": _metric(0.1, 0.1, 1.0, 1.0),
        "adapter": _metric(0.1, 0.1, 1.0, 1.0),
    }
    gate = gate_decision(
        metrics, 3, 3, Namespace(min_gate_samples=32, min_gate_scenes=4)
    )
    assert not gate["authorized"]
    assert gate["reason"] == "underpowered_diagnostic_no_gate_decision"


def test_gate_chooses_adapter_only_for_functional_gain():
    metrics = {
        "zero_depth_tokens": _metric(1.0, 1.0, 0.10, 0.25),
        "raw_depth_tokens": _metric(0.95, 0.95, 0.10, 0.25),
        "adapter": _metric(0.70, 0.70, 0.30, 0.50),
    }
    gate = gate_decision(
        metrics, 64, 8, Namespace(min_gate_samples=32, min_gate_scenes=4)
    )
    assert gate["authorized"]
    assert gate["adapter_qualifies"]
    assert not gate["raw_qualifies"]
    assert gate["choice"] == "latent_adapter"
