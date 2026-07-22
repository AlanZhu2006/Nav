import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / 'scripts' / 'eval' / 'diag_memnav_route_gate_oracle.py'
)
SPEC = importlib.util.spec_from_file_location('diag_memnav_route_gate_oracle', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_threshold_gate_only_suppresses_oracle_hard_rows():
    gate = torch.tensor([0.2, 0.7, 0.9])
    angle = torch.tensor([10.0, 45.0, 100.0])
    actual = MODULE.apply_gate_strategy(gate, angle, 'zero_ge_45')
    torch.testing.assert_close(actual, torch.tensor([0.2, 0.0, 0.0]))


def test_cosine_gate_is_bounded_and_monotonic():
    gate = torch.ones(3)
    angle = torch.tensor([0.0, 90.0, 180.0])
    half = MODULE.apply_gate_strategy(gate, angle, 'cosine_half')
    torch.testing.assert_close(half, torch.tensor([1.0, 0.5, 0.0]), atol=1e-6, rtol=0)
    positive = MODULE.apply_gate_strategy(gate, angle, 'cosine_positive')
    torch.testing.assert_close(positive, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0)
