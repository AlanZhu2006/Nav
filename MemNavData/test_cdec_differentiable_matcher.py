import torch

from MemNavData.cdec_differentiable_matcher import (
    DifferentiablePatchMatcher,
    listwise_positive_loss,
)


def test_matcher_shapes_and_gradients():
    torch.manual_seed(4)
    model = DifferentiablePatchMatcher(
        token_dim=12, projection_dim=6, hidden_dim=16,
        grid_size=2, dropout=0.0)
    tokens = torch.randn(5, 4, 12)
    output = model(
        tokens, torch.tensor([0, 0, 1]), torch.tensor([2, 3, 4]),
        torch.tensor([0.8, 0.7, 0.9]))
    assert output["task_match_logits"].shape == (3,)
    assert output["certificate_pass_logits"].shape == (3,)
    sum(value.sum() for value in output.values()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_listwise_loss_accepts_any_positive():
    labels = torch.tensor([[True, False, True], [False, False, False]])
    good = torch.tensor([[8.0, -5.0, 7.0], [0.0, 0.0, 0.0]])
    bad = torch.tensor([[-5.0, 8.0, -5.0], [0.0, 0.0, 0.0]])
    assert listwise_positive_loss(good, labels) < 1e-3
    assert listwise_positive_loss(bad, labels) > 5.0


def test_bank_reindexing_does_not_change_pair_output():
    torch.manual_seed(2)
    model = DifferentiablePatchMatcher(
        token_dim=8, projection_dim=4, hidden_dim=16,
        grid_size=2, dropout=0.0).eval()
    tokens = torch.randn(2, 4, 8)
    forward = model(tokens, torch.tensor([0]), torch.tensor([1]), torch.tensor([0.5]))
    reindexed = model(tokens.flip(0), torch.tensor([1]), torch.tensor([0]), torch.tensor([0.5]))
    assert torch.allclose(
        forward["task_match_logits"], reindexed["task_match_logits"], atol=1e-6)
