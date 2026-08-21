import torch

from MemNavData.certificate_distilled_compass import (
    CertificateDistilledCompass,
    cdec_loss,
    set_valued_task_loss,
)


def test_permutation_equivariance_and_null_invariance():
    torch.manual_seed(3)
    model = CertificateDistilledCompass(
        5, hidden_dim=16, heads=4, layers=1, dropout=0.0).eval()
    features = torch.randn(2, 4, 5)
    permutation = torch.tensor([2, 0, 3, 1])
    first = model(features)
    second = model(features[:, permutation])
    assert torch.allclose(
        first["task_logits"][:, permutation],
        second["task_logits"][:, :-1], atol=1e-6)
    assert torch.allclose(
        first["task_logits"][:, -1], second["task_logits"][:, -1], atol=1e-6)


def test_set_valued_loss_rewards_either_positive_and_null():
    positive = torch.tensor([[True, False], [False, False]])
    mask = torch.tensor([True, True])
    good = torch.tensor([[8.0, -2.0, -2.0], [-2.0, -2.0, 8.0]])
    bad = torch.tensor([[-2.0, 8.0, -2.0], [8.0, -2.0, -2.0]])
    assert set_valued_task_loss(good, positive, mask) < 1e-3
    assert set_valued_task_loss(bad, positive, mask) > 5.0


def test_all_heads_receive_gradient():
    torch.manual_seed(1)
    model = CertificateDistilledCompass(
        6, hidden_dim=16, heads=4, layers=1, dropout=0.0)
    output = model(torch.randn(3, 4, 6))
    loss = cdec_loss(
        output,
        positive_candidates=torch.tensor([
            [True, False, False, False],
            [False, False, False, False],
            [False, True, False, False],
        ]),
        task_mask=torch.tensor([True, True, True]),
        session_weight=torch.ones(3),
        certificate_pass=torch.tensor([
            [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0],
        ]),
        teacher_top_index=torch.tensor([0, 2, 1]),
        pass_positive_weight=torch.tensor(2.0),
        lambda_pass=0.25,
        lambda_rank=0.1,
    )
    loss.total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
