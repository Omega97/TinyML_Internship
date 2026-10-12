"""Resilient Backpropagation optimizer."""

from __future__ import annotations

import pytest
import torch

from tinymlinternship.nnue.optimizers import Rprop


def test_rprop_minimizes_quadratic():
    torch.manual_seed(0)
    w = torch.tensor([4.0, -4.0], requires_grad=True)
    opt = Rprop([w], lr=0.1)
    c = torch.tensor([1.0, -1.0])
    for _ in range(300):
        opt.zero_grad()
        loss = ((w - c) ** 2).sum()
        loss.backward()
        opt.step()
    assert torch.allclose(w, c, atol=0.1)


def test_rprop_step_size_stays_bounded():
    w = torch.tensor([0.0], requires_grad=True)
    opt = Rprop([w], lr=0.1, step_sizes=(1e-6, 50.0))
    for _ in range(500):
        opt.zero_grad()
        (w * w).sum().backward()
        opt.step()
    ss = float(opt.state[w]["step_size"])
    assert 1e-6 - 1e-9 <= ss <= 50.0 + 1e-9


def test_rprop_rejects_invalid_hyperparams():
    w = torch.tensor([0.0], requires_grad=True)
    with pytest.raises(ValueError):
        Rprop([w], etas=(1.5, 1.2))
    with pytest.raises(ValueError):
        Rprop([w], step_sizes=(50.0, 1e-6))


def test_rprop_backtracks_on_sign_flip():
    """A gradient sign flip must revert the weight and shrink the step size."""
    w = torch.tensor([1.0], requires_grad=True)
    opt = Rprop([w], lr=0.1, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0))

    w.grad = torch.tensor([2.0])  # neutral first step (prev_grad == 0)
    opt.step()
    w1 = w.detach().clone()  # 1.0 - 0.1 = 0.9

    w.grad = torch.tensor([2.0])  # same sign → grow, move more
    opt.step()
    w2 = w.detach().clone()  # 0.9 - 0.12 = 0.78

    w.grad = torch.tensor([-2.0])  # sign flip → backtrack to w1, shrink Δ
    opt.step()

    assert torch.allclose(w.detach(), w1, atol=1e-6), "must revert to pre-update weight"
    assert torch.allclose(opt.state[w]["step_size"], torch.tensor(0.06), atol=1e-6)


def test_rprop_loss_backtracking_keeps_loss_monotone():
    """Full-batch Rprop + loss-based revert must keep the loss non-increasing."""
    w = torch.tensor([4.0], requires_grad=True)
    opt = Rprop([w], lr=0.1)

    def loss():
        return (w * w).sum()

    prev_loss = float(loss().detach())
    prev_state = w.detach().clone()
    losses = [prev_loss]
    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        loss().backward()
        opt.step()
        cur = float(loss().detach())
        if cur > prev_loss:  # loss increased → backtrack + slow down
            w.data.copy_(prev_state)
            opt.shrink_step_sizes()
            opt.reset_tracking()
            cur = prev_loss
        else:
            prev_loss = cur
            prev_state = w.detach().clone()
        losses.append(cur)

    for a, b in zip(losses, losses[1:]):
        assert b <= a + 1e-6, "loss must never increase with backtracking"
    assert losses[-1] < 0.01, f"did not converge: {losses[-1]}"
    assert opt.state[w]["step_size"] < 0.1, "step size must have shrunk on reverts"
