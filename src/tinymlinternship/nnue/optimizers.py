"""Optimizers not shipped with torch.optim — currently Resilient Backpropagation."""

from __future__ import annotations

import torch


class Rprop(torch.optim.Optimizer):
    """Resilient Backpropagation (iRprop+, Riedmiller & Braun / Igel & Hüsken).

    Maintains a per-parameter step size ``Δ`` and adapts it only from the *sign*
    of the gradient, ignoring its magnitude. On a gradient sign flip — the sign
    that the previous update overshot the minimum (i.e. the loss went back up) —
    the parameter is **restored to its pre-update value** and ``Δ`` is shrunk,
    so the loss is forced back down:

    * ``sign(g) == sign(g_prev)`` → ``Δ = min(Δ · η+, Δ_max)``; remember the
      pre-update weight, then ``param ← param − sign(g) · Δ``.
    * ``sign(g) != sign(g_prev)`` → ``Δ = max(Δ · η-, Δ_min)``; ``param`` is
      **reverted** to the remembered pre-update weight (backtracking), and the
      gradient is zeroed so the next step starts fresh.
    * ``sign(g_prev) == 0`` → normal update ``param ← param − sign(g) · Δ``.

    This is a full-batch update rule: call ``step()`` once per epoch with the
    gradient accumulated over the whole dataset (the expert fine-tuner does
    exactly this — see ``runner._fine_tune_experts``). Mini-batch gradients are
    too noisy for the sign-based update and will cause constant sign flips.

    ``lr`` is the initial step size ``Δ0`` (Rprop has no learning-rate schedule;
    the cosine ``expert_lr_end`` decay is skipped for this optimizer).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50.0),
    ) -> None:
        if not 0.0 < float(etas[0]) < 1.0 < float(etas[1]):
            raise ValueError("etas must satisfy 0 < eta_minus < 1 < eta_plus")
        if not 0.0 < float(step_sizes[0]) < float(step_sizes[1]):
            raise ValueError("step_sizes must satisfy 0 < step_min < step_max")
        defaults = dict(lr=float(lr), etas=etas, step_sizes=step_sizes)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step_size"] = torch.full_like(p, group["lr"], memory_format=torch.preserve_format)
                state["prev_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["prev_weight"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eta_minus, eta_plus = group["etas"]
            step_min, step_max = group["step_sizes"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Rprop does not support sparse gradients")
                state = self.state[p]
                step_size = state["step_size"]
                prev_grad = state["prev_grad"]
                prev_weight = state["prev_weight"]

                product = grad * prev_grad
                grow = product > 0
                shrink = product < 0

                new_step = torch.where(
                    grow,
                    torch.clamp(step_size * eta_plus, min=step_min, max=step_max),
                    step_size,
                )
                new_step = torch.where(
                    shrink,
                    torch.clamp(step_size * eta_minus, min=step_min, max=step_max),
                    new_step,
                )
                step_size.copy_(new_step)

                delta = -torch.sign(grad) * step_size
                # iRprop+ backtracking: on a sign flip, revert to the weight
                # before the overshooting update; otherwise apply the update.
                new_weight = torch.where(shrink, prev_weight, p + delta)
                # Remember the pre-update weight for a potential revert next step
                # (kept unchanged on a flip so a single revert undoes one step).
                prev_weight.copy_(torch.where(shrink, prev_weight, p))
                p.copy_(new_weight)
                prev_grad.copy_(torch.where(shrink, torch.zeros_like(grad), grad))

        return loss

    @torch.no_grad()
    def shrink_step_sizes(self, factor: float | None = None) -> None:
        """Multiply every per-parameter step size by ``factor`` (default η-).

        Used for loss-based backtracking: when the full-batch loss increases,
        the caller reverts the weights and slows the whole optimizer down.
        """
        for group in self.param_groups:
            f = float(factor) if factor is not None else float(group["etas"][0])
            for p in group["params"]:
                state = self.state.get(p)
                if state is not None:
                    state["step_size"].mul_(f)

    @torch.no_grad()
    def reset_tracking(self) -> None:
        """Reset sign bookkeeping after an external weight revert.

        Clears ``prev_grad`` and re-snapshots ``prev_weight`` from the current
        parameter values so the next ``step()`` starts from a clean state.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state is not None:
                    state["prev_grad"].zero_()
                    state["prev_weight"].copy_(p.detach())
