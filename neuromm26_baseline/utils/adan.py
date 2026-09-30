"""Adan optimizer (Adaptive Nesterov Momentum) — ported from HMS muku solution.

Reference:
  Xie et al., "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing
  Deep Models", arXiv:2208.06677, 2022.
"""

from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


class Adan(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.2, max_grad_norm=0.0, no_prox=False):
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"Invalid Max grad norm: {max_grad_norm}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        for i, b in enumerate(betas):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"Invalid beta {i}: {b}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm, no_prox=no_prox)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("no_prox", False)

    @torch.no_grad()
    def step(self):
        if self.defaults["max_grad_norm"] > 0:
            device = self.param_groups[0]["params"][0].device
            global_grad_norm = torch.zeros(1, device=device)
            max_grad_norm = torch.tensor(self.defaults["max_grad_norm"], device=device)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        global_grad_norm.add_(p.grad.pow(2).sum())
            global_grad_norm = torch.sqrt(global_grad_norm)
            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + self.defaults["eps"]), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            group["step"] = group.get("step", 0) + 1
            bc1 = 1.0 - beta1 ** group["step"]
            bc2 = 1.0 - beta2 ** group["step"]
            bc3 = 1.0 - beta3 ** group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)
                grad = p.grad.mul_(clip_global_grad_norm)
                if "pre_grad" not in state or group["step"] == 1:
                    state["pre_grad"] = grad
                copy_grad = grad.clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_diff = state["exp_avg_diff"]
                diff = grad - state["pre_grad"]
                update = grad + beta2 * diff

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bc3)).add_(group["eps"])
                update = (exp_avg / bc1 + beta2 * exp_avg_diff / bc2).div_(denom)

                if group["no_prox"]:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                else:
                    p.add_(update, alpha=-group["lr"])
                    p.data.div_(1 + group["lr"] * group["weight_decay"])

                state["pre_grad"] = copy_grad
