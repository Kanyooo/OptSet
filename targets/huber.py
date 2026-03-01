from __future__ import annotations

import torch

from targets.base import TargetFunction


class HuberTarget(TargetFunction):
    def __init__(self, d: int, seed: int, delta: float = 1.0):
        super().__init__("huber")
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(d, d, generator=g)
        self.b = torch.randn(d, generator=g)
        self.delta = delta

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.A.T + self.b
        abs_z = z.abs()
        quadratic = 0.5 * (z**2)
        linear = self.delta * (abs_z - 0.5 * self.delta)
        hub = torch.where(abs_z <= self.delta, quadratic, linear)
        return hub.sum(dim=-1, keepdim=True)
