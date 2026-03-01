from __future__ import annotations

import torch

from targets.base import TargetFunction


class QuadraticTarget(TargetFunction):
    def __init__(self, d: int, seed: int, ridge: float = 1e-2):
        super().__init__("quadratic")
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(d, d, generator=g)
        self.Q = B.T @ B + ridge * torch.eye(d)
        self.b = torch.randn(d, generator=g)
        self.c = torch.randn(1, generator=g)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        quad = 0.5 * torch.einsum("bi,ij,bj->b", x, self.Q, x)
        return (quad + x @ self.b + self.c).unsqueeze(-1)
