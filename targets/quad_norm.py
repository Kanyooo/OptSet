from __future__ import annotations

import torch

from targets.base import TargetFunction


class QuadNormTarget(TargetFunction):
    def __init__(self, d: int, seed: int, ridge: float = 1e-2):
        super().__init__("quadratic_plus_norm")
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(d, d, generator=g)
        self.Q = B.T @ B + ridge * torch.eye(d)
        self.b = torch.randn(d, generator=g)
        self.c = torch.randn(1, generator=g)
        self.A = torch.randn(d, d, generator=g)
        self.delta = torch.randn(d, generator=g)
        self.lam = torch.rand(1, generator=g).item() + 0.1

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        quad = 0.5 * torch.einsum("bi,ij,bj->b", x, self.Q, x)
        norm = self.lam * torch.linalg.norm(x @ self.A.T + self.delta, dim=-1)
        return (quad + norm + x @ self.b + self.c).unsqueeze(-1)
