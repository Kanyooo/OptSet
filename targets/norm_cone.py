from __future__ import annotations

import torch

from targets.base import TargetFunction


class NormConeTarget(TargetFunction):
    def __init__(self, d: int, seed: int, k: int | None = None):
        super().__init__("norm_cone")
        k = k or d
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(k, d, generator=g)
        self.delta = torch.randn(k, generator=g)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x @ self.A.T + self.delta, dim=-1, keepdim=True)
