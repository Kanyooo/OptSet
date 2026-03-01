from __future__ import annotations

import torch

from targets.base import TargetFunction


class StructuredCompositionTarget(TargetFunction):
    def __init__(self, d: int, seed: int, num_max_terms: int = 8, ridge: float = 1e-2):
        super().__init__("structured_composition")
        g = torch.Generator().manual_seed(seed)
        self.P = torch.randn(num_max_terms, d, generator=g)
        self.q = torch.randn(num_max_terms, generator=g)
        B = torch.randn(d, d, generator=g)
        self.Q = B.T @ B + ridge * torch.eye(d)
        self.As = torch.randn(2, d, d, generator=g)
        self.ds = torch.randn(2, d, generator=g)
        self.lam = torch.rand(2, generator=g) + 0.1

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        max_term = (x @ self.P.T + self.q).max(dim=-1).values
        quad = 0.5 * torch.einsum("bi,ij,bj->b", x, self.Q, x)
        cone = 0.0
        for j in range(self.As.shape[0]):
            cone = cone + self.lam[j] * torch.linalg.norm(x @ self.As[j].T + self.ds[j], dim=-1)
        return (max_term + quad + cone).unsqueeze(-1)
