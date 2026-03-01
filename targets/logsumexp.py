from __future__ import annotations

import torch

from targets.base import TargetFunction


class LogSumExpTarget(TargetFunction):
    def __init__(self, d: int, seed: int, num_terms: int = 16, tau: float = 0.5):
        super().__init__("logsumexp")
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(num_terms, d, generator=g)
        self.b = torch.randn(num_terms, generator=g)
        self.tau = tau

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        logits = (x @ self.A.T + self.b) / self.tau
        return (self.tau * torch.logsumexp(logits, dim=-1)).unsqueeze(-1)
