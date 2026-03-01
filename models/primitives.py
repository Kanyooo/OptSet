from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QuadraticPrimitives(nn.Module):
    """sum_h alpha_h/2 * ||B_h x + e_h||^2 with alpha_h>=0."""

    def __init__(self, input_dim: int, num_terms: int = 1, term_dim: int | None = None):
        super().__init__()
        term_dim = term_dim or input_dim
        self.B = nn.Parameter(torch.randn(num_terms, term_dim, input_dim) * 0.05)
        self.e = nn.Parameter(torch.zeros(num_terms, term_dim))
        self.raw_alpha = nn.Parameter(torch.zeros(num_terms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bx = torch.einsum("hkd,bd->bhk", self.B, x) + self.e.unsqueeze(0)
        norm_sq = (bx**2).sum(dim=-1)
        alpha = F.softplus(self.raw_alpha)
        return 0.5 * (norm_sq * alpha.unsqueeze(0)).sum(dim=-1, keepdim=True)


class NormPrimitives(nn.Module):
    """sum_g lambda_g * ||A_g x + d_g|| with lambda_g>=0."""

    def __init__(self, input_dim: int, num_terms: int = 1, term_dim: int | None = None):
        super().__init__()
        term_dim = term_dim or input_dim
        self.A = nn.Parameter(torch.randn(num_terms, term_dim, input_dim) * 0.05)
        self.d = nn.Parameter(torch.zeros(num_terms, term_dim))
        self.raw_lambda = nn.Parameter(torch.zeros(num_terms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ax = torch.einsum("gkd,bd->bgk", self.A, x) + self.d.unsqueeze(0)
        norms = torch.linalg.norm(ax, dim=-1)
        lam = F.softplus(self.raw_lambda)
        return (norms * lam.unsqueeze(0)).sum(dim=-1, keepdim=True)
