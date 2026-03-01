from __future__ import annotations

from torch import nn

from models.relu_icnn import ReLUICNN
from models.primitives import QuadraticPrimitives


class QuadICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, passthrough: bool = True, num_quad_terms: int = 1, quad_term_dim: int | None = None):
        super().__init__()
        self.backbone = ReLUICNN(input_dim, hidden_dim, depth, passthrough)
        self.quad = QuadraticPrimitives(input_dim, num_quad_terms, quad_term_dim)

    def forward(self, x):
        return self.backbone(x) + self.quad(x)
