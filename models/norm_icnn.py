from __future__ import annotations

from torch import nn

from models.relu_icnn import ReLUICNN
from models.primitives import NormPrimitives


class NormICNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, passthrough: bool = True, num_norm_terms: int = 1, norm_term_dim: int | None = None):
        super().__init__()
        self.backbone = ReLUICNN(input_dim, hidden_dim, depth, passthrough)
        self.norm = NormPrimitives(input_dim, num_norm_terms, norm_term_dim)

    def forward(self, x):
        return self.backbone(x) + self.norm(x)
