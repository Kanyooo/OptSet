from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


ActivationFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ICNNConfig:
    input_dim: int
    hidden_dim: int
    depth: int
    passthrough: bool = True


class ICNNBackbone(nn.Module):
    """Convex backbone: z_0=0, z_l=phi(W_l x + U_l z_{l-1}+b_l), with U_l>=0, c>=0."""

    def __init__(self, cfg: ICNNConfig, activation: ActivationFn):
        super().__init__()
        self.cfg = cfg
        self.activation = activation

        self.w_layers = nn.ModuleList()
        self.raw_u_layers = nn.ParameterList()
        self.b_layers = nn.ParameterList()
        in_dim = cfg.input_dim
        h = cfg.hidden_dim
        for _ in range(cfg.depth):
            self.w_layers.append(nn.Linear(in_dim, h, bias=False))
            self.raw_u_layers.append(nn.Parameter(torch.empty(h, h)))
            self.b_layers.append(nn.Parameter(torch.zeros(h)))

        self.raw_c = nn.Parameter(torch.empty(h))
        self.v = nn.Parameter(torch.zeros(cfg.input_dim))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in self.w_layers:
            nn.init.xavier_uniform_(w.weight)
        for u in self.raw_u_layers:
            nn.init.normal_(u, mean=0.0, std=0.02)
        for b in self.b_layers:
            nn.init.zeros_(b)
        nn.init.normal_(self.raw_c, mean=0.0, std=0.02)
        nn.init.zeros_(self.v)
        nn.init.zeros_(self.b0)

    @staticmethod
    def _pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], self.cfg.hidden_dim, device=x.device, dtype=x.dtype)
        for w_layer, raw_u, b in zip(self.w_layers, self.raw_u_layers, self.b_layers):
            u = self._pos(raw_u)
            z = self.activation(w_layer(x) + z @ u.T + b)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_hidden(x)
        c = self._pos(self.raw_c)
        out = z @ c
        if self.cfg.passthrough:
            out = out + x @ self.v + self.b0
        return out.unsqueeze(-1)
