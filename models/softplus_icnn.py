from __future__ import annotations

import torch
import torch.nn.functional as F
from models.icnn_backbone import ICNNBackbone, ICNNConfig


class SoftplusICNN(ICNNBackbone):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, passthrough: bool = True):
        super().__init__(ICNNConfig(input_dim, hidden_dim, depth, passthrough), activation=F.softplus)
