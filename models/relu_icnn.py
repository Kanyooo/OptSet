from __future__ import annotations

import torch
from models.icnn_backbone import ICNNBackbone, ICNNConfig


class ReLUICNN(ICNNBackbone):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, passthrough: bool = True):
        super().__init__(ICNNConfig(input_dim, hidden_dim, depth, passthrough), activation=torch.relu)
