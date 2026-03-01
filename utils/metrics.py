from __future__ import annotations

import torch


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((y_hat - y) ** 2).item()


def relative_error(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.norm((y - y_hat).flatten())
    den = torch.linalg.norm(y.flatten()) + eps
    return (num / den).item()
