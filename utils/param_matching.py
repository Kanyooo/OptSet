from __future__ import annotations

from typing import Callable

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_width_for_budget(
    model_builder: Callable[..., nn.Module],
    input_dim: int,
    depth: int,
    budget: int,
    tolerance: float = 0.05,
    min_width: int = 4,
    max_width: int = 4096,
    **kwargs,
):
    """Binary-ish search for width s.t. parameter count is near budget."""
    lo, hi = min_width, max_width
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = model_builder(input_dim=input_dim, hidden_dim=mid, depth=depth, **kwargs)
        n_params = count_parameters(m)
        rel_gap = abs(n_params - budget) / max(1, budget)
        if best is None or rel_gap < best[2]:
            best = (mid, n_params, rel_gap)
        if n_params < budget:
            lo = mid + 1
        elif n_params > budget:
            hi = mid - 1
        else:
            break
        if best[2] <= tolerance:
            break
    return {"width": best[0], "n_params": best[1], "rel_gap": best[2]}
