from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class TargetFunction:
    name: str

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
