"""Random number utilities ensuring reproducibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RNG:
    """Wrapper around :class:`numpy.random.Generator` with convenience helpers."""

    seed: int
    bit_generator: str = "PCG64"
    _generator: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self._generator is None:
            bitgen_cls = getattr(np.random, self.bit_generator, np.random.PCG64)
            self._generator = np.random.Generator(bitgen_cls(self.seed))

    @property
    def generator(self) -> np.random.Generator:
        return self._generator  # type: ignore[return-value]

    def spawn(self, offset: int) -> "RNG":
        """Create a new RNG with deterministic offset."""
        new_seed = int((self.seed + offset) % (2**63 - 1))
        return RNG(new_seed, self.bit_generator)

    def normal(self, *shape: int, scale: float = 1.0) -> np.ndarray:
        return scale * self.generator.normal(size=shape)

    def uniform(self, *shape: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        return self.generator.uniform(low, high, size=shape)

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return self.generator.integers(low, high, size=size)
