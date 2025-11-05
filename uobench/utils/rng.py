"""Random-number helper utilities for :mod:`uobench`.

This module centralises the random-number handling strategy for the benchmark.
All randomness flows through :class:`RNG`, a tiny immutable wrapper around
``numpy.random.SeedSequence`` capable of spawning statistically independent
sub-streams.  The indirection ensures that every dataset generator can accept a
single integer ``seed`` while the CLI orchestrates reproducible suites of
instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RNG:
    """Factory of reproducible :class:`numpy.random.Generator` objects.

    Parameters
    ----------
    seed:
        Base entropy for the generator.  The wrapper itself is lightweight and
        can be copied freely; each call to :meth:`generator` yields a *fresh*
        ``numpy.random.Generator`` instance so that local state is not shared
        across callers.
    """

    seed: int

    def generator(self) -> np.random.Generator:
        """Return a new generator instance initialised by ``seed``."""

        seq = np.random.SeedSequence(int(self.seed))
        return np.random.default_rng(seq)

    def spawn(self, *counters: Iterable[int] | int) -> "RNG":
        """Create a child :class:`RNG` with deterministic entropy.

        ``counters`` are hashed into a ``SeedSequence`` via
        :meth:`SeedSequence.spawn`.  The resulting instance therefore shares the
        global entropy while being statistically independent from other spawned
        sequences.  The API accepts either a single integer or an iterable of
        integers.
        """

        if len(counters) == 1 and isinstance(counters[0], int):
            tokens = [int(counters[0])]
        else:
            tokens = [int(c) for c in counters]
        child = np.random.SeedSequence([int(self.seed), *tokens])
        return RNG(seed=int(child.entropy))
