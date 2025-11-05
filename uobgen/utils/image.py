"""Image construction helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def make_blocks_2d(height: int, width: int, num_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a piecewise-constant image with ``num_blocks`` rectangular regions."""
    img = np.zeros((height, width), dtype=float)
    blocks = max(1, int(num_blocks))
    for _ in range(blocks):
        h = int(rng.integers(max(1, height // 8), max(2, height // 2)))
        w = int(rng.integers(max(1, width // 8), max(2, width // 2)))
        r = int(rng.integers(0, max(1, height - h + 1)))
        c = int(rng.integers(0, max(1, width - w + 1)))
        value = float(rng.uniform(-1, 1))
        img[r : r + h, c : c + w] = value
    return img


def build_tv_gradients(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return forward-difference edge lists for TV regularization."""
    dx_edges = []
    dy_edges = []
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if j + 1 < width:
                dx_edges.append((idx, idx + 1))
            if i + 1 < height:
                dy_edges.append((idx, idx + width))
    return np.array(dx_edges, dtype=int), np.array(dy_edges, dtype=int)
