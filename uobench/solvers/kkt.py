"""KKT residual utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np


def residual_qp(Q: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(Q @ x - b))


def residual_eq(A: np.ndarray, x: np.ndarray, d: np.ndarray) -> float:
    return float(np.linalg.norm(A @ x - d))


def residual_box(x: np.ndarray, l: np.ndarray, u: np.ndarray) -> float:
    viol = np.maximum(0, l - x) + np.maximum(0, x - u)
    return float(np.linalg.norm(viol))
