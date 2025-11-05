"""Statistical helpers for SNR and coherence diagnostics."""

from __future__ import annotations

from typing import Dict

import numpy as np


def calibrate_sigma_for_snr(y_clean: np.ndarray, m: int, snr: float) -> float:
    """Compute noise sigma so that SNR = ||y||^2 / (m sigma^2)."""
    power = float(np.sum(y_clean**2))
    return np.sqrt(power / (m * snr))


def coherence(A: np.ndarray) -> float:
    col_norms = np.linalg.norm(A, axis=0)
    normalized = A / col_norms
    gram = normalized.T @ normalized
    off_diag = gram - np.eye(gram.shape[0])
    return float(np.max(np.abs(off_diag)))


def summarize_diagnostics(diag: Dict[str, float]) -> str:
    """Create a simple textual summary of key diagnostics."""
    parts = [f"{k}={v:.3e}" for k, v in diag.items()]
    return ", ".join(parts)
