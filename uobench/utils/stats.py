"""Statistical helper functions used to calibrate synthetic data."""

from __future__ import annotations

from typing import Dict

import numpy as np


def calibrate_sigma_for_snr(y_clean: np.ndarray, m: int, snr: float) -> float:
    """Return ``sigma`` so that ``SNR = ||y||^2 / (m sigma^2)``."""

    power = float(np.sum(np.square(y_clean)))
    return float(np.sqrt(power / (max(m, 1) * max(snr, 1e-12))))


def estimate_snr(y: np.ndarray, y_clean: np.ndarray) -> float:
    """Empirically estimate the SNR of noisy observations."""

    noise = y - y_clean
    denom = max(1e-12, float(np.sum(noise**2)))
    return float(np.sum(y_clean**2) / denom)


def coherence(A: np.ndarray) -> float:
    """Return the mutual coherence ``max_{i≠j} |a_i^T a_j|`` of normalised columns."""

    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    normalized = A / col_norms
    gram = normalized.T @ normalized
    np.fill_diagonal(gram, 0.0)
    return float(np.max(np.abs(gram)))


def summarize_diagnostics(diag: Dict[str, float]) -> str:
    """Human readable summary string for diagnostic dictionaries."""

    return ", ".join(f"{k}={v:.3e}" for k, v in sorted(diag.items()))
