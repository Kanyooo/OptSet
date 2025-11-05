"""Noise calibration utilities."""
from __future__ import annotations

import numpy as np


def calibrate_sigma_for_snr(y_clean: np.ndarray, m: int, snr: float) -> float:
    """Calibrate Gaussian noise standard deviation for target SNR.

    ``snr`` is defined as ``||y||^2 / (m * sigma^2)``.
    """
    power = float(np.sum(y_clean ** 2))
    return float(np.sqrt(power / (m * snr)))
