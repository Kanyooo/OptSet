"""Difficulty diagnostics computed for generated instances."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.linalg import cond, smin, spectral_extrema
from ..utils.stats import coherence, estimate_snr


def _spd_metrics(Q: np.ndarray) -> Dict[str, float]:
    lam_min, lam_max = spectral_extrema(Q)
    return {
        "lambda_min": float(lam_min),
        "lambda_max": float(lam_max),
        "cond_Q": float(lam_max / lam_min) if lam_min > 0 else float("inf"),
    }


def _matrix_metrics(A: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "cond_A": cond(A),
        "smin_A": smin(A),
        "coherence": coherence(A),
        "norm_A": float(np.linalg.norm(A, 2)),
    }
    return metrics


def compute(problem_id: str, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute diagnostic quantities tailored to ``problem_id``."""

    diag: Dict[str, float] = {}

    if "Q" in arrays:
        diag.update(_spd_metrics(arrays["Q"]))
    if "H" in arrays:
        lam_min, lam_max = spectral_extrema(arrays["H"])
        diag.update({
            "lambda_min": float(lam_min),
            "lambda_max": float(lam_max),
            "neg_ratio": float(np.mean(np.linalg.eigvalsh(0.5 * (arrays["H"] + arrays["H"].T)) < 0)),
        })
    if "A" in arrays:
        diag.update(_matrix_metrics(arrays["A"]))
    if {"y", "y_clean"}.issubset(arrays.keys()):
        diag["snr_est"] = estimate_snr(arrays["y"], arrays["y_clean"])
    if problem_id in {"A5_TRS"}:
        H = arrays["H"]
        g = arrays["g"]
        eigs, vecs = np.linalg.eigh(0.5 * (H + H.T))
        vmin = vecs[:, 0]
        cos = float(np.clip(abs(np.dot(g, vmin)) / ((np.linalg.norm(g) + 1e-12) * np.linalg.norm(vmin)), 0.0, 1.0))
        diag["angle_vmin"] = float(np.degrees(np.arccos(cos)))
    if problem_id == "C2_LCP":
        diag["delta"] = float(arrays.get("delta", 0.0))
    if problem_id == "D3_SDP":
        L = arrays["L"]
        lam = np.sort(np.linalg.eigvalsh(L))
        if lam.size > 1:
            diag["lambda_2"] = float(lam[1])
        diag["trace_L"] = float(np.trace(L))
    if problem_id == "C3_MPCC":
        b = arrays["b"]
        diag["min_b"] = float(np.min(b))
        diag["max_b"] = float(np.max(b))
    if problem_id == "D1_SOCP":
        diag["radius"] = float(arrays.get("R", 1.0))
    return diag
