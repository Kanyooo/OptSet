"""Difficulty diagnostics for generated problems."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.linalg import cond, smin
from ..utils.stats import coherence


def compute(problem_id: str, arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    diag: Dict[str, float] = {}
    if "Q" in arrays:
        Q = arrays["Q"]
        eigs = np.linalg.eigvalsh(Q)
        diag["condQ"] = float(np.max(eigs) / np.min(np.abs(eigs))) if np.min(np.abs(eigs)) > 0 else float("inf")
        diag["lambda_min"] = float(np.min(eigs))
    if "A" in arrays:
        A = arrays["A"]
        diag["condA"] = cond(A)
        diag["sminA"] = smin(A)
        diag["coherence"] = coherence(A)
    if "H" in arrays:
        H = arrays["H"]
        eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
        neg = np.sum(eigs < 0)
        diag["lambda_min"] = float(np.min(eigs))
        diag["neg_ratio"] = float(neg / len(eigs))
    if "noise_sigma" in arrays:
        diag["noise_sigma"] = float(arrays["noise_sigma"])
    if "y" in arrays and "y_clean" in arrays:
        y = arrays["y_clean"]
        noise = arrays["y"] - y
        power = float(np.sum(y**2))
        sigma = float(np.sum(noise**2) / max(1, len(noise)))
        diag["snr_est"] = power / sigma if sigma > 0 else float("inf")
    if problem_id == "A5_TRS":
        g = arrays["g"]
        H = arrays["H"]
        eigs, vecs = np.linalg.eigh(0.5 * (H + H.T))
        vmin = vecs[:, 0]
        angle = np.degrees(np.arccos(np.clip(np.abs(g @ vmin) / (np.linalg.norm(g) * np.linalg.norm(vmin)), 0, 1)))
        diag["angle"] = float(angle)
    if problem_id == "D3_SDP":
        L = arrays["L"]
        diag["lambda2"] = float(np.sort(np.linalg.eigvalsh(L))[1])
    return diag
