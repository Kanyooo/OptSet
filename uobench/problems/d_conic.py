"""Conic problem generators (D1–D3)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def generate_d1_socp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 60))
    n = int(knobs.get("n", 20))
    R = float(knobs.get("R", 1.0))
    snr = float(knobs.get("snr", 10))
    A = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    y_clean = A @ x_true
    sigma = np.linalg.norm(y_clean) / (snr * np.sqrt(m))
    y = y_clean + rng.normal(size=m) * sigma
    witness = {"cert_type": "primal", "x": np.zeros(n).tolist(), "t": float(np.linalg.norm(y))}
    data = {"A": A, "y": y, "y_clean": y_clean, "R": R, "noise_sigma": sigma}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": witness, "readme": "D1_SOCP"}


def generate_d2_bp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 30))
    n = int(knobs.get("n", 60))
    sparsity = max(1, int(knobs.get("sparsity", 0.1) * n))
    A = rng.normal(size=(m, n))
    col_norms = np.linalg.norm(A, axis=0) + 1e-12
    A = A / col_norms
    support = rng.choice(n, sparsity, replace=False)
    x_true = np.zeros(n)
    x_true[support] = rng.normal(size=sparsity)
    y = A @ x_true
    witness = {"cert_type": "primal", "x_feas": x_true.tolist()}
    data = {"A": A, "y": y, "x_true": x_true}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": witness, "readme": "D2_BP"}


def generate_d3_sdp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 10))
    p = float(knobs.get("p", 0.3))
    W = rng.uniform(size=(n, n))
    mask = rng.uniform(size=(n, n)) < p
    W = np.triu(W * mask, 1)
    W = W + W.T
    L = np.diag(np.sum(W, axis=1)) - W
    witness = {"cert_type": "primal", "X": "identity"}
    data = {"L": L}
    dims = {"n": n}
    return {"data": data, "dims": dims, "witness": witness, "readme": "D3_SDP"}
