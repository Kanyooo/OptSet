"""Nonsmooth problem generators (B1–B6)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..utils.linalg import toeplitz_corr
from ..utils.stats import calibrate_sigma_for_snr


def _design_matrix(m: int, n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    cov = toeplitz_corr(n, rho)
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0) + 1e-12
    A = A / col_norms
    return A


def generate_b1_lasso(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 60))
    n = int(knobs.get("n", 40))
    rho = float(knobs.get("rho", 0.2))
    sparsity = max(1, int(knobs.get("sparsity", 0.1) * n))
    snr = float(knobs.get("snr", 20))
    lam = float(knobs.get("lambda", 0.1))
    A = _design_matrix(m, n, rho, rng)
    support = rng.choice(n, sparsity, replace=False)
    x_true = np.zeros(n)
    x_true[support] = rng.normal(size=sparsity)
    y_clean = A @ x_true
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    noise = rng.normal(size=m) * sigma
    y = y_clean + noise
    data = {"A": A, "y": y, "lambda": lam, "x_true": x_true, "y_clean": y_clean, "noise_sigma": sigma}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": f"B1_LASSO m={m} n={n}"}


def generate_b2_enet(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 60))
    n = int(knobs.get("n", 40))
    rho = float(knobs.get("rho", 0.2))
    lam1 = float(knobs.get("lambda1", 0.1))
    lam2 = float(knobs.get("lambda2", 0.1))
    A = _design_matrix(m, n, rho, rng)
    support = rng.choice(n, max(1, n // 10), replace=False)
    x_true = np.zeros(n)
    x_true[support] = rng.normal(size=len(support))
    y_clean = A @ x_true
    sigma = calibrate_sigma_for_snr(y_clean, m, knobs.get("snr", 20))
    y = y_clean + rng.normal(size=m) * sigma
    data = {"A": A, "y": y, "lambda1": lam1, "lambda2": lam2, "x_true": x_true, "y_clean": y_clean, "noise_sigma": sigma}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": f"B2_ElasticNet m={m} n={n}"}


def generate_b3_svm(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 80))
    n = int(knobs.get("n", 30))
    rho = float(knobs.get("rho", 0.1))
    mu_norm = float(knobs.get("mu_norm", 1.0))
    gamma = float(knobs.get("gamma", 0.1))
    cov = toeplitz_corr(n, rho)
    mu = np.zeros(n)
    mu[0] = mu_norm
    half = m // 2
    A_pos = rng.multivariate_normal(mu, cov, size=half)
    A_neg = rng.multivariate_normal(-mu, cov, size=m - half)
    A = np.vstack([A_pos, A_neg])
    y = np.array([1] * half + [-1] * (m - half))
    data = {"A": A, "y": y, "gamma": gamma}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": "B3_SVM"}


def generate_b4_tv(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    H = int(knobs.get("H", 16))
    W = int(knobs.get("W", 16))
    blocks = int(knobs.get("blocks", 4))
    lam = float(knobs.get("lambda", 0.1))
    noise = float(knobs.get("noise", 0.05))
    image = np.zeros((H, W))
    for _ in range(blocks):
        h = rng.integers(1, H // 2 + 1)
        w = rng.integers(1, W // 2 + 1)
        r = rng.integers(0, H - h + 1)
        c = rng.integers(0, W - w + 1)
        image[r:r+h, c:c+w] = rng.uniform()
    y_clean = image.copy()
    y = y_clean + rng.normal(size=(H, W)) * noise
    data = {"y": y, "y_clean": y_clean, "lambda": lam}
    dims = {"H": H, "W": W}
    return {"data": data, "dims": dims, "witness": {}, "readme": "B4_TV image"}


def generate_b5_group_lasso(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 80))
    n = int(knobs.get("n", 40))
    groups = int(knobs.get("groups", 5))
    lam = float(knobs.get("lambda", 0.1))
    group_sizes = [n // groups] * groups
    for i in range(n - sum(group_sizes)):
        group_sizes[i % groups] += 1
    membership: List[List[int]] = []
    start = 0
    for size in group_sizes:
        membership.append(list(range(start, start + size)))
        start += size
    A = _design_matrix(m, n, knobs.get("rho", 0.2), rng)
    x_true = np.zeros(n)
    for g in membership[: max(1, groups // 2)]:
        x_true[g] = rng.normal(size=len(g))
    max_len = max(len(g) for g in membership)
    groups_matrix = -np.ones((len(membership), max_len), dtype=int)
    for i, g in enumerate(membership):
        groups_matrix[i, : len(g)] = g
    y_clean = A @ x_true
    sigma = calibrate_sigma_for_snr(y_clean, m, knobs.get("snr", 20))
    y = y_clean + rng.normal(size=m) * sigma
    data = {"A": A, "y": y, "groups": groups_matrix, "lambda": lam, "x_true": x_true, "y_clean": y_clean, "noise_sigma": sigma}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": "B5_GroupLasso"}


def generate_b6_nc_sparse(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 100))
    n = int(knobs.get("n", 20))
    rho = float(knobs.get("rho", 0.2))
    lam = float(knobs.get("lambda", 0.1))
    a_param = float(knobs.get("a", 3.7))
    A = _design_matrix(m, n, rho, rng)
    support = rng.choice(n, max(1, n // 5), replace=False)
    x_true = np.zeros(n)
    x_true[support] = rng.normal(size=len(support))
    logits = A @ x_true
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs) * 2 - 1
    data = {"A": A, "y": y, "lambda": lam, "a": a_param, "x_true": x_true}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": "B6_NC_Sparse"}
