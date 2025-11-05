"""Composite and nonsmooth problem generators (family B1–B6)."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ..utils.linalg import toeplitz_corr
from ..utils.stats import calibrate_sigma_for_snr


def _readme(problem: str, knobs: Dict[str, float]) -> str:
    knobs_txt = ", ".join(f"{k}={v}" for k, v in sorted(knobs.items()))
    return f"{problem} generated with {knobs_txt}"


def _sample_sparse_vector(rng: np.random.Generator, n: int, sparsity: int, scale: float = 1.0) -> np.ndarray:
    idx = rng.choice(n, sparsity, replace=False)
    x = np.zeros(n)
    x[idx] = rng.normal(scale=scale, size=sparsity)
    return x


def generate_b1_lasso(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Generate an instance of the LASSO regression problem."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 200))
    n = int(knobs.get("n", 100))
    rho = float(knobs.get("rho", 0.3))
    sparsity = max(1, int(knobs.get("sparsity", 0.1 * n)))
    snr = float(knobs.get("snr", 20.0))
    lam = float(knobs.get("lambda", 0.1))
    if extreme:
        snr = max(snr, 80.0)
        rho = max(rho, 0.95)
    cov = toeplitz_corr(n, rho)
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms
    x_true = _sample_sparse_vector(rng, n, sparsity)
    y_clean = A @ x_true
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    noise = rng.normal(scale=sigma, size=m)
    y = y_clean + noise
    data = {"A": A, "y": y, "lambda": lam, "y_clean": y_clean, "x_true": x_true}
    dims = {"m": m, "n": n}
    used_knobs = {"m": m, "n": n, "rho": rho, "sparsity": sparsity, "snr": snr, "lambda": lam}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": {"x_true": x_true},
        "readme": _readme("B1_LASSO", used_knobs),
    }


def generate_b2_enet(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Elastic net regression dataset."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 200))
    n = int(knobs.get("n", 100))
    rho = float(knobs.get("rho", 0.3))
    lam1 = float(knobs.get("lambda1", 0.1))
    lam2 = float(knobs.get("lambda2", 0.1))
    sparsity = max(1, int(knobs.get("sparsity", 0.1 * n)))
    snr = float(knobs.get("snr", 20.0))
    if extreme:
        lam1 = max(lam1, 1.0)
        lam2 = max(lam2, 1.0)
    cov = toeplitz_corr(n, rho)
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms
    x_true = _sample_sparse_vector(rng, n, sparsity)
    y_clean = A @ x_true
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    y = y_clean + rng.normal(scale=sigma, size=m)
    data = {
        "A": A,
        "y": y,
        "lambda1": lam1,
        "lambda2": lam2,
        "y_clean": y_clean,
        "x_true": x_true,
    }
    dims = {"m": m, "n": n}
    used_knobs = {"m": m, "n": n, "rho": rho, "lambda1": lam1, "lambda2": lam2, "sparsity": sparsity, "snr": snr}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": {"x_true": x_true},
        "readme": _readme("B2_ElasticNet", used_knobs),
    }


def generate_b3_svm(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Linear SVM dataset with controllable class overlap."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 200))
    n = int(knobs.get("n", 50))
    rho = float(knobs.get("rho", 0.2))
    mu_norm = float(knobs.get("mu_norm", 1.0))
    gamma = float(knobs.get("gamma", 0.1))
    if extreme:
        mu_norm = max(mu_norm, 0.2)
        rho = max(rho, 0.99)
    cov = toeplitz_corr(n, rho)
    mu = np.zeros(n)
    mu[0] = mu_norm
    half = m // 2
    X_pos = rng.multivariate_normal(mu, cov, size=half)
    X_neg = rng.multivariate_normal(-mu, cov, size=m - half)
    A = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(half), -np.ones(m - half)])
    data = {"A": A, "y": y, "gamma": gamma, "mu": mu}
    dims = {"m": m, "n": n}
    used_knobs = {"m": m, "n": n, "rho": rho, "mu_norm": mu_norm, "gamma": gamma}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": {},
        "readme": _readme("B3_SVM", used_knobs),
    }


def generate_b4_tv(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Isotropic TV denoising setup for synthetic blocky images."""

    rng = np.random.default_rng(seed)
    H = int(knobs.get("H", 64))
    W = int(knobs.get("W", 64))
    blocks = int(knobs.get("blocks", 8))
    lam = float(knobs.get("lambda", 0.1))
    noise = float(knobs.get("noise", 0.05))
    if extreme:
        noise = max(noise, 0.2)
    ground = np.zeros((H, W))
    block_h = max(1, H // blocks)
    block_w = max(1, W // blocks)
    for i in range(0, H, block_h):
        for j in range(0, W, block_w):
            value = rng.uniform(-1, 1)
            ground[i : min(i + block_h, H), j : min(j + block_w, W)] = value
    y = ground + rng.normal(scale=noise, size=ground.shape)
    size = H * W
    Dx = np.zeros((size, size))
    Dy = np.zeros((size, size))

    def idx(r: int, c: int) -> int:
        return r * W + c

    for r in range(H):
        for c in range(W):
            i = idx(r, c)
            if c < W - 1:
                Dx[i, i] = -1
                Dx[i, idx(r, c + 1)] = 1
            if r < H - 1:
                Dy[i, i] = -1
                Dy[i, idx(r + 1, c)] = 1
    data = {
        "y": y,
        "ground_truth": ground,
        "lambda": lam,
        "Dx": Dx,
        "Dy": Dy,
        "shape": np.array([H, W]),
    }
    dims = {"n": H * W}
    used_knobs = {"H": H, "W": W, "blocks": blocks, "lambda": lam, "noise": noise}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": {"ground_truth": ground},
        "readme": _readme("B4_TV", used_knobs),
    }


def generate_b5_group_lasso(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Group lasso regression instance."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 200))
    n = int(knobs.get("n", 100))
    num_groups = int(knobs.get("groups", 10))
    lam = float(knobs.get("lambda", 0.1))
    rho = float(knobs.get("rho", 0.3))
    sparsity = max(1, int(knobs.get("sparsity", 0.2 * num_groups)))
    snr = float(knobs.get("snr", 20.0))
    if extreme:
        num_groups = max(2, num_groups)
        sparsity = max(1, num_groups // 2)
    cov = toeplitz_corr(n, rho)
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms
    group_sizes = np.full(num_groups, n // num_groups)
    group_sizes[: n % num_groups] += 1
    groups: List[np.ndarray] = []
    start = 0
    for size in group_sizes:
        groups.append(np.arange(start, start + int(size)))
        start += int(size)
    active = rng.choice(num_groups, sparsity, replace=False)
    x_true = np.zeros(n)
    for g in active:
        idx = groups[g]
        x_true[idx] = rng.normal(size=idx.size)
    y_clean = A @ x_true
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    y = y_clean + rng.normal(scale=sigma, size=m)
    group_indices = np.concatenate(groups)
    group_ptr = np.cumsum([0] + [g.size for g in groups])
    data = {
        "A": A,
        "y": y,
        "lambda": lam,
        "group_indices": group_indices,
        "group_ptr": group_ptr,
        "x_true": x_true,
        "y_clean": y_clean,
    }
    dims = {"m": m, "n": n, "groups": num_groups}
    used_knobs = {"m": m, "n": n, "groups": num_groups, "lambda": lam, "rho": rho, "sparsity_groups": sparsity, "snr": snr}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": {"x_true": x_true},
        "readme": _readme("B5_GroupLasso", used_knobs),
    }


def generate_b6_nc_sparse(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Logistic regression with non-convex sparse penalties (SCAD/MCP)."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 400))
    n = int(knobs.get("n", 50))
    rho = float(knobs.get("rho", 0.3))
    snr = float(knobs.get("snr", 15.0))
    lam = float(knobs.get("lambda", 0.1))
    a_param = float(knobs.get("a", 3.7))
    penalty = knobs.get("penalty", "SCAD").upper()
    sparsity = max(1, int(knobs.get("sparsity", 0.1 * n)))
    if extreme:
        lam = min(lam, 0.05)
        snr = max(snr, 60.0)
    cov = toeplitz_corr(n, rho)
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms
    x_true = _sample_sparse_vector(rng, n, sparsity)
    x_true /= np.linalg.norm(x_true) + 1e-12
    x_true *= math.sqrt(snr)
    logits = A @ x_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs) * 2 - 1
    data = {
        "A": A,
        "y": y.astype(float),
        "lambda": lam,
        "a": a_param,
        "penalty": penalty,
        "x_true": x_true,
    }
    dims = {"m": m, "n": n}
    used_knobs = {"m": m, "n": n, "rho": rho, "snr": snr, "lambda": lam, "a": a_param, "penalty": penalty}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": {"x_true": x_true},
        "readme": _readme("B6_NC_Sparse", used_knobs),
    }
