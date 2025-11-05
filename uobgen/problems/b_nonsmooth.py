"""Nonsmooth and composite problems (B1-B6)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..utils import (
    calibrate_sigma_for_snr,
    column_normalize,
    make_blocks_2d,
    build_tv_gradients,
    toeplitz_corr,
)


def _base_instance(problem_id: str, name: str, seed: int) -> Dict[str, Any]:
    return {"id": problem_id, "name": name, "seed": seed, "meta": {}, "data": {}, "reference": {}}


def _toeplitz_design(rng: np.random.Generator, m: int, n: int, rho: float) -> np.ndarray:
    Sigma = toeplitz_corr(n, rho)
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(n))
    return rng.normal(size=(m, n)) @ L.T


def generate_b1_lasso(
    seed: int,
    m: int,
    n: int,
    sparsity: float = 0.05,
    rho: float = 0.0,
    snr: float = 20.0,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("B1_LASSO", "LASSO", seed)
    A = _toeplitz_design(rng, m, n, rho)
    A, norms = column_normalize(A)
    s = max(1, int(sparsity * n))
    support = rng.choice(n, size=s, replace=False)
    x_star = np.zeros(n)
    x_star[support] = rng.normal(size=s)
    y_clean = A @ x_star
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    y = y_clean + rng.normal(scale=sigma, size=m)
    lam = alpha * sigma * np.sqrt(2 * np.log(n))
    instance["meta"] = {
        "family": "nonsmooth",
        "dims": {"m": m, "n": n},
        "knobs": {"sparsity": float(s / n), "rho": float(rho), "snr": float(snr), "alpha": float(alpha)},
        "diagnostics": {"sigma": float(sigma), "mean_norm": float(np.mean(norms))},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "y": y, "lambda": lam}
    instance["reference"] = {"x_star": x_star}
    return instance


def generate_b2_elasticnet(
    seed: int,
    m: int,
    n: int,
    rho: float = 0.0,
    snr: float = 20.0,
    lambda1: float = 1e-3,
    lambda2: float = 1e-3,
    sparsity: float = 0.05,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("B2_ElasticNet", "Elastic Net", seed)
    A = _toeplitz_design(rng, m, n, rho)
    A, norms = column_normalize(A)
    s = max(1, int(sparsity * n))
    support = rng.choice(n, size=s, replace=False)
    x_star = np.zeros(n)
    x_star[support] = rng.normal(size=s)
    y_clean = A @ x_star
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    y = y_clean + rng.normal(scale=sigma, size=m)
    instance["meta"] = {
        "family": "nonsmooth",
        "dims": {"m": m, "n": n},
        "knobs": {
            "rho": float(rho),
            "snr": float(snr),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
        },
        "diagnostics": {"sigma": float(sigma)},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "y": y, "lambda1": lambda1, "lambda2": lambda2}
    instance["reference"] = {"x_star": x_star}
    return instance


def generate_b3_svm(
    seed: int,
    m: int,
    n: int,
    mu_norm: float = 1.0,
    rho: float = 0.0,
    gamma: float = 1e-2,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("B3_SVM", "Linear SVM", seed)
    mu = np.zeros(n)
    mu[0] = mu_norm
    Sigma = toeplitz_corr(n, rho)
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(n))
    half = m // 2
    A_pos = rng.normal(size=(half, n)) @ L.T + mu
    A_neg = rng.normal(size=(m - half, n)) @ L.T - mu
    A = np.vstack([A_pos, A_neg])
    y = np.concatenate([np.ones(half), -np.ones(m - half)])
    A, norms = column_normalize(A)
    instance["meta"] = {
        "family": "nonsmooth",
        "dims": {"m": m, "n": n},
        "knobs": {"mu_norm": float(mu_norm), "rho": float(rho), "gamma": float(gamma)},
        "diagnostics": {"mean_norm": float(np.mean(norms))},
        "has_reference": False,
    }
    instance["data"] = {"A": A, "y": y, "gamma": gamma}
    return instance


def generate_b4_tv(
    seed: int,
    image_size: int,
    blocks: int = 8,
    snr: float = 20.0,
    lam: float = 0.1,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("B4_TV", "Isotropic TV Denoising", seed)
    truth = make_blocks_2d(image_size, image_size, blocks, rng)
    x_true = truth.reshape(-1)
    y_clean = x_true.copy()
    sigma = calibrate_sigma_for_snr(y_clean, x_true.size, snr)
    noisy = x_true + rng.normal(scale=sigma, size=x_true.size)
    dx_edges, dy_edges = build_tv_gradients(image_size, image_size)
    instance["meta"] = {
        "family": "nonsmooth",
        "dims": {"n": x_true.size},
        "knobs": {"blocks": int(blocks), "snr": float(snr), "lambda": float(lam)},
        "diagnostics": {"sigma": float(sigma)},
        "has_reference": True,
    }
    instance["data"] = {
        "y": noisy,
        "lambda": lam,
        "shape": np.array([image_size, image_size]),
        "dx_edges": dx_edges,
        "dy_edges": dy_edges,
    }
    instance["reference"] = {"x_true": x_true}
    return instance


def _build_groups(n: int, mode: str) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    if mode == "uniform":
        size = max(1, n // 20)
        idx = 0
        while idx < n:
            end = min(n, idx + size)
            groups.append(np.arange(idx, end))
            idx = end
    else:
        idx = 0
        while idx < n:
            size = max(1, int(np.ceil((n - idx) / 2)))
            groups.append(np.arange(idx, min(n, idx + size)))
            idx += size
    return groups


def _groups_to_matrix(groups: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(g) for g in groups)
    mat = -np.ones((len(groups), max_len), dtype=int)
    for i, g in enumerate(groups):
        mat[i, : len(g)] = g
    return mat


def generate_b5_group_lasso(
    seed: int,
    m: int,
    n: int,
    rho: float = 0.0,
    group_mode: str = "uniform",
    snr: float = 20.0,
    lam: float = 0.1,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("B5_GroupLasso", "Group Lasso", seed)
    A = _toeplitz_design(rng, m, n, rho)
    A, _ = column_normalize(A)
    groups = _build_groups(n, group_mode)
    num_groups = len(groups)
    active = rng.choice(num_groups, size=max(1, num_groups // 5), replace=False)
    x_star = np.zeros(n)
    for idx in active:
        g = groups[idx]
        x_star[g] = rng.normal(size=len(g))
    y_clean = A @ x_star
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    y = y_clean + rng.normal(scale=sigma, size=m)
    instance["meta"] = {
        "family": "nonsmooth",
        "dims": {"m": m, "n": n},
        "knobs": {"rho": float(rho), "group_mode": group_mode, "snr": float(snr), "lambda": float(lam)},
        "diagnostics": {"num_groups": num_groups},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "y": y, "lambda": lam, "groups": _groups_to_matrix(groups)}
    instance["reference"] = {"x_star": x_star}
    return instance


def generate_b6_nonconvex_sparse(
    seed: int,
    m: int,
    n: int,
    rho: float = 0.0,
    snr: float = 20.0,
    penalty: str = "SCAD",
    a: float = 3.7,
    lam: float = 1e-3,
    sparsity: float = 0.05,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("B6_NC_Sparse", "Nonconvex Sparse Logistic", seed)
    A = _toeplitz_design(rng, m, n, rho)
    A, _ = column_normalize(A)
    s = max(1, int(sparsity * n))
    support = rng.choice(n, size=s, replace=False)
    x_star = np.zeros(n)
    x_star[support] = rng.normal(size=s)
    logits = A @ x_star
    var = np.var(logits)
    if var > 0:
        scale = np.sqrt(snr / (var + 1e-9))
        logits *= scale
        x_star *= scale
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)
    y = np.where(y == 0, -1, 1)
    instance["meta"] = {
        "family": "nonsmooth",
        "dims": {"m": m, "n": n},
        "knobs": {"rho": float(rho), "snr": float(snr), "penalty": penalty, "a": float(a), "lambda": float(lam)},
        "diagnostics": {"sparsity": float(s / n)},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "y": y, "lambda": lam, "penalty": penalty, "a": a}
    instance["reference"] = {"x_star": x_star}
    return instance


PROBLEMS = {
    "B1_LASSO": generate_b1_lasso,
    "B2_ElasticNet": generate_b2_elasticnet,
    "B3_SVM": generate_b3_svm,
    "B4_TV": generate_b4_tv,
    "B5_GroupLasso": generate_b5_group_lasso,
    "B6_NC_Sparse": generate_b6_nonconvex_sparse,
}
