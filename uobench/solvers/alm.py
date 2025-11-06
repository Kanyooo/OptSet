"""Augmented Lagrangian solver for equality-constrained problems."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .gd import _gradient, _objective, _infer_dim, _maybe_plot


def solve_alm(
    problem_id: str,
    arrays: Dict[str, np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
    inner: str = "gd",
    plot: bool = False,
) -> Dict:
    """Solve equality-constrained problems using a simple ALM scheme."""

    if "A" not in arrays or "d" not in arrays:
        raise ValueError("ALM requires linear equality constraints stored under 'A' and 'd'")

    A = arrays["A"]
    d = arrays["d"]
    n = _infer_dim(arrays)
    m = A.shape[0]
    x = np.zeros(n)
    lam = np.zeros(m)
    rho = 1.0
    history = {"prim_resid": [], "dual_resid": [], "kkt": [], "f": [], "constraint": []}

    for it in range(max_iter):
        grad = _gradient(problem_id, arrays, x) + A.T @ (lam + rho * (A @ x - d))
        H = rho * (A.T @ A)
        if problem_id in {"A4_ECQP"} and "Q" in arrays:
            H = H + arrays["Q"]
            rhs = arrays["b"] + rho * A.T @ (d - lam / rho)
            x = np.linalg.solve(H, rhs)
        else:
            step = np.linalg.solve(H + 1e-6 * np.eye(n), grad)
            x = x - step
        r = A @ x - d
        lam = lam + rho * r
        prim = np.linalg.norm(r)
        dual = np.linalg.norm(A.T @ lam)
        history["prim_resid"].append(float(prim))
        history["dual_resid"].append(float(dual))
        history["kkt"].append(float(np.linalg.norm(grad)))
        history["f"].append(float(_objective(problem_id, arrays, x)))
        history["constraint"].append(float(np.linalg.norm(r, ord=np.inf)))
        if prim < tol and dual < tol:
            _maybe_plot(history, f"ALM on {problem_id}", "Objective", plot)
            return {"x": x, "status": "converged", "iters": it, "history": history, "obj": _objective(problem_id, arrays, x)}
        rho = min(rho * 1.5, 1e4)
    _maybe_plot(history, f"ALM on {problem_id}", "Objective", plot)
    return {"x": x, "status": "max_iter", "iters": max_iter, "history": history, "obj": _objective(problem_id, arrays, x)}
