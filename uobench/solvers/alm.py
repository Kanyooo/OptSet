"""Augmented Lagrangian solver for equality constraints."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .gd import solve_gd
from .bb import solve_bb
from .newton import solve_newton

INNER = {
    "gd": solve_gd,
    "bb": solve_bb,
    "newton": solve_newton,
}


def solve_alm(problem_id: str, arrays: Dict[str, np.ndarray], max_iter: int = 100, tol: float = 1e-6, inner: str = "newton") -> Dict:
    if "A" not in arrays or "d" not in arrays:
        raise ValueError("ALM requires equality constraints")
    A = arrays["A"]
    d = arrays["d"]
    n = A.shape[1]
    x = np.zeros(n)
    lam = np.zeros(A.shape[0])
    rho = 1.0
    history = {"prim_resid": [], "dual_resid": [], "kkt": []}
    inner_solver = INNER.get(inner, solve_newton)
    for it in range(max_iter):
        def augmented_obj(z: np.ndarray) -> float:
            r = A @ z - d
            return 0.5 * z @ arrays["Q"] @ z - arrays["b"] @ z + lam @ r + 0.5 * rho * np.linalg.norm(r) ** 2

        arrays_aug = {**arrays, "lambda": rho, "A": A, "d": d}
        result = inner_solver(problem_id, {**arrays, "b": arrays["b"] - A.T @ (lam + rho * (A @ x - d))}, max_iter=50, tol=tol)
        x = result["x"]
        r = A @ x - d
        lam = lam + rho * r
        prim = np.linalg.norm(r)
        dual = np.linalg.norm(A.T @ lam)
        history["prim_resid"].append(float(prim))
        history["dual_resid"].append(float(dual))
        history["kkt"].append(float(prim + dual))
        if prim < tol and dual < tol:
            return {"x": x, "status": "converged", "iters": it, "history": history, "obj": augmented_obj(x)}
        rho *= 1.5
    return {"x": x, "status": "max_iter", "iters": max_iter, "history": history, "obj": augmented_obj(x)}
