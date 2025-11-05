"""Barzilai–Borwein gradient descent."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .gd import _gradient, _objective


def solve_bb(problem_id: str, arrays: Dict[str, np.ndarray], max_iter: int = 500, tol: float = 1e-6) -> Dict:
    n = arrays.get("Q", arrays.get("A", np.zeros((1, 1)))).shape[1]
    x = np.zeros(n)
    history = {"f": [], "step": []}
    grad = _gradient(problem_id, arrays, x)
    alpha = 1.0
    for it in range(max_iter):
        if np.linalg.norm(grad, ord=np.inf) < tol:
            return {"x": x, "status": "converged", "iters": it, "history": history, "obj": _objective(problem_id, arrays, x)}
        s = -alpha * grad
        x_new = x + s
        grad_new = _gradient(problem_id, arrays, x_new)
        y = grad_new - grad
        denom = np.dot(y, s)
        if denom > 0:
            alpha = np.dot(s, s) / denom
        alpha = np.clip(alpha, 1e-4, 10.0)
        x = x_new
        grad = grad_new
        history["f"].append(_objective(problem_id, arrays, x))
        history["step"].append(alpha)
    return {"x": x, "status": "max_iter", "iters": max_iter, "history": history, "obj": _objective(problem_id, arrays, x)}
