"""Barzilai–Borwein gradient method with diagnostic plotting."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .gd import _gradient, _objective, _infer_dim, _maybe_plot
from .kkt import constraint_violation_inf


def solve_bb(
    problem_id: str,
    arrays: Dict[str, np.ndarray],
    max_iter: int = 500,
    tol: float = 1e-6,
    plot: bool = False,
) -> Dict:
    """Run the Barzilai–Borwein method with BB1/BB2 step safeguards."""

    n = _infer_dim(arrays)
    x = np.zeros(n)
    history = {"f": [], "step": [], "constraint": []}
    grad = _gradient(problem_id, arrays, x)
    alpha = 1.0
    for it in range(max_iter):
        violation = constraint_violation_inf(problem_id, arrays, x)
        if np.linalg.norm(grad, ord=np.inf) < tol and violation < 10 * tol:
            f_val = _objective(problem_id, arrays, x)
            history["f"].append(f_val)
            history["constraint"].append(violation)
            _maybe_plot(history, f"BB on {problem_id}", "Objective", plot)
            return {
                "x": x,
                "status": "converged",
                "iters": it,
                "history": history,
                "obj": f_val,
            }
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
        history["constraint"].append(violation)
        history["step"].append(alpha)
    final_obj = _objective(problem_id, arrays, x)
    history["f"].append(final_obj)
    history["constraint"].append(constraint_violation_inf(problem_id, arrays, x))
    _maybe_plot(history, f"BB on {problem_id}", "Objective", plot)
    return {
        "x": x,
        "status": "max_iter",
        "iters": max_iter,
        "history": history,
        "obj": final_obj,
    }
