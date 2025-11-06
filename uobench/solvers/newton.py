"""Damped Newton method with feasibility monitoring."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .gd import _gradient, _objective, _infer_dim, _maybe_plot
from .kkt import constraint_violation_inf


def _hessian(problem_id: str, arrays: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    if problem_id in {"A1_QP", "A4_ECQP"}:
        return arrays["Q"]
    if problem_id in {"A2_LogReg", "B6_NC_Sparse"}:
        A = arrays["A"]
        y = arrays["y"]
        logits = A @ x
        w = np.exp(y * logits) / (1 + np.exp(y * logits))**2
        H = (A.T * w) @ A / A.shape[0] + arrays.get("lambda", 0.0) * np.eye(A.shape[1])
        return H
    return np.eye(len(x))


def solve_newton(
    problem_id: str,
    arrays: Dict[str, np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
    plot: bool = False,
) -> Dict:
    """Run damped Newton iterations with line search and plotting."""

    n = _infer_dim(arrays)
    x = np.zeros(n)
    history = {"f": [], "step": [], "kkt": [], "constraint": []}
    for it in range(max_iter):
        grad = _gradient(problem_id, arrays, x)
        H = _hessian(problem_id, arrays, x)
        eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
        shift = max(0.0, -np.min(eigs)) + 1e-6
        H_reg = H + shift * np.eye(n)
        step = np.linalg.solve(H_reg, grad)
        violation = constraint_violation_inf(problem_id, arrays, x)
        if np.linalg.norm(grad, ord=np.inf) < tol and violation < 10 * tol:
            f_val = _objective(problem_id, arrays, x)
            history["kkt"].append(float(np.linalg.norm(grad)))
            history["f"].append(f_val)
            history["constraint"].append(violation)
            _maybe_plot(history, f"Newton on {problem_id}", "Objective", plot)
            return {
                "x": x,
                "status": "converged",
                "iters": it,
                "history": history,
                "obj": f_val,
            }
        t = 1.0
        f = _objective(problem_id, arrays, x)
        while t > 1e-8:
            x_new = x - t * step
            f_new = _objective(problem_id, arrays, x_new)
            if f_new <= f - 1e-4 * t * grad @ step:
                break
            t *= 0.5
        x = x_new
        history["f"].append(f)
        history["step"].append(t)
        history["kkt"].append(float(np.linalg.norm(grad)))
        history["constraint"].append(violation)
    final_obj = _objective(problem_id, arrays, x)
    history["f"].append(final_obj)
    history["constraint"].append(constraint_violation_inf(problem_id, arrays, x))
    _maybe_plot(history, f"Newton on {problem_id}", "Objective", plot)
    return {
        "x": x,
        "status": "max_iter",
        "iters": max_iter,
        "history": history,
        "obj": final_obj,
    }
