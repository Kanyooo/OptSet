"""Proximal and projected methods."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.linalg import project_box, project_simplex, project_ball
from .gd import _objective, _infer_dim, _maybe_plot
from .kkt import constraint_violation_inf


def prox_l1(v: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)


def prox_l2(v: np.ndarray, lam: float) -> np.ndarray:
    return v / (1 + lam)


def solve_fista(
    problem_id: str,
    arrays: Dict[str, np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
    plot: bool = False,
) -> Dict:
    """ISTA/FISTA solver for LASSO-style objectives with plotting hooks."""

    A = arrays.get("A")
    if A is None:
        raise ValueError("FISTA requires matrix A")
    m, n = A.shape
    x = np.zeros(n)
    y = x.copy()
    t = 1.0
    L = np.linalg.norm(A, 2) ** 2
    history = {"f": [], "constraint": []}
    for k in range(1, max_iter + 1):
        grad = A.T @ (A @ y - arrays["y"]) / m
        step = y - grad / L
        if problem_id in {"B1_LASSO", "D2_BP"}:
            x_new = prox_l1(step, arrays.get("lambda", 0.1) / L)
        else:
            x_new = prox_l2(step, arrays.get("lambda2", 0.1) / L)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        obj_val = float(0.5 * np.linalg.norm(A @ x - arrays["y"]) ** 2 + arrays.get("lambda", 0.1) * np.linalg.norm(x, 1))
        history["f"].append(obj_val)
        history["constraint"].append(constraint_violation_inf(problem_id, arrays, x))
        if np.linalg.norm(grad, ord=np.inf) < tol and history["constraint"][-1] < 10 * tol:
            _maybe_plot(history, f"FISTA on {problem_id}", "Objective", plot)
            return {"x": x, "status": "converged", "iters": k, "history": history, "obj": obj_val}
    _maybe_plot(history, f"FISTA on {problem_id}", "Objective", plot)
    return {"x": x, "status": "max_iter", "iters": max_iter, "history": history, "obj": history["f"][-1]}


def solve_projected_gd(
    problem_id: str,
    arrays: Dict[str, np.ndarray],
    max_iter: int = 200,
    tol: float = 1e-6,
    plot: bool = False,
) -> Dict:
    """Projected gradient descent for trust-region, box, and VI problems."""

    if problem_id == "A5_TRS":
        proj = lambda z: project_ball(z, arrays["delta"])
        grad_fn = lambda x: arrays["H"] @ x + arrays["g"]
    elif problem_id == "A6_BoxQP":
        proj = lambda z: project_box(z, arrays["l"], arrays["u"])
        grad_fn = lambda x: arrays["H"] @ x + arrays["c"]
    elif problem_id == "C1_VI":
        if "l" in arrays:
            proj = lambda z: project_box(z, arrays["l"], arrays["u"])
        else:
            proj = project_simplex
        grad_fn = lambda x: arrays["Q"] @ x + arrays["c"]
    elif problem_id == "C2_LCP":
        proj = lambda z: np.maximum(0.0, z)
        grad_fn = lambda x: arrays["M"] @ x + arrays["q"]
    else:
        proj = lambda z: z
        grad_fn = lambda x: arrays.get("Q", np.eye(len(x))) @ x
    n = _infer_dim(arrays)
    x = np.zeros(n)
    history = {"f": [], "constraint": []}
    for it in range(max_iter):
        grad = grad_fn(x)
        violation = constraint_violation_inf(problem_id, arrays, x)
        if np.linalg.norm(grad, ord=np.inf) < tol and violation < 10 * tol:
            history["f"].append(float(np.linalg.norm(grad)))
            history["constraint"].append(violation)
            _maybe_plot(history, f"Projected GD on {problem_id}", "Gradient norm", plot)
            return {"x": x, "status": "converged", "iters": it, "history": history, "obj": float(np.linalg.norm(grad))}
        x = proj(x - 0.1 * grad)
        history["f"].append(float(np.linalg.norm(grad)))
        history["constraint"].append(violation)
    history["constraint"].append(constraint_violation_inf(problem_id, arrays, x))
    _maybe_plot(history, f"Projected GD on {problem_id}", "Gradient norm", plot)
    return {"x": x, "status": "max_iter", "iters": max_iter, "history": history, "obj": history["f"][-1] if history["f"] else None}
