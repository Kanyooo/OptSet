"""Gradient descent baseline with optional descent plotting."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _infer_dim(arrays: Dict[str, np.ndarray]) -> int:
    if "Q" in arrays:
        return arrays["Q"].shape[0]
    if "A" in arrays:
        return arrays["A"].shape[1]
    if "H" in arrays:
        return arrays["H"].shape[0]
    if "M" in arrays:
        return arrays["M"].shape[0]
    if "x_true" in arrays:
        return arrays["x_true"].shape[0]
    if "shape" in arrays:
        return int(np.prod(arrays["shape"]))
    raise ValueError("Cannot infer variable dimension from arrays")


def _objective(problem_id: str, arrays: Dict[str, np.ndarray], x: np.ndarray) -> float:
    if problem_id in {"A1_QP", "A4_ECQP"}:
        Q = arrays["Q"]
        b = arrays["b"]
        return 0.5 * x @ Q @ x - b @ x
    if problem_id == "A2_LogReg" or problem_id == "B6_NC_Sparse":
        A = arrays["A"]
        y = arrays["y"]
        lam = arrays.get("lambda", 0.0)
        logits = A @ x
        return float(np.mean(np.log1p(np.exp(-y * logits))) + 0.5 * lam * np.linalg.norm(x) ** 2)
    return float(0.5 * np.linalg.norm(x) ** 2)


def _gradient(problem_id: str, arrays: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    if problem_id in {"A1_QP", "A4_ECQP"}:
        Q = arrays["Q"]
        b = arrays["b"]
        return Q @ x - b
    if problem_id in {"A2_LogReg", "B6_NC_Sparse"}:
        A = arrays["A"]
        y = arrays["y"]
        lam = arrays.get("lambda", 0.0)
        logits = A @ x
        probs = 1 / (1 + np.exp(y * logits))
        grad = -(A.T @ (y * probs)) / A.shape[0] + lam * x
        return grad
    return x


def _maybe_plot(history: Dict[str, list], title: str, ylabel: str, plot: bool) -> None:
    if not plot:
        return
    import os
    import matplotlib

    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    values = history.get("f", [])
    if not values:
        return
    ax.plot(range(len(values)), values, marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    plt.show()


def solve_gd(
    problem_id: str,
    arrays: Dict[str, np.ndarray],
    max_iter: int = 500,
    tol: float = 1e-6,
    plot: bool = False,
) -> Dict:
    n = _infer_dim(arrays)
    x = np.zeros(n)
    history = {"f": [], "step": []}
    alpha = 1.0
    for it in range(max_iter):
        grad = _gradient(problem_id, arrays, x)
        norm = np.linalg.norm(grad, ord=np.inf)
        if norm < tol:
            history["f"].append(_objective(problem_id, arrays, x))
            _maybe_plot(history, f"GD on {problem_id}", "Objective", plot)
            return {
                "x": x,
                "status": "converged",
                "iters": it,
                "history": history,
                "obj": _objective(problem_id, arrays, x),
            }
        f = _objective(problem_id, arrays, x)
        t = alpha
        while t > 1e-8:
            x_new = x - t * grad
            f_new = _objective(problem_id, arrays, x_new)
            if f_new <= f - 1e-4 * t * np.linalg.norm(grad) ** 2:
                break
            t *= 0.5
        x = x - t * grad
        history["f"].append(f)
        history["step"].append(t)
        alpha = t * 1.5
    history["f"].append(_objective(problem_id, arrays, x))
    _maybe_plot(history, f"GD on {problem_id}", "Objective", plot)
    return {
        "x": x,
        "status": "max_iter",
        "iters": max_iter,
        "history": history,
        "obj": _objective(problem_id, arrays, x),
    }
