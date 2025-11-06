"""Gradient descent baseline with optional descent and feasibility plotting."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .kkt import constraint_violation_inf

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

    values = history.get("f", [])
    if not values:
        return
    has_constraint = "constraint" in history and history["constraint"]
    if has_constraint:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        obj_ax, constr_ax = axes
    else:
        fig, obj_ax = plt.subplots()
        constr_ax = None
    iterations = list(range(len(values)))
    obj_ax.plot(iterations, values, marker="o", linewidth=1.5)
    obj_ax.set_title(title)
    obj_ax.set_xlabel("Iteration")
    obj_ax.set_ylabel(ylabel)
    obj_ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if constr_ax is not None:
        constr_vals = history.get("constraint", [])
        constr_ax.plot(iterations, constr_vals, marker="s", color="#bb2121", linewidth=1.5)
        constr_ax.set_title("Constraint violation (∞-norm)")
        constr_ax.set_xlabel("Iteration")
        constr_ax.set_ylabel("‖violation‖∞")
        constr_ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        constr_ax.set_yscale("symlog")
    fig.tight_layout()
    plt.show()


def solve_gd(
    problem_id: str,
    arrays: Dict[str, np.ndarray],
    max_iter: int = 500,
    tol: float = 1e-6,
    plot: bool = False,
) -> Dict:
    """Run plain gradient descent with Armijo backtracking.

    Parameters
    ----------
    problem_id:
        Identifier of the generated instance, e.g. ``"A1_QP"``.  The ID
        selects gradients/objectives and determines which constraints are
        monitored for feasibility diagnostics.
    arrays:
        Dictionary of NumPy arrays output by the generator.  Required entries
        depend on ``problem_id``; the solvers access only documented keys
        (``Q``, ``A``, ``b`` and similar) and never mutate them.
    max_iter:
        Hard limit on the number of gradient steps.  Every iteration stores the
        objective and constraint violation in ``history`` so the notebook guide
        can render them easily.
    tol:
        Stopping tolerance measured on the infinity-norm of the gradient.
    plot:
        When ``True`` an interactive Matplotlib summary of the stored history is
        displayed.  The figure always includes the objective curve and, when
        relevant, the constraint violation curve.
    """

    n = _infer_dim(arrays)
    x = np.zeros(n)
    history = {"f": [], "step": [], "constraint": []}
    alpha = 1.0
    for it in range(max_iter):
        grad = _gradient(problem_id, arrays, x)
        norm = np.linalg.norm(grad, ord=np.inf)
        violation = constraint_violation_inf(problem_id, arrays, x)
        if norm < tol and violation < 10 * tol:
            f_val = _objective(problem_id, arrays, x)
            history["f"].append(f_val)
            history["constraint"].append(violation)
            _maybe_plot(history, f"GD on {problem_id}", "Objective", plot)
            return {
                "x": x,
                "status": "converged",
                "iters": it,
                "history": history,
                "obj": f_val,
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
        history["constraint"].append(violation)
        alpha = t * 1.5
    final_obj = _objective(problem_id, arrays, x)
    history["f"].append(final_obj)
    history["constraint"].append(constraint_violation_inf(problem_id, arrays, x))
    _maybe_plot(history, f"GD on {problem_id}", "Objective", plot)
    return {
        "x": x,
        "status": "max_iter",
        "iters": max_iter,
        "history": history,
        "obj": final_obj,
    }
