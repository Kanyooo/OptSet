"""KKT residual utilities and constraint violation helpers.

These routines are shared across baseline solvers so that every iteration can
track both objective values and feasibility margins.  Providing a single place
that translates ``problem_id`` into algebraic checks keeps the solvers readable
and ensures the generated plots convey the same story for every algorithm.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def residual_qp(Q: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(Q @ x - b))


def residual_eq(A: np.ndarray, x: np.ndarray, d: np.ndarray) -> float:
    return float(np.linalg.norm(A @ x - d))


def residual_box(x: np.ndarray, l: np.ndarray, u: np.ndarray) -> float:
    viol = np.maximum(0, l - x) + np.maximum(0, x - u)
    return float(np.linalg.norm(viol))


def constraint_violation_inf(problem_id: str, arrays: Dict[str, np.ndarray], x: np.ndarray) -> float:
    """Return the infinity-norm constraint violation for ``problem_id``.

    Parameters
    ----------
    problem_id:
        Identifier of the benchmark problem.  The mapping mirrors the registry
        in :mod:`uobench.core.spec`.
    arrays:
        Data payload returned by the corresponding generator.
    x:
        Current primal iterate produced by a solver.

    Notes
    -----
    The function intentionally handles only the constraint families appearing
    in S-scale demos.  When a problem has multiple coupled variables (e.g.
    SOCP with ``(x, t)``) the caller is expected to pass the packed vector and
    the generator stores enough arrays so the violation can be computed here.
    """

    if problem_id == "A4_ECQP" and "A" in arrays and "d" in arrays:
        return float(np.linalg.norm(arrays["A"] @ x - arrays["d"], ord=np.inf))
    if problem_id == "A5_TRS" and "delta" in arrays:
        radius = float(arrays["delta"])
        return max(0.0, float(np.linalg.norm(x, ord=2) - radius))
    if problem_id == "A6_BoxQP" and {"l", "u"}.issubset(arrays.keys()):
        l = arrays["l"]
        u = arrays["u"]
        lower = np.maximum(0.0, l - x)
        upper = np.maximum(0.0, x - u)
        return float(np.linalg.norm(lower + upper, ord=np.inf))
    if problem_id == "C1_VI" and {"l", "u"}.issubset(arrays.keys()):
        l = arrays["l"]
        u = arrays["u"]
        lower = np.maximum(0.0, l - x)
        upper = np.maximum(0.0, x - u)
        return float(np.linalg.norm(lower + upper, ord=np.inf))
    if problem_id == "B3_SVM" and "proj_radius" in arrays:
        radius = float(arrays["proj_radius"])
        return max(0.0, float(np.linalg.norm(x, ord=2) - radius))
    if problem_id == "C2_LCP" and "M" in arrays and "q" in arrays:
        z = np.maximum(0.0, x)
        w = arrays["M"] @ z + arrays["q"]
        w = np.maximum(0.0, w)
        return float(np.linalg.norm(np.minimum(z, w), ord=np.inf))
    if problem_id == "D1_SOCP" and {"A", "y", "R"}.issubset(arrays.keys()):
        # Pack variables as [x, t]; violation is max(||Ax-y||-t, ||x||-R).
        n = arrays["A"].shape[1]
        x_vec = x[:n]
        t = x[-1]
        resid1 = np.linalg.norm(arrays["A"] @ x_vec - arrays["y"], ord=2) - t
        resid2 = np.linalg.norm(x_vec, ord=2) - arrays["R"]
        return float(np.maximum(resid1, resid2))
    if problem_id == "D2_BP" and "A" in arrays and "y" in arrays:
        return float(np.linalg.norm(arrays["A"] @ x - arrays["y"], ord=np.inf))
    # Unconstrained or unsupported problems default to zero violation.
    return 0.0
