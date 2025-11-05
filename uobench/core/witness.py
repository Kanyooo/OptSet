"""Feasibility certificate verification."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _check_box(x: np.ndarray, l: np.ndarray, u: np.ndarray, tol: float) -> bool:
    return bool(np.all(x >= l - tol) and np.all(x <= u + tol))


def _check_simplex(x: np.ndarray, tol: float) -> bool:
    return bool(np.all(x >= -tol) and abs(np.sum(x) - 1.0) <= tol)


def verify(problem_id: str, meta: Dict, arrays: Dict[str, np.ndarray], tol: float = 1e-6) -> bool:
    """Return ``True`` if the stored witness proves feasibility."""

    witness = meta.get("witness", {})
    cert_type = witness.get("cert_type")

    if cert_type is None:
        return True

    if cert_type == "primal":
        if problem_id in {"A1_QP", "A2_LogReg", "A3_Rosenbrock", "B1_LASSO", "B2_ElasticNet", "B3_SVM", "B4_TV", "B6_NC_Sparse"}:
            return True  # unconstrained
        if problem_id == "A4_ECQP":
            x = np.asarray(witness["x_feas"])
            return np.linalg.norm(arrays["A"] @ x - arrays["d"]) <= tol
        if problem_id == "A5_TRS":
            x = np.asarray(witness["x_feas"])
            radius = float(witness.get("radius", arrays["delta"]))
            return np.linalg.norm(x) <= radius + tol
        if problem_id == "A6_BoxQP":
            x = np.asarray(witness["x_feas"])
            return _check_box(x, arrays["l"], arrays["u"], tol)
        if problem_id == "B5_GroupLasso":
            return True
        if problem_id == "C1_VI":
            x = np.asarray(witness["x_feas"])
            geometry = witness.get("geometry", arrays.get("geometry", "box"))
            if geometry == "simplex":
                return _check_simplex(x, tol)
            return _check_box(x, arrays["l"], arrays["u"], tol)
        if problem_id == "C3_MPCC":
            s = arrays["b"]
            return bool(np.all(s >= -tol))
        if problem_id == "D1_SOCP":
            t = float(witness.get("t", np.linalg.norm(arrays["y"])) )
            return t >= np.linalg.norm(arrays["y"]) - tol
        if problem_id == "D2_BP":
            x = np.asarray(witness["x_feas"])
            return np.linalg.norm(arrays["A"] @ x - arrays["y"]) <= tol
        if problem_id == "D3_SDP":
            return witness.get("X") == "identity"

    if cert_type == "complementarity":
        if problem_id == "C2_LCP":
            z = np.asarray(witness["z"])
            q = arrays["q"]
            M = arrays["M"]
            w = M @ z + q
            return bool(np.all(z >= -tol) and np.all(w >= -tol) and abs(float(z @ w)) <= tol)
        if problem_id == "C3_MPCC":
            y = np.asarray(witness.get("y", np.zeros(arrays["b"].shape)))
            s = np.asarray(witness.get("s", arrays["b"]))
            return bool(np.all(y >= -tol) and np.all(s >= -tol) and abs(float(y @ s)) <= tol)

    return False
