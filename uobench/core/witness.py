"""Feasibility witnesses and verification routines."""

from __future__ import annotations

from typing import Dict

import numpy as np


def verify(problem_id: str, meta: Dict, arrays: Dict[str, np.ndarray], tol: float = 1e-6) -> bool:
    witness = meta.get("witness", {})
    cert_type = witness.get("cert_type")
    if cert_type == "primal":
        if problem_id == "A4_ECQP":
            x = np.asarray(witness["x_feas"])
            A = arrays["A"]
            d = arrays["d"]
            return np.linalg.norm(A @ x - d) <= tol
        if problem_id == "A5_TRS":
            x = np.asarray(witness["x_feas"])
            return np.linalg.norm(x) <= arrays["delta"] + tol
        if problem_id == "A6_BoxQP":
            x = np.asarray(witness["x_feas"])
            return np.all(x >= arrays["l"] - tol) and np.all(x <= arrays["u"] + tol)
        if problem_id == "C1_VI":
            x = np.asarray(witness["x_feas"])
            if witness.get("set") == "box":
                l = arrays["l"]
                u = arrays["u"]
                return np.all(x >= l - tol) and np.all(x <= u + tol)
            if witness.get("set") == "simplex":
                return np.all(x >= -tol) and abs(np.sum(x) - 1) <= tol
        if problem_id == "C3_MPCC":
            s = arrays["b"]
            return np.all(s >= -tol)
        if problem_id == "D1_SOCP":
            x = np.asarray(witness["x"])
            t = witness["t"]
            y = arrays["y"]
            return np.linalg.norm(x) <= arrays["R"] + tol and np.linalg.norm(y) <= t + tol
        if problem_id == "D2_BP":
            x = np.asarray(witness["x_feas"])
            A = arrays["A"]
            y = arrays["y"]
            return np.linalg.norm(A @ x - y) <= tol
        if problem_id == "D3_SDP":
            return witness.get("X") == "identity"
    if cert_type == "complementarity":
        if problem_id == "C2_LCP":
            z = np.asarray(witness["z"])
            q = arrays["q"]
            M = arrays["M"]
            w = M @ z + q
            return np.all(z >= -tol) and np.all(w >= -tol) and np.abs(z @ w) <= tol
    if cert_type is None:
        return True
    return False
