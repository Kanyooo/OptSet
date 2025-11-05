"""Problem registry for UOBGen."""
from __future__ import annotations

from typing import Callable, Dict

from .a_smooth import PROBLEMS as A_PROBLEMS
from .b_nonsmooth import PROBLEMS as B_PROBLEMS
from .c_semismooth import PROBLEMS as C_PROBLEMS
from .d_conic import PROBLEMS as D_PROBLEMS

Generator = Callable[..., dict]

PROBLEM_REGISTRY: Dict[str, Generator] = {}
PROBLEM_REGISTRY.update(A_PROBLEMS)
PROBLEM_REGISTRY.update(B_PROBLEMS)
PROBLEM_REGISTRY.update(C_PROBLEMS)
PROBLEM_REGISTRY.update(D_PROBLEMS)

PROBLEM_NAMES = {
    "A1_QP": "Strongly-convex Quadratic",
    "A2_LogReg": "L2 Logistic Regression",
    "A3_Rosenbrock": "Rosenbrock Chain",
    "A4_ECQP": "Equality-constrained QP",
    "A5_TRS": "Trust-region Subproblem",
    "A6_BoxQP": "Box-constrained QP",
    "B1_LASSO": "LASSO",
    "B2_ElasticNet": "Elastic Net",
    "B3_SVM": "Linear SVM",
    "B4_TV": "Isotropic TV",
    "B5_GroupLasso": "Group Lasso",
    "B6_NC_Sparse": "Nonconvex Sparse Logistic",
    "C1_VI": "Variational Inequality",
    "C2_LCP": "Linear Complementarity",
    "C3_MPCC": "Simple MPCC",
    "D1_SOCP": "SOCP Robust Regression",
    "D2_BP": "Basis Pursuit",
    "D3_SDP": "MaxCut SDP",
}

__all__ = ["PROBLEM_REGISTRY", "PROBLEM_NAMES"]
