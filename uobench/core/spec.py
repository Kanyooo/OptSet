"""Problem registry and suite specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from ..problems import a_smooth, b_nonsmooth, c_semismooth, d_conic


@dataclass
class ProblemSpec:
    problem_id: str
    name: str
    family: str
    generator: Callable[[int, Dict, bool], Dict]


PROBLEM_REGISTRY: Dict[str, ProblemSpec] = {
    "A1_QP": ProblemSpec("A1_QP", "Strongly-convex QP", "smooth", a_smooth.generate_a1_qp),
    "A2_LogReg": ProblemSpec("A2_LogReg", "L2-regularized Logistic Regression", "smooth", a_smooth.generate_a2_logreg),
    "A3_Rosenbrock": ProblemSpec("A3_Rosenbrock", "Rosenbrock Chain", "smooth", a_smooth.generate_a3_rosenbrock),
    "A4_ECQP": ProblemSpec("A4_ECQP", "Equality constrained QP", "smooth", a_smooth.generate_a4_ecqp),
    "A5_TRS": ProblemSpec("A5_TRS", "Trust-region subproblem", "smooth", a_smooth.generate_a5_trs),
    "A6_BoxQP": ProblemSpec("A6_BoxQP", "Box constrained QP", "smooth", a_smooth.generate_a6_boxqp),
    "B1_LASSO": ProblemSpec("B1_LASSO", "LASSO", "nonsmooth", b_nonsmooth.generate_b1_lasso),
    "B2_ElasticNet": ProblemSpec("B2_ElasticNet", "Elastic Net", "nonsmooth", b_nonsmooth.generate_b2_enet),
    "B3_SVM": ProblemSpec("B3_SVM", "Linear SVM", "nonsmooth", b_nonsmooth.generate_b3_svm),
    "B4_TV": ProblemSpec("B4_TV", "Isotropic TV denoising", "nonsmooth", b_nonsmooth.generate_b4_tv),
    "B5_GroupLasso": ProblemSpec("B5_GroupLasso", "Group Lasso", "nonsmooth", b_nonsmooth.generate_b5_group_lasso),
    "B6_NC_Sparse": ProblemSpec("B6_NC_Sparse", "Nonconvex sparse logistic", "nonsmooth", b_nonsmooth.generate_b6_nc_sparse),
    "C1_VI": ProblemSpec("C1_VI", "Linear VI", "vi", c_semismooth.generate_c1_vi),
    "C2_LCP": ProblemSpec("C2_LCP", "Linear Complementarity", "vi", c_semismooth.generate_c2_lcp),
    "C3_MPCC": ProblemSpec("C3_MPCC", "MPCC", "vi", c_semismooth.generate_c3_mpcc),
    "D1_SOCP": ProblemSpec("D1_SOCP", "SOCP robust regression", "conic", d_conic.generate_d1_socp),
    "D2_BP": ProblemSpec("D2_BP", "Basis pursuit", "conic", d_conic.generate_d2_bp),
    "D3_SDP": ProblemSpec("D3_SDP", "MaxCut SDP", "conic", d_conic.generate_d3_sdp),
}


SUITE_SPECS: Dict[str, Dict] = {}


def load_suite_spec(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.loads(fh.read())
    return data


def initialize_suites(config_dir: Path | None = None) -> None:
    global SUITE_SPECS
    config_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "config" / "suites"
    for path in config_dir.glob("*.yaml"):
        SUITE_SPECS[path.stem] = load_suite_spec(path)


initialize_suites()
