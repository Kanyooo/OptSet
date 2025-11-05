"""Verification helpers combining structural and solvability checks.

This module provides two layers of checks:

1. Structural property validation (SPD, conditioning, SNR, feasibility, etc.).
2. Simple reference solvers that operate on S-scale instances to ensure the
   datasets are numerically tractable and yield small residuals.

The public entry point :func:`verify_instance` is used by the CLI and the test
suite to certify generated datasets.  Each verification returns a structured
report describing property diagnostics and, when available, the outcome of the
reference solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - import shim for direct script execution
    from .io import load_instance
except ImportError:  # pragma: no cover
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from uobgen.io import load_instance

@dataclass
class PropertyReport:
    """Structured result for property checks."""

    ok: bool
    messages: List[str]
    metrics: Dict[str, float]


@dataclass
class SolverReport:
    """Structured result for reference solver checks."""

    attempted: bool
    ok: bool
    residual: Optional[float]
    iterations: int
    message: str


@dataclass
class VerificationReport:
    """Combined verification outcome for a single instance directory."""

    path: Path
    problem_id: str
    scale: str
    properties: PropertyReport
    solver: SolverReport


def _spectral_norm(matrix: np.ndarray, iters: int = 25) -> float:
    """Estimate the spectral norm via the power method."""

    vec = np.random.default_rng(0).normal(size=matrix.shape[1])
    vec /= np.linalg.norm(vec) + 1e-12
    for _ in range(iters):
        vec = matrix.T @ (matrix @ vec)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return 0.0
        vec /= norm
    return float(np.linalg.norm(matrix @ vec))


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Soft-thresholding proximal operator."""

    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _check_a1_qp(data: Dict[str, np.ndarray], meta: Dict[str, object]) -> PropertyReport:
    Q = data["Q"]
    eigs = np.linalg.eigvalsh(Q)
    cond = float(np.max(eigs) / np.min(eigs))
    diagnostics = meta.get("diagnostics", {})
    target = float(diagnostics.get("cond", cond))
    messages: List[str] = []
    if np.min(eigs) <= 0:
        messages.append("Q is not positive definite")
    if abs(cond - target) > 0.1 * target:
        messages.append(f"condition number mismatch (got {cond:.2e}, target {target:.2e})")
    metrics = {"lambda_min": float(np.min(eigs)), "lambda_max": float(np.max(eigs)), "cond": cond}
    return PropertyReport(ok=not messages, messages=messages, metrics=metrics)


def _check_a4_ecqp(data: Dict[str, np.ndarray], _meta: Dict[str, object], reference: Dict[str, np.ndarray]) -> PropertyReport:
    A = data["A"]
    d = data["d"]
    x_star = reference.get("x_star")
    messages: List[str] = []
    residual = np.linalg.norm(A @ x_star - d) if x_star is not None else np.nan
    if x_star is None or residual > 1e-6 * (np.linalg.norm(d) + 1e-12):
        messages.append("stored reference solution violates constraints")
    metrics = {"feasibility": float(residual)}
    return PropertyReport(ok=not messages, messages=messages, metrics=metrics)


def _check_b1_lasso(data: Dict[str, np.ndarray], meta: Dict[str, object], reference: Dict[str, np.ndarray]) -> PropertyReport:
    A = data["A"]
    lam = float(data["lambda"])
    x_star = reference.get("x_star")
    messages: List[str] = []
    norms = np.linalg.norm(A, axis=0)
    if not np.allclose(norms[norms > 0], 1.0, atol=1e-2):
        messages.append("design matrix columns not normalized")
    if x_star is not None:
        y_clean = A @ x_star
        sigma = float(meta.get("diagnostics", {}).get("sigma", 1.0))
        snr_est = (np.linalg.norm(y_clean) ** 2) / (A.shape[0] * sigma**2 + 1e-12)
        target = float(meta.get("knobs", {}).get("snr", snr_est))
        if abs(snr_est - target) > 0.1 * target:
            messages.append(f"SNR mismatch (got {snr_est:.2f}, target {target:.2f})")
    metrics = {"lambda": lam, "max_col_norm": float(np.max(norms))}
    return PropertyReport(ok=not messages, messages=messages, metrics=metrics)


def _check_c1_vi(data: Dict[str, np.ndarray], meta: Dict[str, object]) -> PropertyReport:
    Q = data["Q"]
    eigs = np.linalg.eigvalsh(Q)
    mu = float(meta.get("knobs", {}).get("mu", 0.0))
    messages: List[str] = []
    if np.min(eigs) < mu - 1e-6:
        messages.append("monotonicity parameter violated")
    metrics = {"lambda_min": float(np.min(eigs))}
    return PropertyReport(ok=not messages, messages=messages, metrics=metrics)


def _check_c2_lcp(data: Dict[str, np.ndarray], meta: Dict[str, object]) -> PropertyReport:
    M = data["M"]
    eigs = np.linalg.eigvalsh(M)
    delta = float(meta.get("knobs", {}).get("delta", 0.0))
    messages: List[str] = []
    if np.min(eigs) < delta - 5e-2 * max(delta, 1e-12):
        messages.append("minimum eigenvalue below target delta")
    metrics = {"lambda_min": float(np.min(eigs)), "lambda_max": float(np.max(eigs))}
    return PropertyReport(ok=not messages, messages=messages, metrics=metrics)


def _check_d1_socp(data: Dict[str, np.ndarray], meta: Dict[str, object]) -> PropertyReport:
    A = data["A"]
    norms = np.linalg.norm(A, axis=0)
    messages: List[str] = []
    if not np.allclose(norms[norms > 0], 1.0, atol=1e-2):
        messages.append("design matrix columns not normalized")
    metrics = {"max_col_norm": float(np.max(norms)), "R": float(data["R"])}
    return PropertyReport(ok=not messages, messages=messages, metrics=metrics)


PROPERTY_CHECKS: Dict[str, Callable[[Dict[str, np.ndarray], Dict[str, object], Dict[str, np.ndarray]], PropertyReport]] = {
    "A1_QP": lambda data, meta, ref: _check_a1_qp(data, meta),
    "A4_ECQP": lambda data, meta, ref: _check_a4_ecqp(data, meta, ref),
    "B1_LASSO": _check_b1_lasso,
    "C1_VI": lambda data, meta, ref: _check_c1_vi(data, meta),
    "C2_LCP": lambda data, meta, ref: _check_c2_lcp(data, meta),
    "D1_SOCP": lambda data, meta, ref: _check_d1_socp(data, meta),
}


def _solve_a1_qp(data: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> SolverReport:
    Q = data["Q"]
    b = data["b"]
    x = np.linalg.solve(Q, b)
    residual = float(np.linalg.norm(Q @ x - b))
    return SolverReport(True, residual < 1e-8 * (np.linalg.norm(b) + 1e-12), residual, 1, "direct solve")


def _solve_a4_ecqp(data: Dict[str, np.ndarray]) -> SolverReport:
    Q = data["Q"]
    A = data["A"]
    b = data["b"]
    d = data["d"]
    n = Q.shape[0]
    p = A.shape[0]
    KKT = np.block([[Q, A.T], [A, np.zeros((p, p))]])
    rhs = np.concatenate([b, d])
    sol = np.linalg.solve(KKT, rhs)
    x = sol[:n]
    residual = float(np.linalg.norm(A @ x - d))
    return SolverReport(True, residual < 1e-6 * (np.linalg.norm(d) + 1e-12), residual, 1, "KKT solve")


def _solve_b1_lasso(data: Dict[str, np.ndarray], max_iter: int = 200) -> SolverReport:
    A = data["A"]
    y = data["y"]
    lam = float(data["lambda"])
    m, n = A.shape
    L = _spectral_norm(A) ** 2
    if L == 0:
        return SolverReport(False, False, None, 0, "singular design matrix")
    step = 1.0 / L
    x = np.zeros(n)
    z = x.copy()
    t = 1.0
    for k in range(max_iter):
        grad = A.T @ (A @ z - y)
        x_next = _soft_threshold(z - step * grad, lam * step)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        z = x_next + (t - 1) / t_next * (x_next - x)
        x, t = x_next, t_next
    grad = A.T @ (A @ x - y)
    kkt = np.zeros_like(x)
    active = np.abs(x) > 1e-8
    kkt[active] = grad[active] + lam * np.sign(x[active])
    inactive = ~active
    kkt[inactive] = np.maximum(np.abs(grad[inactive]) - lam, 0.0)
    residual = float(np.linalg.norm(kkt, ord=np.inf))
    ok = residual < 1e-4
    return SolverReport(True, ok, residual, max_iter, "FISTA (LASSO)")


def _solve_b3_svm(data: Dict[str, np.ndarray], max_iter: int = 200) -> SolverReport:
    A = data["A"]
    y = data["y"]
    gamma = float(data["gamma"])
    m, n = A.shape
    x = np.zeros(n)
    step = 1.0 / (gamma + _spectral_norm(A) ** 2 + 1e-12)
    for k in range(max_iter):
        margins = y * (A @ x)
        mask = margins < 1
        grad = gamma * x - (A[mask].T @ (y[mask])) / m
        x -= step * grad
    margins = y * (A @ x)
    violations = margins < 1
    subgrad = gamma * x - (A[violations].T @ (y[violations])) / m
    residual = float(np.linalg.norm(subgrad))
    ok = residual < 1e-4
    return SolverReport(True, ok, residual, max_iter, "Projected gradient (SVM)")


def _project_box(x: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, l), u)


def _solve_c1_vi(data: Dict[str, np.ndarray], max_iter: int = 300) -> SolverReport:
    Q = data["Q"]
    c = data["c"]
    set_type = data.get("set_type", "box")
    if set_type == "box":
        l = data["l"]
        u = data["u"]
        project = lambda z: _project_box(z, l, u)
    else:
        # Simplex projection via sorting
        def project(z: np.ndarray) -> np.ndarray:
            u_sorted = np.sort(z)[::-1]
            cssv = np.cumsum(u_sorted)
            rho = np.nonzero(u_sorted * np.arange(1, len(z) + 1) > (cssv - 1))[0][-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            w = np.maximum(z - theta, 0)
            return w

    L = _spectral_norm(Q)
    if L == 0:
        return SolverReport(False, False, None, 0, "singular operator")
    step = 1.0 / L
    x = np.zeros_like(c)
    for k in range(max_iter):
        grad = Q @ x + c
        x = project(x - step * grad)
    residual = float(np.linalg.norm(project(x - step * (Q @ x + c)) - x))
    ok = residual < 1e-4
    return SolverReport(True, ok, residual, max_iter, "Projected gradient (VI)")


def _solve_c2_lcp(data: Dict[str, np.ndarray], max_iter: int = 200) -> SolverReport:
    M = data["M"]
    q = data["q"]
    z = np.zeros_like(q)
    diag = np.diag(M)
    for k in range(max_iter):
        for i in range(M.shape[0]):
            w_i = q[i] + M[i, :] @ z
            if diag[i] == 0:
                continue
            z_i = z[i] - w_i / diag[i]
            z[i] = max(0.0, z_i)
    w = M @ z + q
    residual = float(max(np.linalg.norm(np.minimum(z, 0.0)), np.linalg.norm(np.minimum(w, 0.0)), np.linalg.norm(z * w)))
    ok = residual < 1e-4
    return SolverReport(True, ok, residual, max_iter * M.shape[0], "Projected Gauss-Seidel (LCP)")


def _solve_d1_socp(data: Dict[str, np.ndarray]) -> SolverReport:
    A = data["A"]
    y = data["y"]
    R = float(data["R"])
    AtA = A.T @ A
    Aty = A.T @ y
    try:
        x_ls = np.linalg.solve(AtA + 1e-8 * np.eye(AtA.shape[0]), Aty)
    except np.linalg.LinAlgError:
        return SolverReport(False, False, None, 0, "normal equations singular")
    norm_x = np.linalg.norm(x_ls)
    if norm_x <= R + 1e-8:
        x = x_ls
        lam = 0.0
    else:
        def norm_at_lambda(lam: float) -> float:
            sol = np.linalg.solve(AtA + lam * np.eye(AtA.shape[0]), Aty)
            return float(np.linalg.norm(sol))

        lam_low = 0.0
        lam_high = 1.0
        while norm_at_lambda(lam_high) > R:
            lam_high *= 2.0
        for _ in range(50):
            lam_mid = 0.5 * (lam_low + lam_high)
            if norm_at_lambda(lam_mid) > R:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
        x = np.linalg.solve(AtA + lam_high * np.eye(AtA.shape[0]), Aty)
        x_norm = np.linalg.norm(x)
        if x_norm > R:
            x *= R / (x_norm + 1e-12)
        lam = lam_high
    residual = float(np.linalg.norm(A @ x - y))
    feasible = np.linalg.norm(x) <= R + 1e-6
    grad = AtA @ x - Aty
    if lam == 0.0 or np.linalg.norm(x) < R - 1e-6:
        grad_res = np.linalg.norm(grad)
    else:
        grad_res = np.linalg.norm(grad + lam * x)
    ok = feasible and grad_res < 1e-4
    return SolverReport(True, ok, residual, 1, "Trust-region reduction (SOCP)")


SOLVER_CHECKS: Dict[str, Callable[[Dict[str, np.ndarray], Dict[str, np.ndarray]], SolverReport]] = {
    "A1_QP": lambda data, ref: _solve_a1_qp(data, ref),
    "A4_ECQP": lambda data, ref: _solve_a4_ecqp(data),
    "B1_LASSO": lambda data, ref: _solve_b1_lasso(data),
    "B3_SVM": lambda data, ref: _solve_b3_svm(data),
    "C1_VI": lambda data, ref: _solve_c1_vi(data),
    "C2_LCP": lambda data, ref: _solve_c2_lcp(data),
    "D1_SOCP": lambda data, ref: _solve_d1_socp(data),
}


def verify_instance(path: Path) -> VerificationReport:
    """Verify a generated instance directory."""

    loaded = load_instance(path)
    meta = loaded["meta"]
    problem_id = meta["id"]
    scale = meta.get("meta", {}).get("scale", "-")
    arrays = loaded["data"]
    reference = loaded.get("reference", {})
    meta_info = meta.get("meta", {})
    checker = PROPERTY_CHECKS.get(problem_id)
    if checker is None:
        prop_report = PropertyReport(True, [], {})
    else:
        prop_report = checker(arrays, meta_info, reference)
    solver_fn = SOLVER_CHECKS.get(problem_id)
    if solver_fn is None or scale not in {"S", "-"}:
        solver_report = SolverReport(False, False, None, 0, "not evaluated")
    else:
        solver_report = solver_fn(arrays, reference)
    return VerificationReport(path=path, problem_id=problem_id, scale=scale, properties=prop_report, solver=solver_report)


def verify_tree(root: Path) -> List[VerificationReport]:
    """Run verification on all instances contained in ``root`` recursively."""

    reports: List[VerificationReport] = []
    for meta_path in root.rglob("meta.json"):
        reports.append(verify_instance(meta_path.parent))
    return reports


def summarise_reports(reports: Iterable[VerificationReport]) -> str:
    """Create a human-readable summary table."""

    header = f"{'Problem':10s} {'Scale':5s} {'Props':7s} {'Solver':7s} Residual      Path"
    lines = [header, "-" * len(header)]
    for rep in reports:
        props = "OK" if rep.properties.ok else "FAIL"
        if rep.solver.attempted:
            solver_status = "OK" if rep.solver.ok else "FAIL"
            residual = f"{rep.solver.residual:.2e}" if rep.solver.residual is not None else "-"
        else:
            solver_status = "SKIP"
            residual = "-"
        lines.append(f"{rep.problem_id:10s} {rep.scale:5s} {props:7s} {solver_status:7s} {residual:12s} {rep.path}")
    return "\n".join(lines)
