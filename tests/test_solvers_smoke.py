from uobench.core.spec import PROBLEM_REGISTRY
from uobench.solvers.gd import solve_gd
from uobench.solvers.prox import solve_fista, solve_projected_gd
from uobench.solvers.alm import solve_alm


def test_gradient_descent_runs_on_a1():
    inst = PROBLEM_REGISTRY["A1_QP"].generator(0, {"n": 12, "kappa": 200}, False)
    res = solve_gd("A1_QP", inst["data"], max_iter=10)
    assert res["status"] in {"converged", "max_iter"}


def test_fista_runs_on_b1():
    knobs = {"m": 30, "n": 15, "rho": 0.2, "sparsity": 3, "snr": 15.0, "lambda": 0.2}
    inst = PROBLEM_REGISTRY["B1_LASSO"].generator(1, knobs, False)
    res = solve_fista("B1_LASSO", inst["data"], max_iter=15)
    assert res["status"] in {"converged", "max_iter"}


def test_projected_gd_handles_trust_region():
    inst = PROBLEM_REGISTRY["A5_TRS"].generator(0, {"n": 10, "neg_ratio": 0.3, "delta": 1.0, "theta": 80}, False)
    res = solve_projected_gd("A5_TRS", inst["data"], max_iter=20)
    assert res["status"] in {"converged", "max_iter"}


def test_alm_solves_ecqp_smoke():
    inst = PROBLEM_REGISTRY["A4_ECQP"].generator(2, {"n": 15, "p": 4, "kappa": 100}, False)
    res = solve_alm("A4_ECQP", inst["data"], max_iter=10)
    assert res["status"] in {"converged", "max_iter"}
