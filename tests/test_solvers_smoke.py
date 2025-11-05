from uobench.core.spec import PROBLEM_REGISTRY
from uobench.solvers.gd import solve_gd
from uobench.solvers.prox import solve_fista
from uobench.solvers.alm import solve_alm


def test_gd_a1():
    inst = PROBLEM_REGISTRY["A1_QP"].generator(0, {"n": 8, "kappa": 100}, False)
    res = solve_gd("A1_QP", inst["data"], max_iter=5)
    assert res["status"] in {"converged", "max_iter"}


def test_fista_b1():
    inst = PROBLEM_REGISTRY["B1_LASSO"].generator(0, {"m": 20, "n": 10, "rho": 0.1, "sparsity": 0.2, "snr": 10, "lambda": 0.1}, False)
    res = solve_fista("B1_LASSO", inst["data"], max_iter=5)
    assert res["status"] in {"converged", "max_iter"}


def test_alm_a4():
    inst = PROBLEM_REGISTRY["A4_ECQP"].generator(0, {"n": 10, "p": 3, "kappa": 10}, False)
    res = solve_alm("A4_ECQP", inst["data"], max_iter=3)
    assert res["status"] in {"converged", "max_iter"}
