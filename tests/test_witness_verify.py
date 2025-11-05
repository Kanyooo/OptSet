from uobench.core.spec import PROBLEM_REGISTRY
from uobench.core.witness import verify


def test_verify_a4():
    spec = PROBLEM_REGISTRY["A4_ECQP"]
    inst = spec.generator(0, {"n": 20, "p": 4, "kappa": 100}, False)
    ok = verify("A4_ECQP", {"witness": inst["witness"]}, inst["data"], tol=1e-6)
    assert ok


def test_verify_d2():
    spec = PROBLEM_REGISTRY["D2_BP"]
    inst = spec.generator(0, {"m": 10, "n": 20, "sparsity": 0.1}, False)
    ok = verify("D2_BP", {"witness": inst["witness"]}, inst["data"], tol=1e-6)
    assert ok
