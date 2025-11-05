from uobench.core.spec import PROBLEM_REGISTRY
from uobench.core.witness import verify


def test_a4_ecqp_witness_passes():
    inst = PROBLEM_REGISTRY["A4_ECQP"].generator(0, {"n": 30, "p": 5, "kappa": 200}, False)
    meta = {"id": "A4_ECQP", "witness": inst["witness"]}
    assert verify("A4_ECQP", meta, inst["data"])


def test_c2_lcp_witness_passes():
    inst = PROBLEM_REGISTRY["C2_LCP"].generator(0, {"n": 10, "delta": 1e-2}, False)
    meta = {"id": "C2_LCP", "witness": inst["witness"]}
    assert verify("C2_LCP", meta, inst["data"])


def test_d2_bp_witness_reproduces_measurements():
    inst = PROBLEM_REGISTRY["D2_BP"].generator(0, {"m": 15, "n": 40, "sparsity": 4}, False)
    meta = {"id": "D2_BP", "witness": inst["witness"]}
    assert verify("D2_BP", meta, inst["data"])
