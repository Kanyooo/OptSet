import numpy as np

from uobench.core.spec import PROBLEM_REGISTRY


def test_a1_qp_spd():
    spec = PROBLEM_REGISTRY["A1_QP"]
    inst = spec.generator(0, {"n": 16, "kappa": 100}, False)
    Q = inst["data"]["Q"]
    eigs = np.linalg.eigvalsh(0.5 * (Q + Q.T))
    assert eigs.min() > 0


def test_b1_lasso_snr():
    spec = PROBLEM_REGISTRY["B1_LASSO"]
    inst = spec.generator(0, {"m": 40, "n": 20, "sparsity": 0.1, "rho": 0.1, "snr": 10, "lambda": 0.1}, False)
    y = inst["data"]["y"]
    y_clean = inst["data"]["y_clean"]
    noise = y - y_clean
    snr_est = np.sum(y_clean**2) / (len(y) * np.var(noise))
    assert snr_est > 1


def test_c2_lcp_certificate():
    spec = PROBLEM_REGISTRY["C2_LCP"]
    inst = spec.generator(0, {"n": 10, "delta": 0.1}, False)
    witness = inst["witness"]
    assert witness["cert_type"] == "complementarity"
    assert np.allclose(witness["z"], 0)
