import numpy as np

from uobench.core.spec import PROBLEM_REGISTRY
from uobench.core.witness import verify


def test_a1_qp_condition_number():
    spec = PROBLEM_REGISTRY["A1_QP"]
    knobs = {"n": 32, "kappa": 1e3}
    inst = spec.generator(seed=0, knobs=knobs, extreme=False)
    Q = inst["data"]["Q"]
    eigs = np.linalg.eigvalsh(Q)
    cond = np.max(eigs) / np.min(eigs)
    assert 0.5 * knobs["kappa"] <= cond <= 2.0 * knobs["kappa"]


def test_a4_ecqp_witness_feasible():
    spec = PROBLEM_REGISTRY["A4_ECQP"]
    inst = spec.generator(seed=1, knobs={"n": 20, "p": 5, "kappa": 100}, extreme=False)
    meta = {"id": "A4_ECQP", "witness": inst["witness"]}
    ok = verify("A4_ECQP", meta, inst["data"])
    assert ok


def test_b1_lasso_snr_matches_target():
    spec = PROBLEM_REGISTRY["B1_LASSO"]
    knobs = {"m": 80, "n": 40, "rho": 0.2, "sparsity": 4, "snr": 25.0, "lambda": 0.1}
    inst = spec.generator(seed=0, knobs=knobs, extreme=False)
    y = inst["data"]["y"]
    y_clean = inst["data"]["y_clean"]
    noise = y - y_clean
    est = np.sum(y_clean**2) / (np.sum(noise**2) + 1e-12)
    assert 0.5 * knobs["snr"] <= est <= 2.0 * knobs["snr"]


def test_c2_lcp_delta_controls_spectrum():
    spec = PROBLEM_REGISTRY["C2_LCP"]
    knobs = {"n": 16, "delta": 1e-3, "cond": 1e2}
    inst = spec.generator(seed=2, knobs=knobs, extreme=False)
    M = inst["data"]["M"]
    eigs = np.linalg.eigvalsh(M)
    assert np.min(eigs) >= 0.5 * knobs["delta"]


def test_d3_sdp_laplacian_structure():
    spec = PROBLEM_REGISTRY["D3_SDP"]
    inst = spec.generator(seed=0, knobs={"n": 20, "p": 0.4}, extreme=False)
    L = inst["data"]["L"]
    assert np.allclose(L, L.T)
    degrees = inst["data"]["degrees"]
    assert np.allclose(np.diag(L), degrees)
