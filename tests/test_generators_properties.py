import pytest
np = pytest.importorskip("numpy")

import pytest


from uobgen.problems.a_smooth import generate_a1_qp, generate_a4_ecqp, generate_a5_trs
from uobgen.problems.b_nonsmooth import generate_b1_lasso, generate_b2_elasticnet, generate_b4_tv
from uobgen.problems.c_semismooth import generate_c2_lcp
from uobgen.problems.d_conic import generate_d3_sdp


def test_a1_condition_number():
    inst = generate_a1_qp(seed=0, n=12, kappa=1e3)
    Q = inst["data"]["Q"]
    eigs = np.linalg.eigvalsh(Q)
    cond = eigs[-1] / eigs[0]
    assert cond == pytest.approx(1e3, rel=0.1)


def test_a4_spd():
    inst = generate_a4_ecqp(seed=0, n=12, p=4, kappa_Q=1e2)
    Q = inst["data"]["Q"]
    eigs = np.linalg.eigvalsh(Q)
    assert eigs.min() > 0


def _snr(y_clean, y_noisy):
    m = y_clean.size
    noise = y_noisy - y_clean
    power = np.sum(y_clean ** 2)
    sigma_sq = np.sum(noise ** 2) / m
    return power / sigma_sq


def test_b1_snr_calibration():
    inst = generate_b1_lasso(seed=0, m=40, n=12, snr=20.0)
    A = inst["data"]["A"]
    x_star = inst["reference"]["x_star"]
    y = inst["data"]["y"]
    snr = _snr(A @ x_star, y)
    assert snr == pytest.approx(20.0, rel=0.1)


def test_b2_snr_calibration():
    inst = generate_b2_elasticnet(seed=0, m=40, n=12, snr=15.0)
    A = inst["data"]["A"]
    x_star = inst["reference"]["x_star"]
    y = inst["data"]["y"]
    snr = _snr(A @ x_star, y)
    assert snr == pytest.approx(15.0, rel=0.1)


def test_a5_negative_ratio_and_angle():
    inst = generate_a5_trs(seed=0, n=20, neg_ratio=0.2, theta_deg=70)
    H = inst["data"]["H"]
    eigs = np.linalg.eigvalsh(H)
    negative_ratio = np.mean(eigs < 0)
    assert negative_ratio == pytest.approx(0.2, rel=0.2)
    g = inst["data"]["g"]
    v_min = np.linalg.eigh(H)[1][:, 0]
    cos_angle = abs(np.dot(g, v_min)) / (np.linalg.norm(g) * np.linalg.norm(v_min))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    assert angle >= 65


def test_b4_tv_gradients():
    inst = generate_b4_tv(seed=0, image_size=10, blocks=4)
    dx = inst["data"]["dx_edges"]
    dy = inst["data"]["dy_edges"]
    assert dx.shape[1] == 2 and dy.shape[1] == 2
    assert dx.shape[0] > 0 and dy.shape[0] > 0


def test_c2_lcp_delta():
    inst = generate_c2_lcp(seed=0, n=8, delta=1e-3)
    M = inst["data"]["M"]
    min_eig = np.min(np.linalg.eigvalsh(M))
    assert min_eig >= 1e-3 - 1e-6


def test_d3_sdp_properties():
    inst = generate_d3_sdp(seed=0, n=12, p=0.3)
    L = inst["data"]["L"]
    assert np.allclose(L, L.T, atol=1e-8)
    row_sums = L.sum(axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-6)
