import pytest
np = pytest.importorskip("numpy")


from uobgen.problems.a_smooth import (
    generate_a1_qp,
    generate_a2_logreg,
    generate_a3_rosenbrock,
    generate_a4_ecqp,
    generate_a5_trs,
    generate_a6_boxqp,
)
from uobgen.problems.b_nonsmooth import (
    generate_b1_lasso,
    generate_b2_elasticnet,
    generate_b3_svm,
    generate_b4_tv,
    generate_b5_group_lasso,
    generate_b6_nonconvex_sparse,
)
from uobgen.problems.c_semismooth import (
    generate_c1_vi,
    generate_c2_lcp,
    generate_c3_mpcc,
)
from uobgen.problems.d_conic import (
    generate_d1_socp,
    generate_d2_basis_pursuit,
    generate_d3_sdp,
)


def test_a_generators_shapes():
    inst = generate_a1_qp(seed=0, n=10, kappa=1e3)
    assert inst["data"]["Q"].shape == (10, 10)
    inst2 = generate_a2_logreg(seed=0, m=20, n=5, rho=0.5)
    assert inst2["data"]["A"].shape == (20, 5)
    inst3 = generate_a3_rosenbrock(seed=0, n=6)
    assert inst3["data"]["x0"].shape == (6,)
    inst4 = generate_a4_ecqp(seed=0, n=12, p=3)
    assert inst4["data"]["A"].shape == (3, 12)
    inst5 = generate_a5_trs(seed=1, n=8)
    assert inst5["data"]["H"].shape == (8, 8)
    inst6 = generate_a6_boxqp(seed=0, n=7)
    assert inst6["data"]["H"].shape == (7, 7)


def test_b_generators_shapes():
    inst = generate_b1_lasso(seed=0, m=30, n=10)
    assert inst["data"]["A"].shape == (30, 10)
    inst = generate_b2_elasticnet(seed=1, m=30, n=10)
    assert inst["data"]["A"].shape == (30, 10)
    inst = generate_b3_svm(seed=0, m=40, n=6)
    assert inst["data"]["A"].shape == (40, 6)
    inst = generate_b4_tv(seed=0, image_size=8)
    assert inst["data"]["dx_edges"].shape[1] == 2
    inst = generate_b5_group_lasso(seed=0, m=40, n=12)
    assert inst["data"]["groups"].ndim == 2
    inst = generate_b6_nonconvex_sparse(seed=0, m=30, n=8)
    assert inst["data"]["A"].shape == (30, 8)


def test_c_generators_shapes():
    inst = generate_c1_vi(seed=0, n=6)
    assert inst["data"]["Q"].shape == (6, 6)
    inst = generate_c2_lcp(seed=0, n=5)
    assert inst["data"]["M"].shape == (5, 5)
    inst = generate_c3_mpcc(seed=0, n=6, p=3)
    assert inst["data"]["A"].shape == (3, 6)


def test_d_generators_shapes():
    inst = generate_d1_socp(seed=0, m=30, n=8)
    assert inst["data"]["A"].shape == (30, 8)
    inst = generate_d2_basis_pursuit(seed=0, m=10, n=15)
    assert inst["data"]["A"].shape == (10, 15)
    inst = generate_d3_sdp(seed=0, n=10)
    assert inst["data"]["L"].shape == (10, 10)
