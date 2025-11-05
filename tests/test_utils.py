import pytest
np = pytest.importorskip("numpy")


from uobgen.utils import column_normalize, geometric_spectrum, haar_orthogonal, toeplitz_corr


def test_haar_orthogonal_identity():
    rng = np.random.default_rng(0)
    Q = haar_orthogonal(5, rng)
    should_be_identity = Q.T @ Q
    assert np.allclose(should_be_identity, np.eye(5), atol=1e-6)


def test_geometric_spectrum_condition_number():
    eigs = geometric_spectrum(10, 1.0, 100.0)
    cond = eigs[-1] / eigs[0]
    assert np.isclose(cond, 100.0)


def test_toeplitz_corr_structure():
    Sigma = toeplitz_corr(4, 0.5)
    assert np.allclose(Sigma[0], [1.0, 0.5, 0.25, 0.125])


def test_column_normalize():
    A = np.array([[3.0, 0.0], [4.0, 2.0]])
    normed, norms = column_normalize(A)
    assert np.allclose(norms, [5.0, np.sqrt(4)])
    assert np.allclose(np.linalg.norm(normed, axis=0), 1.0)
