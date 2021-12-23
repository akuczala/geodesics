import numpy as np

from geodesics.utils import gram_schmidt


def test_gram_schmidt():
    d = np.random.randint(2, 6)
    randvecs = np.random.randn(d, d)
    while np.isclose(np.linalg.det(randvecs), 0):
        randvecs = np.random.randn(d, d)
    inner = np.dot
    gs_vecs = gram_schmidt(inner, randvecs)
    dots = np.array([[np.dot(v1, v2) for v1 in gs_vecs] for v2 in gs_vecs])
    check_dots = np.isclose(dots, 0) | np.eye(d).astype(bool)
    assert check_dots.all()

