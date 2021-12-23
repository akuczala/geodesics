import numpy as np

from geodesics.metric_library import sc_metric_generator, zee_metric_generator
from tests.utils import are_orthogonal


def test_basis_generation():
    metric = zee_metric_generator(3, 1.0)
    pos = np.array([0, 0.5])
    basis = metric.calc_tangent_basis(pos)
    assert are_orthogonal(lambda v1, v2: metric.inner(v1,v2, pos), basis)

test_basis_generation()