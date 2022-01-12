import numpy as np

from geodesics.coordinate_map_library import POLAR_MAPPING
from geodesics.draw import get_light_cone_vecs
from geodesics.metric_library import sc_metric_generator, zee_metric_generator
from geodesics.tangent_vector import TangentVector, TangentVectorType
from tests.utils import are_orthogonal, gen_timelike_vector


def test_basis_generation():
    metric = zee_metric_generator(3, 1.0)
    pos = np.array([0, 0.5, 0])
    basis = metric.calc_tangent_basis(pos)
    assert are_orthogonal(lambda v1, v2: metric.inner(v1, v2, pos), basis)


def test_tangent_basis_generation():
    metric = zee_metric_generator(3, 1.0)
    pos = np.random.random() * np.array([1, 3, 7]) + np.array([0, 0.01, 0])
    # print(pos)
    u = gen_timelike_vector(metric, pos)
    tv = metric.normalize_tangent_vector(TangentVector(x=pos, u=u))
    assert metric.classify_tangent_vector(tv) == TangentVectorType.TIMELIKE
    basis = metric.calc_spatial_basis_for_timelike_tangent(tv)
    assert are_orthogonal(lambda v1, v2: metric.inner(v1, v2, pos), [tv.u] + basis)
    # print(basis)
    # print([metric.classify_tangent_vector(TangentVector(u=b, x=pos)) for b in basis])
    assert all(
        metric.classify_tangent_vector(TangentVector(u=b, x=pos)) == TangentVectorType.SPACELIKE
        for b in basis
    )


# todo make this a real test
def test_spatial_basis():
    metric = zee_metric_generator(3, 1.0)
    return metric.calc_spatial_basis_for_timelike_tangent(
        TangentVector(
            x=np.array([0, 0.9, 0]),
            u=np.array([0.8, -0.2, 0])
        )
    )


metric = zee_metric_generator(3, 1.0)
x = np.array([0, 2, 0])
vecs = get_light_cone_vecs(metric, POLAR_MAPPING, TangentVector(x=np.array([0, 2, 0]), u=np.array([1, 0, 0])), 8)
print([metric.classify_tangent_vector(TangentVector(u=v, x=x)) for v in vecs])
