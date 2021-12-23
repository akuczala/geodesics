import numpy as np

from geodesics.geodesic_generator import GeodesicGenerator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


def get_null_vec_towards(metric_space: MetricSpace, timelike_tv: TangentVector, direction: np.ndarray):
    spacelike_v = metric_space.calc_ortho_tangent_vector(timelike_tv, direction).u
    return metric_space.calc_null_tangent(timelike_tv.u, spacelike_v, timelike_tv.x)