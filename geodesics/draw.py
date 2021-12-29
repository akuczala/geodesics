from typing import List

import numpy as np

from geodesics.geodesic_generator import GeodesicGenerator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector
from tests.utils import gen_S1_vecs, normalize_euclid


def get_null_vec_towards(metric_space: MetricSpace, timelike_tv: TangentVector, spacelike_v: np.ndarray):
    return metric_space.calc_null_tangent(timelike_tv.u, spacelike_v, timelike_tv.x)


# todo generalize beyond 2 + 1 spacetime
def get_light_cone_vecs(metric_space: MetricSpace, timelike_tv: TangentVector, n_vecs: int) -> List[np.ndarray]:
    spatial_frame = metric_space.calc_spatial_basis_for_timelike_tangent(timelike_tv)
    assert len(spatial_frame) == 2
    cartesian_s1_vecs = gen_S1_vecs(n_vecs)
    transformed_s1_vecs = [
        metric_space.spatial_to_cartesian_map.tangent_inverse_map(timelike_tv.x[1:], v) for v in cartesian_s1_vecs
    ]

    # print([
    #     metric_space.classify_tangent_vector(TangentVector(x=timelike_tv.x, u=spatial_frame[0] * s1_vec[0] + spatial_frame[1] * s1_vec[1]))
    #     for s1_vec in s1_vecs
    # ])
    return [
        #normalize_euclid(
        get_null_vec_towards(
            metric_space, timelike_tv,
            spatial_frame[0] * s1_vec[0] + spatial_frame[1] * s1_vec[1]
        )
        #)
        for s1_vec in transformed_s1_vecs
    ]

# def draw_tangent_light_cone(self, metric: MetricSpace, tv: TangentVector, n_vecs: int):
#     null_vecs = get_light_cone_vecs(metric, tv, n_vecs)
#     pos_vert = self.bm.verts.new(self.tangent_to_3d(TangentVector(x=tv.x, u=np.zeros_like(tv.x))))
#     cone_verts = [self.bm.verts.new(self.tangent_to_3d(TangentVector(x=tv.x, u=v))) for v in null_vecs]
#     for vert in cone_verts:
#         bmesh.ops.contextual_create(self.bm, geom=[pos_vert, vert])
#     return self.bm
