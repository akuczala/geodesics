import itertools
import pickle
from typing import Tuple

import numpy as np
import sympy as sp

from blenderhelper.draw_data import DrawDataList, CurveData
from geodesics.coordinate_map import CoordinateMap
from geodesics.coordinate_map_library import POLAR_MAPPING, SPHERICAL_MAPPING
from geodesics.geodesic import Geodesic
from geodesics.geodesic_generator import TerminationCondition, GeodesicGenerator
from geodesics.metric_library import morris_thorne_wormhole_generator, flat_polar
from geodesics.metric_space import MetricSpace
from geodesics.sphere_projection import perspective_to_sphere
from geodesics.tangent_vector import TangentVector, TangentVectorType


def get_coordinate_mapping(metric: MetricSpace):
    t, r, ph, th, x, y, z = sp.symbols('t r ph th x y z')
    return CoordinateMap(
        domain_coordinates=[t, r, th, ph],
        image_coordinates=[x, y, z],
        mapping=sp.Array([
            r * sp.cos(ph) * sp.sin(th),
            r * sp.sin(ph) * sp.sin(th),
            r * sp.cos(th)
        ]).subs(metric.param_values)
    )


print('generating metric')
metric = flat_polar(4)
print('calculating connections')
gg = GeodesicGenerator(metric, termination_condition=TerminationCondition.none())
coordinate_mapping = get_coordinate_mapping(metric)

draw_data = DrawDataList.new()

def make_null_geo_args(timelike_tv: TangentVector) -> Tuple[TangentVector, np.ndarray]:
    assert metric.classify_tangent_vector(timelike_tv) == TangentVectorType.TIMELIKE
    tv = timelike_tv
    u0 = metric.calc_null_tangent_fast(
        np.array([tv.u[0], 0, 0, 0]),
        np.array([0, tv.u[1], tv.u[2], tv.u[3]]), tv.x, check=False)
    null_tv = TangentVector(x=tv.x, u=u0)
    return null_tv, np.linspace(0, 5, 30)

def get_geo_final(r0: float, th: float, ph: float) -> Geodesic:
    x0 = np.array([0, r0, np.pi / 2, 0], dtype=float)
    # todo: roughly 1/4 of time is spent calculating direction
    direction = SPHERICAL_MAPPING.tangent_inverse_map(
        domain_pos=x0[1:],
        image_vec=np.array([np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)])
    )
    u0 = np.array([1.0, 0.1 * direction[0], 0.1 * direction[1], 0.1 * direction[2]])
    geo = gg.calc_geodesic(*make_null_geo_args(TangentVector(x=x0, u=u0)))
    return geo

x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
ph_grid, lat_grid = np.vectorize(perspective_to_sphere, signature='(),()->(),()')(x_grid, y_grid)
ph_grid = ph_grid + np.pi
th_grid = np.pi/2 + lat_grid

r0 = 2.0
for ph, th in zip(ph_grid.ravel(),th_grid.ravel()):
    geo = get_geo_final(r0, th ,ph)
    draw_data.append(CurveData([coordinate_mapping.eval(x) for x in geo.x]))

with open('flat_raytracing.pkl', 'wb') as f:
    pickle.dump(draw_data, f)
