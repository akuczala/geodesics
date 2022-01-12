import itertools
import pickle

import numpy as np
import sympy as sp

from blenderhelper.draw_data import DrawDataList, CurveData
from geodesics.coordinate_map import CoordinateMap
from geodesics.coordinate_map_library import POLAR_MAPPING
from geodesics.geodesic_generator import TerminationCondition, GeodesicGenerator
from geodesics.metric_library import morris_thorne_wormhole_generator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector, TangentVectorType


def get_coordinate_mapping(metric: MetricSpace):
    t, r, ph, x, y, z, b0 = sp.symbols('t r ph x y z b0')
    s = sp.sqrt(b0 ** 2 + r ** 2)
    return CoordinateMap(
        domain_coordinates=[t, r, ph],
        image_coordinates=[x, y, z],
        mapping=sp.Array([
            s * sp.cos(ph),
            s * sp.sin(ph),
            r / sp.sqrt(r * r) * b0 * sp.log(s / b0 + sp.sqrt((s / b0) ** 2 - 1))
        ]).subs(metric.param_values)
    )


print('generating metric')
metric = morris_thorne_wormhole_generator(3, b0_val=3.0)
print('calculating connections')
gg = GeodesicGenerator(metric, termination_condition=TerminationCondition.none())
coordinate_mapping = get_coordinate_mapping(metric)

draw_data = DrawDataList.new()


def make_null_geo_args(timelike_tv):
    assert metric.classify_tangent_vector(timelike_tv) == TangentVectorType.TIMELIKE
    tv = timelike_tv
    u0 = metric.calc_null_tangent(np.array([tv.u[0], 0, 0]), np.array([0, tv.u[1], tv.u[2]]), tv.x)
    null_tv = TangentVector(x=tv.x, u=u0)
    return null_tv, np.linspace(0, 40, 100)


th0 = np.pi * 7/8
dth = np.pi * 0.05
for r0, ph in itertools.product([8], np.linspace(th0 - dth, th0 + dth, 30)):
    x0 = np.array([0, r0, 0], dtype=float)

    direction = POLAR_MAPPING.tangent_inverse_map(
        domain_pos=x0[1:],
        image_vec=np.array([np.cos(ph), np.sin(ph)])
    )
    u0 = np.array([1.0, 0.1 * direction[0], 0.1 * direction[1]])
    print('calculating geodesic')
    geo = gg.calc_geodesic(*make_null_geo_args(TangentVector(x=x0, u=u0)))
    draw_data.append(CurveData([coordinate_mapping.eval(x) for x in geo.x]))

with open('wormhole_raytracing.pkl', 'wb') as f:
    pickle.dump(draw_data, f)
