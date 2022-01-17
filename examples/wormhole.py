import itertools
import pickle

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

from blenderhelper.draw_data import DrawDataList, CurveData
from geodesics.coordinate_map import CoordinateMap
from geodesics.geodesic_generator import TerminationCondition
from geodesics.metric_library import morris_thorne_wormhole_generator
from geodesics.metric_space import MetricSpace
from geodesics.scipy_geodesic_generator import ScipyGeodesicGenerator
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
gg = ScipyGeodesicGenerator(metric, termination_condition=TerminationCondition.none())
coordinate_mapping = get_coordinate_mapping(metric)

draw_data = DrawDataList.new()


def make_null_geo_args(timelike_tv):
    assert metric.classify_tangent_vector(timelike_tv) == TangentVectorType.TIMELIKE
    tv = timelike_tv
    u0 = metric.calc_null_tangent(np.array([tv.u[0], 0, 0]), np.array([0, tv.u[1], tv.u[2]]), tv.x)
    null_tv = TangentVector(x=tv.x, u=u0)
    return null_tv, np.linspace(0, 20, 50)


def make_timelike_geo_args(timelike_tv):
    assert metric.classify_tangent_vector(timelike_tv) == TangentVectorType.TIMELIKE
    return timelike_tv, np.linspace(0, 200, 50)

# ph0_range = np.linspace(0, 2 * np.pi, 20 )
ph0_range = [0]
for r0, ph0 in itertools.product([4], ph0_range):
    x0 = np.array([0, r0, ph0], dtype=float)
    u0 = np.array([1.0, -0.01, 0.00152], dtype=float)
    print('calculating geodesic')
    #geo = gg.calc_geodesic(*make_timelike_geo_args(TangentVector(x=x0, u=u0)))
    geo = gg.calc_geodesic(TangentVector(x=x0, u=u0), np.linspace(0,1600,50))
    fig, axes = plt.subplots(1,2)
    axes[0].plot(geo.x[:,1])
    axes[1].plot(geo.x[:, 2])
    #axes[2].plot(geo.x[:, 3])
    plt.show()
    draw_data.append(CurveData([coordinate_mapping.eval(x) for x in geo.x]))

with open('wormhole.pkl', 'wb') as f:
    pickle.dump(draw_data, f)
