import itertools
import pickle

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

from blenderhelper.draw_data import DrawDataList, CurveData
from geodesics.coordinate_map import CoordinateMap
from geodesics.metric_library import double_wormhole_generator
from geodesics.metric_space import MetricSpace
from geodesics.scipy_geodesic_generator import ScipyGeodesicGenerator, TerminationCondition
from geodesics.tangent_vector import TangentVector, TangentVectorType


def get_coordinate_mapping(metric: MetricSpace):
    t, s, ps, ph, a, beta = sp.symbols('t s ps ph a beta')
    x, y, z = sp.symbols('x y z')
    tau = -beta * sp.cos(ps)
    b = a / (sp.cosh(tau) - sp.cos(s))
    return CoordinateMap(
        domain_coordinates=[t, ps, s],
        image_coordinates=[x, y, z],
        mapping=sp.Array([
            b * sp.sinh(tau),
            b * sp.sin(s),
            sp.sin(ps)
        ]).subs(metric.param_values),
        calc_jac=False
    )


print('generating metric')
metric = double_wormhole_generator(3, a_val=3.0, beta_val=2)
print('calculating connections')
gg = ScipyGeodesicGenerator(metric, termination_condition=TerminationCondition.none())
print('calculating coordinate mapping')
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

print('calculating geodesics')
# ph0_range = np.linspace(0, 2 * np.pi, 20 )
ps_range = [0.4*np.pi]
#s_range = np.pi*(1+np.linspace(-0.05,0.05,40))
#s_range = np.pi + np.array([-1,0,1])*np.pi*0.025
s_range = [np.pi*1.01]
fig, axes = plt.subplots(1,2)
for ps, s in itertools.product(ps_range, s_range):
    x0 = np.array([0, ps, s], dtype=float)
    u0 = np.array([1.0, -0.005, 0.00], dtype=float)
    print('calculating geodesic')
    #geo = gg.calc_geodesic(*make_timelike_geo_args(TangentVector(x=x0, u=u0)))
    geo = gg.calc_geodesic(TangentVector(x=x0, u=u0), np.linspace(0,2000,50))
    axes[0].plot(geo.x[:,1])
    axes[1].plot(geo.x[:, 2])
    #axes[2].plot(geo.x[:, 3])

    draw_data.append(CurveData([coordinate_mapping.eval(x) for x in geo.x]))
plt.show()
with open('double_wormhole.pkl', 'wb') as f:
    pickle.dump(draw_data, f)
