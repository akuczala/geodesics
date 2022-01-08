import itertools
import pickle

import numpy as np
import sympy as sp

# sc orbit
# [0,8,0],[1,0.0,0.03]
# metric = geodesics.metric_library.zee_metric_generator(3, 1.0)
from blenderhelper.draw_data import DrawDataList, CurveData, FrameData, ConeData
from geodesics.coordinate_map import CoordinateMap
from geodesics.draw import get_light_cone_vecs
from geodesics.geodesic_generator import TerminationCondition, GeodesicGenerator
from geodesics.metric_library import flat_polar, zee_metric_generator, ori_2007_metric_generator, \
    ori_1993_metric_generator
from geodesics.tangent_vector import TangentVector, TangentVectorType


def get_mapping():
    t, r, ph, x, y = sp.symbols('t r ph x y')
    return CoordinateMap(
        domain_coordinates=[t, r, ph],
        image_coordinates=[t, x, y],
        mapping=sp.Array([t, r * sp.cos(ph), r * sp.sin(ph)])
    )


#metric = ori_2007_metric_generator(3, 1.0)
print('generating metric')
metric = ori_1993_metric_generator(3, a_val=0.2, b_val=2, r0_val=3, d_val=1)
#metric = flat_polar(3)
print('calculating connections')
gg = GeodesicGenerator(metric, termination_condition=TerminationCondition.stop_on_coordinate_value(1, 0.02))
cartesian_mapping = get_mapping()

draw_data = DrawDataList.new()

for r0, ph0 in itertools.product(np.linspace(2.5,3.5,10),[0]):
    #    tau_range = np.concatenate(
    #            (np.arange(0,7, 0.1),
    #            np.arange(7,10, 0.01))
    #            )
    tau_range = np.arange(0, 15, 0.05)
    x0 = np.array([0, r0, ph0], dtype=float)
    u0 = np.array([1.0, 0.0, 0.0], dtype=float)
    tv = metric.normalize_tangent_vector(TangentVector(x=x0, u=u0))
    assert metric.classify_tangent_vector(tv) == TangentVectorType.TIMELIKE
    print('calculating geodesic')
    try:
        geo = gg.calc_geodesic(tv, tau_range)
        draw_data.append(CurveData([cartesian_mapping.eval(x) for x in geo.x]))
    except:
        print(f'failed to generate geodesic for r0={r0}')

    # print('calculating cones / frames')
    # for tv in geo.tv[::5]:
    #     frame = [
    #         (lambda v: 0.5 * v / np.linalg.norm(v))(cartesian_mapping.tangent_map(TangentVector(tv.x, v)).u)
    #         for v in metric.calc_spatial_basis_for_timelike_tangent(tv)
    #     ]
    #     light_cone_vecs = [
    #         0.5 * cartesian_mapping.tangent_map(TangentVector(tv.x, v)).u
    #         for v in get_light_cone_vecs(metric, timelike_tv=tv, n_vecs=16)
    #     ]
    #     draw_data.append(
    #         FrameData(
    #             point=cartesian_mapping.eval(tv.x),
    #             vecs=frame
    #         )
    #     )
    #     draw_data.append(
    #         ConeData(
    #             apex=cartesian_mapping.eval(tv.x),
    #             vecs=light_cone_vecs
    #         )
    #     )

with open('ori.pkl', 'wb') as f:
    pickle.dump(draw_data, f)
