import numpy as np

from geodesics.geodesic import Geodesic
from geodesics.geodesic_generator import GeodesicGenerator, TerminationCondition
from geodesics.metric_library import sc_metric_generator, zee_metric_generator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


# todo figure out why worldline runs into horizon in zee case
def draw_test(x0list, u0list, t_range):
    metric = zee_metric_generator(3, 1.0)
    gg = GeodesicGenerator(metric, termination_condition=TerminationCondition.stop_on_coordinate_value(1, 0.02))
    print('zee')
    x0 = np.array(x0list,dtype=np.float)
    u0 = np.array(u0list,dtype=np.float)
    tv = metric.normalize_tangent_vector(TangentVector(x=x0, u=u0))
    print(metric.tangent_vector_sqlen(tv))
    geo = gg.calc_geodesic(tv, t_range)
    print(geo.x.shape)
    for tv in geo.tv:
        pass
        # print(metric.eval_g(x))
        # print(metric.christ.subs(metric.pos_to_subs_dict(x)).subs(metric.param_values))
        #print(gg.ivp_fun(0, np.concatenate((tv.x,tv.u)),*metric.param_values.values()))
    return geo
