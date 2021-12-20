import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from jupyterthemes import jtplot

from geodesics.geodesic_generator import GeodesicGenerator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


def plot_world_space_lines(metric_space):
    worldline_init_positions = [np.array([0, r0]) for r0 in np.linspace(1.2, 2, 5)]
    metric_gg = GeodesicGenerator(metric_space)
    tangents = metric_space.calc_coordinate_tangents()
    worldlines = [
        metric_gg.calc_geodesic(
            tv0=metric_space.get_coordinate_tangents_at_pos(x0)[0],
            t_span=(0, 1), n_pts=20
        ) for x0 in worldline_init_positions
    ]

    spacelines = [
        metric_gg.calc_geodesic(
            tv0=metric_space.normalize_tangent_vector(metric_space.calc_ortho_tangent_vector(tv, np.array([0, 1]))),
            t_span=(0, 0.3), n_pts=20
        ) for worldline in worldlines for tv in worldline.tv[::4]
    ]
    for geo in worldlines:
        plt.plot(geo.x[:, 1], geo.x[:, 0], c='red')
    for geo in spacelines:
        plt.plot(geo.x[:, 1], geo.x[:, 0], c='cyan')


def plot_light_cones(metric_space: MetricSpace):
    metric_gg = GeodesicGenerator(metric_space)
    r0_vals = np.linspace(1.1, 3, 5)
    t0_vals = np.linspace(0, 1, 5)
    for t0 in t0_vals:
        for r0 in r0_vals:
            x = np.array([t0, r0])
            for u0 in [metric_space.calc_null_tangent(np.array([1, 0]), np.array([0, s]), x) for s in (-1, 1)]:
                geo = metric_gg.calc_geodesic(
                    tv0=TangentVector(x=x, u=u0),
                    t_span=(0, 0.5), n_pts=20
                )
                plt.plot(geo.x[:, 1], geo.x[:, 0], c='yellow')


jtplot.style()

t, r, rs = sp.symbols('t r rs')
sc_metric = MetricSpace(
    coordinates=(t, r),
    params=(rs,),
    g=sp.Array([[(1 - rs / r), 0], [0, -(1 - rs / r) ** (-1)]]),
    param_values={rs: 1}
)
plot_world_space_lines(sc_metric)
plot_light_cones(sc_metric)
# plt.gca().set_aspect(1)
plt.show()
