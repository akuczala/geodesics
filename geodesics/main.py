import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from jupyterthemes import jtplot

from geodesics.geodesic import Geodesic
from geodesics.geodesic_generator import GeodesicGenerator
from geodesics.metric_library import sc_metric_generator, zee_metric_generator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


def plot_world_space_lines(metric_space):
    worldline_init_positions = [np.array([0, r0]) for r0 in np.linspace(1.2, 2, 5)]
    metric_gg = GeodesicGenerator(metric_space)
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


def plot_light_cone_grid(metric_space: MetricSpace):
    metric_gg = GeodesicGenerator(metric_space)
    r0_vals = np.linspace(1.1, 3, 5)
    t0_vals = np.linspace(0, 1, 5)
    for t0 in t0_vals:
        for r0 in r0_vals:
            plot_light_cone(metric_gg, np.array([t0, r0]))


def plot_along_geodesic(gg: GeodesicGenerator, plot_fun, geo: Geodesic, step=1):
    for tv in geo.tv[::step]:
        plot_fun(gg, tv)


def plot_light_cone(gg: GeodesicGenerator, tv: TangentVector):
    timelike_v = tv.u
    spacelike_v = gg.metric_space.calc_ortho_tangent_vector(tv, np.array([0, 1])).u
    for u0 in [gg.metric_space.calc_null_tangent(timelike_v, s * spacelike_v, tv.x) for s in (-1, 1)]:
        geo = gg.calc_geodesic(
            tv0=TangentVector(x=tv.x, u=u0),
            t_span=(0, 0.1), n_pts=20
        )
        plt.plot(geo.x[:, 1], geo.x[:, 0], c='yellow')


def plot_tangent_light_cone(gg: GeodesicGenerator, tv: TangentVector, size=1):
    timelike_v = tv.u
    spacelike_v = gg.metric_space.calc_ortho_tangent_vector(tv, np.array([0, 1])).u
    for s in (-1, 1):
        null_vec = gg.metric_space.calc_null_tangent(timelike_v, s * spacelike_v, tv.x)
        plot_vec = size * null_vec / np.linalg.norm(null_vec)
        p0 = tv.x
        p1 = p0 + plot_vec
        plt.plot([p0[1], p1[1]], [p0[0], p1[0]], c='yellow')


def plot_test(metric: MetricSpace):
    gg = GeodesicGenerator(metric)

    for x0 in [np.array([0, r0]) for r0 in (1.5, 2, 4)]:
        worldline = gg.calc_geodesic(
            metric.get_coordinate_tangents_at_pos(x0)[0],
            t_span=(0, 6), n_pts=20
        )
        plt.plot(worldline.x[:, 1], worldline.x[:, 0], c='red')
        plot_along_geodesic(
            GeodesicGenerator(metric),
            plot_light_cone,
            geo=worldline,
            step=4
        )
    plt.gca().set_aspect(1)
    plt.show()


def plot_test_3(metric: MetricSpace):
    gg = GeodesicGenerator(metric)
    for r0 in (2,3,4,5):
        worldline = gg.calc_geodesic(
            metric.normalize_tangent_vector(TangentVector(x=np.array([0, r0]), u=np.array([0.9, 0.1]))),
            t_span=(0, 12), n_pts=20
        )
        plt.plot(worldline.x[:, 1], worldline.x[:, 0], c='red')
        plot_along_geodesic(
            GeodesicGenerator(metric),
            lambda gg, tv: plot_tangent_light_cone(gg, tv, size=0.2),
            geo=worldline,
            step=1
        )
    plt.gca().set_aspect(0.5)
    plt.show()


def plot_test_2(gg: GeodesicGenerator):
    for x0 in [np.array([t0, 4.0]) for t0 in range(4)]:
        u = gg.metric_space.calc_null_tangent(np.array([1, 0]), np.array([0, -1]), x0)
        geo = gg.calc_geodesic(TangentVector(x=x0, u=u), t_span=(0, 4), n_pts=20)
        plt.plot(geo.x[:, 1], geo.x[:, 0], c='yellow')
    for x0 in [np.array([t0, 0.5]) for t0 in range(4)]:
        u = gg.metric_space.calc_null_tangent(np.array([1, -0.4]), np.array([0, 1]), x0)
        geo = gg.calc_geodesic(TangentVector(x=x0, u=u), t_span=(0, 4), n_pts=20)
        plt.plot(geo.x[:, 1], geo.x[:, 0], c='yellow')
    plt.gca().set_aspect(1)
    plt.show()


jtplot.style()
metric = zee_metric_generator(1)
# plot_test_2(GeodesicGenerator(metric))
plot_test_3(metric)
# metric.get_coordinate_tangents_at_pos(np.array([0, 1.0]))

# plot_world_space_lines(metric )
