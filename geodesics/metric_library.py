import numpy as np
import sympy as sp

from geodesics.coordinate_map import CoordinateMap
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector

r, ph = sp.symbols('r ph')
POLAR_MAPPING = CoordinateMap(
    domain_coordinates=sp.symbols('r ph'),
    image_coordinates=sp.symbols('x y'),
    mapping=sp.Array([r * sp.cos(ph), r * sp.sin(ph)])
)


def flat(dim: int) -> MetricSpace:
    return MetricSpace(
        coordinates=sp.symbols('t x y z')[:dim],
        params=tuple(),
        param_values={},
        # g=sp.Array(sp.diag([1 if i == 0 else -1 for i in range(dim)], dtype=np.float))
        g=sp.Array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ])[:dim, :dim],
    )


def flat_polar(dim: int) -> MetricSpace:
    t, r, th, ph, rs = sp.symbols('tb r θ φ rs')
    g = sp.Array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -r ** 2, 0],
        [0, 0, 0, -r ** 2 * sp.sin(th)]
    ])
    return MetricSpace(
        coordinates={2: (t, r), 3: (t, r, ph), 4: (t, r, th, ph)}[dim],
        params=tuple(),
        g=g[:dim, :dim],
        param_values={},
        spatial_to_cartesian_map=POLAR_MAPPING  # todo generalize to spherical
    )


def sc_metric_generator(dim: int, rs_val: float) -> MetricSpace:
    t, r, th, ph, rs = sp.symbols('t r θ φ rs')
    g = sp.Array([
        [(1 - rs / r), 0, 0, 0],
        [0, -(1 - rs / r) ** (-1), 0, 0],
        [0, 0, -r ** 2, 0],
        [0, 0, 0, -r ** 2 * sp.sin(th)]
    ])

    return MetricSpace(
        coordinates={2: (t, r), 3: (t, r, ph), 4: (t, r, th, ph)}[dim],
        params=(rs,),
        g=g[:dim, :dim],
        param_values={rs: rs_val},
        spatial_to_cartesian_map=POLAR_MAPPING  # todo generalize to spherical
    )


def zee_metric_generator(dim: int, rs_val: float) -> MetricSpace:
    tb, r, th, ph, rs = sp.symbols('tb r θ φ rs')
    g = -sp.Array([
        [(rs - r) / r, 2 * rs / r, 0, 0],
        [2 * rs / r, (r + rs) / r, 0, 0],
        [0, 0, r ** 2, 0],
        [0, 0, 0, r ** 2 * sp.sin(th)]
    ])
    return MetricSpace(
        coordinates={2: (tb, r), 3: (tb, r, ph), 4: (tb, r, th, ph)}[dim],
        params=(rs,),
        g=g[:dim, :dim],
        param_values={rs: rs_val},
        spatial_to_cartesian_map=POLAR_MAPPING
    )


def ori_2007_metric_generator(dim: int, mu_val: float) -> MetricSpace:
    v, r, th, ph, mu = sp.symbols('v r θ φ mu')
    g = sp.Array([
        [1 - 2 * mu / r, 2, 0, 0],
        [2, 0, 0, 0],
        [0, 0, r ** 2, 0],
        [0, 0, 0, r ** 2 * sp.sin(th)]
    ])
    return MetricSpace(
        coordinates={2: (v, r), 3: (v, r, ph), 4: (v, r, th, ph)}[dim],
        params=(mu,),
        g=g[:dim, :dim],
        param_values={mu: mu_val},
        spatial_to_cartesian_map=POLAR_MAPPING
    )


def ori_1993_metric_generator(dim: int, a_val, b_val, r0_val, d_val):
    assert a_val > 0 and b_val > 0 and r0_val > 0 and 0 < d_val < r0_val
    t, r, ph, z, a, b, r0, d = sp.symbols('t r φ z a b r0 d')
    rho = (r - r0) ** 2 + z ** 2
    h = sp.exp(-2 * ((r - r0) / d) ** 4)
    g_mink = sp.Array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -r ** 2, 0],
        [0, 0, 0, -1]
    ])
    t_dep = 2 * sp.sin(sp.pi * a * t / 2)**2
    #t_dep = a * t
    gm_t_ph = r * h * t_dep
    gm_r_ph = r * h * (-b) * (r - r0)
    gm_z_ph = r * h * z
    gm_ph_ph = (r * h) ** 2 * ((b * rho) ** 2 - t_dep ** 2)
    g_perturb = sp.Array([
        [0, 0, gm_t_ph, 0],
        [0, 0, gm_r_ph, 0],
        [gm_t_ph, gm_r_ph, gm_ph_ph, gm_z_ph],
        [0, 0, gm_z_ph, 0]
    ])
    g = sp.simplify(g_mink - g_perturb)
    if dim < 4:
        g = g.subs({z: 0})
    return MetricSpace(
        coordinates=(t, r, ph, z)[:dim],
        params=(a, b, r0, d),
        g=g[:dim, :dim],
        param_values={a: a_val, b: b_val, r0: r0_val, d: d_val},
        spatial_to_cartesian_map=POLAR_MAPPING
    )
