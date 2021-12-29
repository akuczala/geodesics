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
        #g=sp.Array(sp.diag([1 if i == 0 else -1 for i in range(dim)], dtype=np.float))
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
    def position_to_cartesian(pos: np.ndarray) -> np.ndarray:
        t, r, ph = sp.symbols('t r ph')
        return sp.Array([t, r * sp.cos(ph), r * sp.sin(ph)]).subs()
    def tangent_vector_to_cartesian(tv: TangentVector):
        pass
    return MetricSpace(
        coordinates={2: (t, r), 3: (t, r, ph), 4: (t, r, th, ph)}[dim],
        params=tuple(),
        g=g[:dim, :dim],
        param_values={},
        spatial_to_cartesian_map=POLAR_MAPPING # todo generalize to spherical
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
        spatial_to_cartesian_map=POLAR_MAPPING # todo generalize to spherical
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
