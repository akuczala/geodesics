import sympy as sp

from geodesics.metric_space import MetricSpace


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
        param_values={rs: rs_val}
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
        param_values={rs: rs_val}
    )
