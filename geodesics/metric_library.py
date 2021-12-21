import sympy as sp

from geodesics.metric_space import MetricSpace


def sc_metric_generator(rs_val: float) -> MetricSpace:
    t, r, rs = sp.symbols('t r rs')
    return MetricSpace(
        coordinates=(t, r),
        params=(rs,),
        g=sp.Array([[(1 - rs / r), 0], [0, -(1 - rs / r) ** (-1)]]),
        param_values={rs: rs_val}
    )


def zee_metric_generator(rs_val: float) -> MetricSpace:
    tb, r, rs = sp.symbols('tb r rs')
    return MetricSpace(
        coordinates=(tb, r),
        params=(rs,),
        g=-sp.Array([
            [(rs - r) / r, 2 * rs / r],
            [2 * rs / r, (r + rs) / r]
        ]),
        param_values={rs: rs_val}
    )
