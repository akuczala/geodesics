from typing import Dict

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from geodesics.constants import SympySymbol
from geodesics.geodesic import Geodesic, y_to_x, y_to_u, x_u_to_y
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


class TerminationCondition:
    def __init__(self, cond):
        self.cond = cond

    @classmethod
    def none(cls) -> "TerminationCondition":
        return cls(None)

    # @classmethod
    # def stop_on_sympy_true(cls, sympy_cond: SympyBoolean, metric: MetricSpace) -> "TerminationCondition":
    #     def cond(t, y):
    #         boolean(sympy_cond.subs(metric.pos_to_subs_dict(y[:len(y)//2])))
    @classmethod
    def stop_on_sympy_zero(cls, sympy_expr, metric: MetricSpace) -> "TerminationCondition":
        def cond(t, y, *args):
            return float(sympy_expr.subs(metric.pos_to_subs_dict(y_to_x(y))))

        cond.terminal = True
        return cls(cond)

    @classmethod
    def stop_on_coordinate_value(cls, coordinate_index: int, value: float) -> "TerminationCondition":
        def cond(t, y, *args):
            return y_to_x(y)[coordinate_index] - value

        cond.terminal = True
        return cls(cond)

    @property
    def condition(self):
        return self.cond


class GeodesicGenerator:
    def __init__(self, metric_space: MetricSpace, termination_condition=TerminationCondition.none(), simplify_fn=lambda x: x):
        self.metric_space = metric_space
        u = sp.IndexedBase('u')
        uvec = sp.Array([u[i] for i in range(len(metric_space.coordinates))])
        self.y = metric_space.coordinates + tuple(uvec.tolist())
        self.termination_condition = termination_condition
        # ^i_jk u^m u^n
        # ^i_k u^n
        self.Guu = simplify_fn(sp.tensorcontraction(
            sp.tensorcontraction(
                sp.tensorproduct(metric_space.christ, uvec, uvec), (1, 3)
            ),
            (1, 2)
        ))
        Guu_list = sp.lambdify(self.y + metric_space.params, self.Guu, 'numpy')
        self.Guu_np = lambda *args: np.array(Guu_list(*args))
        self.ivp_fun = self.get_ivp_fun()
        self.jac_fun = self.get_jac_fun(simplify_fn)

    @property
    def param_values(self) -> Dict[SympySymbol, float]:
        return self.metric_space.param_values

    def get_ivp_fun(self):
        def ivp_fun(t, y, *params):
            udot = -self.Guu_np(*y, *params)
            xdot = y_to_u(y)
            return x_u_to_y(xdot, udot)

        return ivp_fun

    def get_jac_fun(self, simplify_fn):
        # df_i/dy_j
        u = sp.IndexedBase('u')
        f = sp.Array([u[i] for i in range(len(self.metric_space.coordinates))] + list(self.Guu))
        jac_expr = simplify_fn(sp.permutedims(sp.derive_by_array(f, self.y), (1, 0)))
        jac_list = sp.lambdify(self.y + self.metric_space.params, jac_expr, 'numpy')
        jac_np = lambda *args: np.array(jac_list(*args))

        def jac_fun(t, y, *params):
            return jac_np(*y, *params)

        return jac_fun

    def calc_geodesic(self, tv0: TangentVector, t_range) -> Geodesic:
        return Geodesic(solve_ivp(
            self.ivp_fun, (t_range[0], t_range[-1]),
            t_eval=t_range,
            y0=np.concatenate((tv0.x, tv0.u)),
            args=tuple(self.metric_space.param_values[symbol] for symbol in self.metric_space.params),
            jac=self.jac_fun,
            dense_output=True,
            events=self.termination_condition.condition,
            method='Radau'
        ))
