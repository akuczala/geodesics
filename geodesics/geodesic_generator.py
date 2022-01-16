from abc import abstractmethod, ABC
from typing import Dict

import numpy as np
import sympy as sp

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


class GeodesicGenerator(ABC):
    def __init__(self, metric_space: MetricSpace, termination_condition=TerminationCondition.none(), simplify_fn=lambda x: x):
        self.metric_space = metric_space
        u = sp.IndexedBase('u')
        v = sp.IndexedBase('v')
        uvec = sp.Array([u[i] for i in range(len(metric_space.coordinates))])
        vvec = sp.Array([v[i] for i in range(len(metric_space.coordinates))])
        self.y = metric_space.coordinates + tuple(uvec.tolist())
        self.y_pt = self.y + tuple(vvec.tolist())
        self.termination_condition = termination_condition
        self.Guv = self.calc_Guv(simplify_fn, uvec, vvec)
        self.Guu = self.calc_Guu(simplify_fn, uvec)

    def calc_Guv(self, simplify_fn, uvec, vvec):
        # ^i_jk u^m v^n
        # ^i_k v^n
        return simplify_fn(sp.tensorcontraction(
            sp.tensorcontraction(
                sp.tensorproduct(self.metric_space.christ, uvec, vvec), (1, 3)
            ),
            (1, 2)
        ))

    def calc_Guu(self, simplify_fn, uvec):
        return self.calc_Guv(simplify_fn, uvec, uvec)

    # def calc_Guu(self, simplify_fn, uvec):
    #     # ^i_jk u^m u^n
    #     # ^i_k u^n
    #     return simplify_fn(sp.tensorcontraction(
    #         sp.tensorcontraction(
    #             sp.tensorproduct(self.metric_space.christ, uvec, uvec), (1, 3)
    #         ),
    #         (1, 2)
    #    ))
    @property
    def param_values(self) -> Dict[SympySymbol, float]:
        return self.metric_space.param_values

    @abstractmethod
    def calc_geodesic(self, tv0: TangentVector, t_range, use_jac=False, **kwargs) -> Geodesic:
        pass
