from dataclasses import dataclass
from typing import Dict

import numpy as np
import sympy as sp
from NumbaLSODA import lsoda_sig, lsoda
from numba import jit, cfunc, carray

from geodesics.constants import SympySymbol
from geodesics.geodesic import Geodesic, y_to_u, x_u_to_y
from geodesics.geodesic_generator import TerminationCondition
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


class NumbaGeodesicGenerator:
    def __init__(self, metric_space: MetricSpace, termination_condition=TerminationCondition.none(),
                 simplify_fn=lambda x: x):
        self.metric_space = metric_space
        u = sp.IndexedBase('u')
        uvec = sp.Array([u[i] for i in range(len(metric_space.coordinates))])
        self.y = metric_space.coordinates + tuple(uvec.tolist())
        self.termination_condition = termination_condition
        # ^i_jk u^m u^n
        # ^i_k u^n
        self.Guu = simplify_fn(sp.tensorcontraction(
            sp.tensorcontraction(
                sp.tensorproduct(metric_space.christ.subs(self.metric_space.param_values), uvec, uvec), (1, 3)
            ),
            (1, 2)
        ))
        Guu_arr = jit(nopython=True)(sp.lambdify([sp.Array(self.y)], sp.Matrix(self.Guu).T, 'numpy'))
        self.Guu_np = jit(nopython=True)(lambda v: Guu_arr(v).reshape(-1))
        self.ivp_fun = self.get_ivp_cfun(self.Guu_np)

    @property
    def param_values(self) -> Dict[SympySymbol, float]:
        return self.metric_space.param_values

    def get_ivp_fun_scipy(self, Guu_np):
        @jit(nopython=True)
        def ivp_fun(t, y):
            udot = -Guu_np(y)
            xdot = y_to_u(y)
            return x_u_to_y(xdot, udot)

        return ivp_fun

    def get_ivp_fun(self, Guu_np):
        @jit(nopython=True)
        def ivp_fun(t, y):
            udot = -Guu_np(y)
            xdot = y[len(y) // 2:]
            return np.concatenate((xdot, udot))

        return ivp_fun

    def get_ivp_cfun(self, Guu_np):
        ylen = self.metric_space.dim * 2

        @cfunc(lsoda_sig)
        def ivp_fun(t, y, dy, p):
            y_ = carray(y, (ylen,))
            udot = -Guu_np(y_)
            xdot = y_[ylen // 2:]
            dy_ = np.concatenate((xdot, udot))
            for i in range(len(dy_)):
                dy[i] = dy_[i]

        return ivp_fun

    def calc_geodesic(self, tv0: TangentVector, t_range) -> Geodesic:
        ysol, success = lsoda(self.ivp_fun.address, np.concatenate((tv0.x, tv0.u)), t_range)
        return Geodesic(SodaSolution(y=ysol.T, t=t_range))


@dataclass
class SodaSolution:
    y: np.ndarray
    t: np.ndarray
