from dataclasses import dataclass

import numpy as np
import sympy as sp
from NumbaLSODA import lsoda_sig, lsoda
from numba import cfunc, carray, njit

from geodesics.geodesic import Geodesic
from geodesics.geodesic_generator import TerminationCondition, GeodesicGenerator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


class NumbaGeodesicGenerator(GeodesicGenerator):
    def __init__(self, metric_space: MetricSpace,
                 simplify_fn=lambda x: x):
        super().__init__(metric_space, TerminationCondition.none(), simplify_fn)
        Guu_arr = njit(sp.lambdify([sp.Array(self.y)], sp.Matrix(self.Guu).T, 'numpy'))
        self.Guu_np = njit(lambda v: Guu_arr(v).reshape(-1))
        self.ivp_fun = self.get_ivp_fun()

    def get_ivp_fun(self):
        ylen = self.metric_space.dim * 2
        Guu_np = self.Guu_np

        @cfunc(lsoda_sig)
        def ivp_fun(t, y, dy, p):
            y_ = carray(y, (ylen,))
            udot = -Guu_np(y_)
            xdot = y_[ylen // 2:]
            dy_ = np.concatenate((xdot, udot))
            for i in range(len(dy_)):
                dy[i] = dy_[i]

        return ivp_fun

    def calc_geodesic(self, tv0: TangentVector, t_range: np.ndarray, **kwargs) -> Geodesic:
        ysol, success = lsoda(self.ivp_fun.address, np.concatenate((tv0.x, tv0.u)), t_range)
        return Geodesic(SodaSolution(y=ysol.T, t=t_range))


@dataclass
class SodaSolution:
    y: np.ndarray
    t: np.ndarray
