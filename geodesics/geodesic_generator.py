from typing import Dict

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from geodesics.constants import SympySymbol
from geodesics.geodesic import Geodesic
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


class GeodesicGenerator:
    def __init__(self, metric_space: MetricSpace):
        self.metric_space = metric_space
        u = sp.IndexedBase('u')
        uvec = sp.Array([u[i] for i in range(len(metric_space.coordinates))])
        self.y = metric_space.coordinates + tuple(uvec.tolist())
        # ^i_jk u^m u^n
        # ^i_k u^n
        Guu = sp.tensorcontraction(
            sp.tensorcontraction(
                sp.tensorproduct(metric_space.christ, uvec, uvec), (1, 3)
            ),
            (1, 2)
        )
        Guu_list = sp.lambdify(self.y + metric_space.params, Guu, 'numpy')
        self.Guu_np = lambda *args: np.array(Guu_list(*args))
        # Guu_jac = sp.derive_by_array(Guu,self.y)
        self.ivp_fun = self.get_ivp_fun()

    @property
    def param_values(self) -> Dict[SympySymbol, float]:
        return self.metric_space.param_values

    def get_ivp_fun(self):
        def ivp_fun(t, y, *params):
            udot = -self.Guu_np(*y, *params)
            xdot = y[len(y) // 2:]
            return np.concatenate((xdot, udot))

        return ivp_fun

    def calc_geodesic(self, tv0: TangentVector, t_span, n_pts) -> Geodesic:
        return Geodesic(solve_ivp(
            self.ivp_fun, t_span,
            t_eval=np.linspace(*t_span, n_pts),
            y0=np.concatenate((tv0.x, tv0.u)),
            args=tuple(self.metric_space.param_values[symbol] for symbol in self.metric_space.params),
            dense_output=True
        ))
