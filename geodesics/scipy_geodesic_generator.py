import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from geodesics.geodesic import Geodesic, y_to_u, x_u_to_y
from geodesics.geodesic_generator import TerminationCondition, GeodesicGenerator
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


class ScipyGeodesicGenerator(GeodesicGenerator):
    def __init__(self, metric_space: MetricSpace, termination_condition=TerminationCondition.none(), simplify_fn=lambda x: x):
        super().__init__(metric_space, termination_condition, simplify_fn)
        Guu_list = sp.lambdify(self.y + metric_space.params, self.Guu, 'numpy')
        self.Guu_np = lambda *args: np.array(Guu_list(*args))
        self.ivp_fun = self.get_ivp_fun()
        self.jac_fun = self.get_jac_fun(simplify_fn)

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

    def calc_geodesic(self, tv0: TangentVector, t_range, use_jac=False, **kwargs) -> Geodesic:
        ivp_kwargs = {'dense_output': True}
        ivp_kwargs.update(kwargs)
        ivp_kwargs.update({'jac': self.jac_fun} if use_jac else {})
        return Geodesic(solve_ivp(
            self.ivp_fun, (t_range[0], t_range[-1]),
            t_eval=t_range,
            y0=np.concatenate((tv0.x, tv0.u)),
            args=tuple(self.metric_space.param_values[symbol] for symbol in self.metric_space.params),
            events=self.termination_condition.condition,
            **ivp_kwargs
        ))
