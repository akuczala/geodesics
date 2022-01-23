import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from geodesics.geodesic import Geodesic
from geodesics.geodesic_generator import GeodesicGenerator
from geodesics.metric_space import MetricSpace
from geodesics.numpy_geodesic import NumpyGeodesic
from geodesics.numpy_geodesic_transport import NumpyGeodesicTransport
from geodesics.tangent_vector import TangentVector


def y_to_x(y: np.ndarray) -> np.ndarray:
    return y[:len(y) // 2]


def y_to_u(y: np.ndarray) -> np.ndarray:
    return y[len(y) // 2:]


def x_u_to_y(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate((x, u))


def tv_to_y(tv: TangentVector):
    return x_u_to_y(tv.x, tv.u)


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


class ScipyGeodesicGenerator(GeodesicGenerator):
    def __init__(self, metric_space: MetricSpace, termination_condition=TerminationCondition.none(),
                 simplify_fn=lambda x: x):
        super().__init__(metric_space, simplify_fn)
        self.termination_condition = termination_condition
        Guu_list = sp.lambdify(self.y + metric_space.params, self.Guu, 'numpy')
        self.Guu_np = lambda *args: np.array(Guu_list(*args))
        Guv_list = sp.lambdify(self.y_pt + metric_space.params, self.Guv, 'numpy')
        self.Guv_np = lambda *args: np.array(Guv_list(*args))
        self.ivp_fun = self.get_ivp_fun()
        self.jac_fun = self.get_jac_fun(simplify_fn)

        self.pt_ivp_fun = self.get_pt_ivp_fun()

    def get_ivp_fun(self):
        def ivp_fun(t, y, *params):
            udot = -self.Guu_np(*y, *params)
            xdot = y_to_u(y)
            return x_u_to_y(xdot, udot)

        return ivp_fun

    def get_pt_ivp_fun(self):
        def ivp_fun(t, y_pt, *params):
            y = y_pt[:2 * len(y_pt) // 3]
            udot = -self.Guu_np(*y, *params)
            vdot = -self.Guv_np(*y_pt, *params)
            xdot = y_to_u(y)
            return np.concatenate((x_u_to_y(xdot, udot), vdot))

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
        sol = solve_ivp(
            self.ivp_fun, (t_range[0], t_range[-1]),
            t_eval=t_range,
            y0=np.concatenate((tv0.x, tv0.u)),
            args=tuple(self.metric_space.param_values[symbol] for symbol in self.metric_space.params),
            events=self.termination_condition.condition,
            **ivp_kwargs
        )
        return NumpyGeodesic(x=y_to_x(sol.y).T, u=y_to_u(sol.y).T, tau_range=sol.t)

    def calc_parallel_transport(self, tv0: TangentVector, v0: np.ndarray, t_range, use_jac=False, raise_on_fail=False,
                                **kwargs) -> NumpyGeodesicTransport:
        ivp_kwargs = {'dense_output': True}
        ivp_kwargs.update(kwargs)
        ivp_kwargs.update({'jac': self.jac_fun} if use_jac else {})
        sol = solve_ivp(
            self.pt_ivp_fun, (t_range[0], t_range[-1]),
            t_eval=t_range,
            y0=np.concatenate((tv0.x, tv0.u, v0)),
            args=tuple(self.metric_space.param_values[symbol] for symbol in self.metric_space.params),
            events=self.termination_condition.condition,
            **ivp_kwargs
        )
        if raise_on_fail and not sol.success:
            raise Exception(f'Solver failed to find solution: {sol.message}')
        geo_y = sol.y[:2 * len(sol.y) // 3]
        v = sol.y[2 * len(sol.y) // 3:]
        return NumpyGeodesicTransport(x=y_to_x(geo_y).T, u=y_to_u(geo_y).T, tau_range=sol.t, v=v.T)
