from abc import abstractmethod, ABC
from typing import Dict

import sympy as sp

from geodesics.constants import SympySymbol
from geodesics.geodesic import Geodesic
from geodesics.metric_space import MetricSpace
from geodesics.tangent_vector import TangentVector


class GeodesicGenerator(ABC):
    def __init__(self, metric_space: MetricSpace, simplify_fn=lambda x: x):
        self.metric_space = metric_space
        u = sp.IndexedBase('u', shape=(metric_space.dim,))
        v = sp.IndexedBase('v', shape=(metric_space.dim,))
        uvec = sp.Array([u[i] for i in range(len(metric_space.coordinates))])
        vvec = sp.Array([v[i] for i in range(len(metric_space.coordinates))])
        self.y = metric_space.coordinates + tuple(uvec.tolist())
        self.y_pt = self.y + tuple(vvec.tolist())
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
