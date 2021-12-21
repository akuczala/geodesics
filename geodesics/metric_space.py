from typing import Tuple, Dict, List

import numpy as np
import sympy as sp

from geodesics.constants import SympySymbol, SympyArray, EPSILON, SympyMatrix
from geodesics.tangent_vector import TangentVector, TangentVectorType
from geodesics.utils import solve_real_quad, sympy_matrix_to_numpy


class MetricSpace:
    def __init__(self, coordinates: Tuple[SympySymbol], params: Tuple[SympySymbol], g: SympyArray,
                 param_values: Dict[SympySymbol, float]):
        self.coordinates = coordinates
        self.params = params
        self.g = g
        self.ginv = sp.Array(g.tomatrix().inv())
        self.christ = self.calc_christoffel()

        self.param_values = param_values

    @property
    def dim(self) -> int:
        return len(self.coordinates)

    def pos_to_subs_dict(self, pos) -> Dict[SympySymbol, float]:
        return {coord: xi for coord, xi in zip(self.coordinates, pos)}

    def eval_g(self, pos) -> np.ndarray:
        subs_dict = self.pos_to_subs_dict(pos)
        subs_dict.update(self.param_values)
        return np.array(self.g.subs(subs_dict).tolist(), dtype=np.float)

    def inner(self, v1, v2, pos) -> float:
        return np.dot(v1, np.dot(self.eval_g(pos), v2)).item()

    def normalize_tangent_vector(self, tv: TangentVector) -> TangentVector:
        assert self.classify_tangent_vector(tv) != TangentVectorType.NULL
        return TangentVector(x=tv.x, u=tv.u / np.sqrt(np.abs(self.tangent_vector_sqlen(tv))))

    def tangent_vector_sqlen(self, tv: TangentVector) -> float:
        return self.inner(tv.u, tv.u, tv.x)

    def classify_tangent_vector(self, tv: TangentVector) -> TangentVectorType:
        return TangentVectorType.len_to_type(self.tangent_vector_sqlen(tv))

    def calc_coordinate_tangents(self) -> List[SympyMatrix]:
        return [sp.Matrix([1 / sp.sqrt(abs(self.g[i, i])) if j == i else sp.S(0) for j in range(self.dim)]) for i in
                range(self.dim)]

    def get_coordinate_tangents_at_pos(self, x) -> List[TangentVector]:
        subs_dict = self.pos_to_subs_dict(x)
        return [
            TangentVector(x=x, u=sympy_matrix_to_numpy(u.subs(subs_dict).subs(self.param_values)).ravel())
            for u in self.calc_coordinate_tangents()
        ]

    def calc_null_tangent(self, v1, v2, pos) -> np.ndarray:
        """
        Calculate a null tangent vector from one timelike vector and one spacelike vector
        """
        vtype_dict = {self.classify_tangent_vector(TangentVector(x=pos, u=v)): v for v in (v1, v2)}
        vt, vs = vtype_dict[TangentVectorType.TIMELIKE], vtype_dict[TangentVectorType.SPACELIKE]
        dot = lambda v1, v2: self.inner(v1, v2, pos)
        # return vec of form vt + s vs
        quad_sols = solve_real_quad(a=dot(vs, vs), b=2 * dot(vt, vs), c=dot(vt, vt))
        positive_sols = [s for s in quad_sols if s > EPSILON]
        if len(positive_sols) == 0:
            raise ValueError(f"Could not solve for null vector using timelike {vt} and spacelike {vs}")
        return vt + positive_sols[0] * vs

    def calc_ortho_tangent_vector(self, tv: TangentVector, d) -> TangentVector:
        return TangentVector(x=tv.x, u=d - self.inner(tv.u, d, tv.x) / self.inner(tv.u, tv.u, tv.x) * tv.u)

    def calc_christoffel(self) -> SympyArray:
        dg = sp.permutedims(sp.derive_by_array(self.g, self.coordinates), (1, 2, 0))  # dg_ij/dx_k
        # dg_mk,l + dg_ml,k - dg_kl,m
        # m -> 0, k -> 1, l -> 2
        # (0,1,2), (0,2,1), (1,2,0)
        dg_permutes = dg + sp.permutedims(dg, (0, 2, 1)) - sp.permutedims(dg, (2, 1, 0))
        # g^ij (...)_mkl
        # dot j, m (1,2)
        return sp.tensorcontraction(
            sp.tensorproduct(self.ginv, dg_permutes),
            (1, 2)
        ) / 2
