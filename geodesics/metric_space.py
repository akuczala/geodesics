from typing import Tuple, Dict, List

import numpy as np
import sympy as sp

from geodesics.constants import SympySymbol, SympyArray, EPSILON, SympyMatrix
from geodesics.tangent_vector import TangentVector, TangentVectorType
from geodesics.utils import solve_real_quad, sympy_matrix_to_numpy, calc_orthogonal, gram_schmidt


class MetricSpace:
    def __init__(self, coordinates: Tuple[SympySymbol, ...], params: Tuple[SympySymbol, ...], g: SympyArray,
                 param_values: Dict[SympySymbol, float]):
        self.coordinates = coordinates
        self.params = params
        self.g = g
        self.ginv = sp.Array(g.tomatrix().inv())
        self.christ = self.calc_christoffel()

        self.param_values = param_values
        #d = self.dim
        #g_lambda_flat = njit(sp.lambdify([self.coordinates], g.subs(self.param_values).reshape(d * d), 'numpy'))
        #self.g_lambda = njit(lambda x: np.array(g_lambda_flat(x)).reshape(d,d))
        self.g_lambda = sp.lambdify([self.coordinates], sp.Matrix(g.subs(self.param_values)), 'numpy')

    @property
    def dim(self) -> int:
        return len(self.coordinates)

    def pos_to_subs_dict(self, pos) -> Dict[SympySymbol, float]:
        return {coord: xi for coord, xi in zip(self.coordinates, pos)}

    def eval_g(self, pos) -> np.ndarray:
        return self.g_lambda(pos)

    # todo return function
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

    def calc_null_tangent(self, v1: np.ndarray, v2: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """
        Calculate a null tangent vector from one timelike vector and one spacelike vector
        """
        v1_type, v2_type = [self.classify_tangent_vector(TangentVector(x=pos, u=v)) for v in (v1, v2)]
        vtype_dict = {v1_type: v1, v2_type: v2}
        if not (TangentVectorType.TIMELIKE in vtype_dict and TangentVectorType.SPACELIKE in vtype_dict):
            raise ValueError(f"Cannot calculate null vector with vectors {v1} ({v1_type}), {v2} ({v2_type})")
        vt, vs = vtype_dict[TangentVectorType.TIMELIKE], vtype_dict[TangentVectorType.SPACELIKE]
        dot = lambda v1, v2: self.inner(v1, v2, pos)
        # return vec of form vt + s vs
        quad_sols = solve_real_quad(a=dot(vs, vs), b=2 * dot(vt, vs), c=dot(vt, vt))
        positive_sols = [s for s in quad_sols if s > EPSILON]
        if len(positive_sols) == 0:
            raise ValueError(f"Could not solve for null vector using timelike {vt} and spacelike {vs}")
        null_vec = vt + positive_sols[0] * vs
        if self.classify_tangent_vector(TangentVector(x=pos,u=null_vec)) == TangentVectorType.NULL:
            return null_vec
        else:
            raise ValueError(f"Solution {null_vec} is not null. Input {v1},{v2} at position {pos}")

    def calc_null_tangent_fast(self, v_timelike: np.ndarray, v_spacelike: np.ndarray, pos: np.ndarray, check=True) -> np.ndarray:
        """
        Calculate a null tangent vector from one timelike vector and one spacelike vector.
        Not actually that much faster
        """
        if check:
            if not self.classify_tangent_vector(TangentVector(x=pos, u=v_timelike)) == TangentVectorType.TIMELIKE:
                raise ValueError(f"{v_timelike} is not timelike")
            if not self.classify_tangent_vector(TangentVector(x=pos, u=v_spacelike)) == TangentVectorType.SPACELIKE:
                raise ValueError(f"{v_spacelike} is not spacelike")
        vt, vs = v_timelike, v_spacelike
        dot = lambda v1, v2: self.inner(v1, v2, pos)
        # return vec of form vt + s vs
        quad_sols = solve_real_quad(a=dot(vs, vs), b=2 * dot(vt, vs), c=dot(vt, vt))
        positive_sols = [s for s in quad_sols if s > EPSILON]
        if len(positive_sols) == 0:
            raise ValueError(f"Could not solve for null vector using timelike {vt} and spacelike {vs}")
        null_vec = vt + positive_sols[0] * vs
        if self.classify_tangent_vector(TangentVector(x=pos,u=null_vec)) == TangentVectorType.NULL:
            return null_vec
        else:
            raise ValueError(f"Solution {null_vec} is not null. Input {vt},{vs} at position {pos}")

    def calc_ortho_tangent_vector(self, tv: TangentVector, d) -> TangentVector:
        return TangentVector(x=tv.x, u=calc_orthogonal(lambda v1, v2: self.inner(v1, v2, tv.x), tv.u, d))

    def calc_tangent_basis(self, pos: np.ndarray):
        return gram_schmidt(lambda v1, v2: self.inner(v1, v2, pos), np.eye(self.dim))

    def calc_spatial_basis_for_timelike_tangent(self, tv: TangentVector):
        if not self.classify_tangent_vector(tv) == TangentVectorType.TIMELIKE:
            raise ValueError(f"Input {tv} is not timelike")
        pos = tv.x
        timelike_v = tv.u
        drop_i = np.argmax(np.abs(timelike_v))
        # rm basis vector with largest euclidean projection onto timelike vec
        basis = np.delete(np.eye(self.dim), drop_i, axis=0)
        basis = [timelike_v] + list(basis)
        #print(basis)
        # spacelike_vecs = [
        #     v for v in vecs
        #     if self.classify_tangent_vector(TangentVector(x=pos, u=v)) == TangentVectorType.SPACELIKE
        # ][:self.dim-1]
        # print(spacelike_vecs)
        return gram_schmidt(lambda v1, v2: self.inner(v1, v2, pos), basis)[1:]

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

    def calc_riemann(self):
        #mu rho nu sigma
        dchrist = sp.derive_by_array(self.christ, self.coordinates)
        #rho mu nu sigma
        christ_sq = sp.tensorcontraction(sp.tensorproduct(self.christ, self.christ), (2, 3))
        #rho sigma mu nu
        half_R = (
                sp.permutedims(dchrist, (1,3,0,2))
                + sp.permutedims(christ_sq, (0,3,1,2)))
        return half_R - sp.permutedims(half_R, (0,1,3,2))
