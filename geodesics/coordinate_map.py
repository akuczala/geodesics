from typing import List

import numpy as np
import sympy as sp
from sympy.matrices.common import NonInvertibleMatrixError, NonSquareMatrixError

from geodesics.constants import SympyMatrix, SympySymbol, SympyArray
from geodesics.tangent_vector import TangentVector


class CoordinateMap:
    def __init__(self, domain_coordinates: List[SympySymbol], image_coordinates: List[SympySymbol],
                 mapping: SympyArray, calc_jac=True, calc_inverse=True):
        self.domain_coordinates = domain_coordinates
        self.image_coordinates = image_coordinates
        self.mapping = mapping
        # todo njit these?
        mapping_arr = sp.lambdify([self.domain_coordinates], sp.Matrix(self.mapping).T, 'numpy')
        self.mapping_lambda = (lambda v: mapping_arr(v).reshape(-1))
        if calc_jac:
            self.jacobian = self.calc_jacobian()
            self.jacobian_lambda = sp.lambdify([self.domain_coordinates], self.jacobian, 'numpy')
            if calc_inverse:
                try:
                    self.inverse_jacobian = self.jacobian.inv()
                    # todo optional simplify
                    self.inverse_jacobian = sp.simplify(self.inverse_jacobian)
                    self.inverse_jacobian_lambda = sp.lambdify([self.domain_coordinates], self.inverse_jacobian, 'numpy')
                except NonSquareMatrixError:
                    print(f'Warning: jacobian {self.jacobian} is not square')
                except NonInvertibleMatrixError:
                    print(f'Warning: jacobian {self.jacobian} is non-invertible')

    def eval(self, domain_pos: np.ndarray) -> np.ndarray:
        return self.mapping_lambda(domain_pos)

    def calc_jacobian(self) -> SympyMatrix:
        return sp.derive_by_array(self.mapping, self.domain_coordinates).tomatrix().T

    def tangent_map(self, tv: TangentVector) -> TangentVector:
        return TangentVector(u=np.dot(self.jacobian_lambda(tv.x), tv.u), x=tv.x)

    def tangent_inverse_map(self, domain_pos: np.ndarray, image_vec: np.ndarray) -> np.ndarray:
        return np.dot(self.inverse_jacobian_lambda(domain_pos), image_vec)

    @staticmethod
    def coords_to_subs(coords: List[SympySymbol], values: np.ndarray):
        if len(coords) != len(values):
            raise ValueError(f"Coordinates {coords} and coordinate values {values} must have the same length")
        return {c: v for c, v in zip(coords, values)}
