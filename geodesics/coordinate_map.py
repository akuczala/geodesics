from typing import List

import numpy as np
import sympy as sp
from sympy.matrices.common import NonInvertibleMatrixError

from geodesics.constants import SympyMatrix, SympySymbol, SympyArray
from geodesics.tangent_vector import TangentVector
from geodesics.utils import sympy_matrix_to_numpy


class CoordinateMap:
    def __init__(self, domain_coordinates: List[SympySymbol], image_coordinates: List[SympySymbol], mapping : SympyArray):
        self.domain_coordinates = domain_coordinates
        self.image_coordinates = image_coordinates
        self.mapping = mapping
        self.jacobian = self.calc_jacobian()
        try:
            self.inverse_jacobian = self.jacobian.inv()
            self.inverse_jacobian = sp.simplify(self.inverse_jacobian)
        except NonInvertibleMatrixError:
            print(f'Warning: jacobian {self.jacobian} is non-invertible')

    def eval(self, domain_pos: np.ndarray) -> np.ndarray:
        return np.array(self.mapping.subs(self.coords_to_subs(self.domain_coordinates, domain_pos)), dtype=float)

    def calc_jacobian(self) -> SympyMatrix:
        return sp.derive_by_array(self.mapping, self.domain_coordinates).tomatrix().T

    def tangent_map(self, tv: TangentVector) -> TangentVector:
        u_image = np.dot(
            sympy_matrix_to_numpy(
                self.jacobian.subs(self.coords_to_subs(self.domain_coordinates, tv.x))
            ),
            tv.u
        )
        return TangentVector(u=u_image, x=self.eval(tv.x))

    def tangent_inverse_map(self, domain_pos: np.ndarray, image_vec: np.ndarray):
        return np.dot(
            sympy_matrix_to_numpy(
                self.inverse_jacobian.subs(self.coords_to_subs(self.domain_coordinates, domain_pos))
            ),
            image_vec
        )

    @staticmethod
    def coords_to_subs(coords: List[SympySymbol], values: np.ndarray):
        if len(coords) != len(values):
            raise ValueError(f"Coordinates {coords} and coordinate values {values} must have the same length")
        return {c: v for c, v in zip(coords, values)}
