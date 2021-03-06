from typing import List

import numpy as np
from numpy import ndarray

from geodesics.tangent_vector import TangentVector


def y_to_x(y):
    return y[:len(y) // 2]


def y_to_u(y):
    return y[len(y) // 2:]


def x_u_to_y(x, u):
    return np.concatenate((x, u))


def tv_to_y(tv: TangentVector):
    return x_u_to_y(tv.x, tv.u)


class Geodesic:
    def __init__(self, sol):
        self.sol = sol

    @property
    def dim(self) -> int:
        return self.sol.y.shape[0]

    @property
    def x(self) -> ndarray:
        return y_to_x(self.sol.y).T

    @property
    def u(self) -> ndarray:
        return y_to_u(self.sol.y).T

    @property
    def tv(self) -> List[TangentVector]:
        return [TangentVector(x=x, u=u) for x, u in zip(self.x, self.u)]

    @property
    def tau(self) -> ndarray:
        return self.sol.t
