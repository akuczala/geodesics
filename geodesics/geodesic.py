from typing import List

from numpy import ndarray

from geodesics.tangent_vector import TangentVector


class Geodesic:
    def __init__(self, sol):
        self.sol = sol

    @property
    def dim(self) -> int:
        return self.sol.y.shape[0]

    @property
    def x(self) -> ndarray:
        return self.sol.y[:self.dim // 2].T

    @property
    def u(self) -> ndarray:
        return self.sol.y[self.dim // 2:].T

    @property
    def tv(self) -> List[TangentVector]:
        return [TangentVector(x=x, u=u) for x, u in zip(self.x, self.u)]

    @property
    def tau(self) -> ndarray:
        return self.sol.t
