from abc import abstractmethod
from typing import List

from numpy import ndarray

from geodesics.tangent_vector import TangentVector


class Geodesic:

    @property
    @abstractmethod
    def x(self) -> ndarray:
        pass

    @property
    @abstractmethod
    def u(self) -> ndarray:
        pass

    @property
    def tv(self) -> List[TangentVector]:
        return [TangentVector(x=x, u=u) for x, u in zip(self.x, self.u)]

    @property
    @abstractmethod
    def tau(self) -> ndarray:
        pass
