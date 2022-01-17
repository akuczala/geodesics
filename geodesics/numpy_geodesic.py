import numpy as np
from numpy import ndarray

from geodesics.geodesic import Geodesic


class NumpyGeodesic(Geodesic):
    def __init__(self, x: np.ndarray, u: np.ndarray, tau_range: np.ndarray):
        self._x = x
        self._u = u
        self._tau_range = tau_range

    @property
    def x(self) -> ndarray:
        return self._x

    @property
    def u(self) -> ndarray:
        return self._u

    @property
    def tau(self) -> ndarray:
        return self._tau_range
