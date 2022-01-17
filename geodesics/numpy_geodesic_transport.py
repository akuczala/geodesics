import numpy as np

from geodesics.numpy_geodesic import NumpyGeodesic


class NumpyGeodesicTransport(NumpyGeodesic):
    def __init__(self, x: np.ndarray, u: np.ndarray, tau_range: np.ndarray, v: np.ndarray):
        super().__init__(x, u, tau_range)
        self.v = v