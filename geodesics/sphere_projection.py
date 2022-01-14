from typing import Tuple, Callable

import numpy as np
from numba import njit


@njit
def perspective_to_sphere(x: float, y: float) -> Tuple[float, float]:
    """
    returns ph, th in (-np.pi/2, np.pi/2)
    """
    ph = np.arctan(x)
    return np.arctan(x), np.arctan(np.cos(ph) * y)


@njit
def equirect_to_perspective(ph: float, lat: float) -> Tuple[float, float]:
    return np.tan(ph), np.tan(lat) / np.cos(ph)


@njit
def intfloorclip(n: int, x: float) -> int:
    if x < 0.0:
        return 0
    if x > n - 1:
        return n - 1
    return int(x)


@njit
def to_unit_interval(x: float, low: float, high: float) -> float:
    return (x - low) / (high - low)


def sphere_to_equirect_pixel(
            im: np.ndarray,
            ph_range: Tuple[float, float] = (-np.pi, np.pi),
            th_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
    ) -> Callable[[float, float], np.ndarray]:
        ni = im.shape[0]
        nj = im.shape[1]

        @njit
        def f(ph: float, th: float) -> np.ndarray:
            j = intfloorclip(nj, to_unit_interval(ph, *ph_range) * nj)
            i = intfloorclip(ni, to_unit_interval(th, *th_range) * ni)
            return im[i, j]

        return f
# def get_perspective_pixel_fn(
#         im: np.ndarray,
#         ph_range: Tuple[float, float] = (-np.pi, np.pi),
#         th_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
# ) -> Callable[[float, float], np.ndarray]:
#     ni = im.shape[0]
#     nj = im.shape[1]
#
#     @njit
#     def f(x: float, y: float) -> np.ndarray:
#         ph, th = perspective_to_sphere(x, y)
#         j = intfloorclip(nj, to_unit_interval(ph, *ph_range) * nj)
#         i = intfloorclip(ni, to_unit_interval(th, *th_range) * ni)
#         return im[i, j]
#
#     return f
