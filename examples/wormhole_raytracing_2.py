import cProfile
import itertools
import time
from pstats import SortKey
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from geodesics.coordinate_map_library import POLAR_MAPPING, SPHERICAL_MAPPING
from geodesics.geodesic_generator import TerminationCondition, GeodesicGenerator
from geodesics.metric_library import morris_thorne_wormhole_generator
from geodesics.numba_geodesic_generator import NumbaGeodesicGenerator
from geodesics.tangent_vector import TangentVector, TangentVectorType

print('generating metric')
metric = morris_thorne_wormhole_generator(4, b0_val=3.0)
print('calculating connections')
gg = NumbaGeodesicGenerator(metric, termination_condition=TerminationCondition.none())


def make_null_geo_args(timelike_tv: TangentVector) -> Tuple[TangentVector, np.ndarray]:
    assert metric.classify_tangent_vector(timelike_tv) == TangentVectorType.TIMELIKE
    tv = timelike_tv
    u0 = metric.calc_null_tangent_fast(
        np.array([tv.u[0], 0, 0, 0]),
        np.array([0, tv.u[1], tv.u[2], tv.u[3]]), tv.x, check=False)
    null_tv = TangentVector(x=tv.x, u=u0)
    return null_tv, np.linspace(0, 80, 2)


TAU_RANGE = np.linspace(0, 100, 2)


def get_ray_final(r0: float, th: float, ph: float) -> np.ndarray:
    x0 = np.array([0, r0, np.pi / 2, 0], dtype=float)
    # todo: roughly 1/4 of time is spent calculating direction
    direction = SPHERICAL_MAPPING.tangent_inverse_map(
        domain_pos=x0[1:],
        image_vec=np.array([np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)])
    )
    u0 = np.array([1.0, 0.1 * direction[0], 0.1 * direction[1], 0.1 * direction[2]])
    # print('calculating geodesic')
    # todo: almost all of the remaining time is spent making initial null vector!
    # boo = make_null_geo_args(TangentVector(x=x0, u=u0))
    geo = gg.calc_geodesic(*make_null_geo_args(TangentVector(x=x0, u=u0)))
    # geo = gg.calc_geodesic(TangentVector(x=x0, u=u0), TAU_RANGE)
    x_final = geo.x[-1]
    #null_args = make_null_geo_args(TangentVector(x=x0, u=u0))
    #x_final = x0
    return x_final


def intfloorclip(n: int, x: float) -> int:
    return int(np.floor(np.clip(x, 0, n - 1)))


def to_pixel(im: np.ndarray, ph0: float) -> np.ndarray:
    ni = im.shape[0]
    nj = im.shape[1]

    def f(th: float, ph: float):
        i = intfloorclip(ni, ((th) % np.pi) / np.pi * ni)
        j = intfloorclip(nj, ((ph + ph0) % (2 * np.pi)) / (2 * np.pi) * nj)
        return im[i, j]

    return f


def pos_to_pixel(to_pixel_pos, to_pixel_neg, pos):
    if pos[1] > 0:
        return to_pixel_pos(pos[2], pos[3])
    else:
        return to_pixel_neg(pos[2], pos[3])


cube_im = plt.imread('test-equirectangular.png')[:, :, :-1]  # rm alpha
assert cube_im.shape[-1] == 3
church_im = plt.imread('test-2.png')[:, :, :-1]
assert church_im.shape[-1] == 3
cube_im = cube_im / np.max(cube_im)
church_im = church_im / np.max(church_im)

th_range = np.linspace(np.pi / 4, 3 * np.pi / 4, 80)
ph_range = np.linspace(np.pi - np.pi / 4, np.pi + np.pi / 4, 80)

th_grid, ph_grid = np.meshgrid(th_range, ph_range, indexing='ij')


def calc_pos_array():
    return np.vectorize(
        lambda th, ph: get_ray_final(4.0, th, ph),
        signature='(),()->(d)'
    )(th_grid, ph_grid)


#cProfile.run("calc_pos_array()",sort=SortKey.CUMULATIVE)

t0 = time.time()
pos_array = calc_pos_array()
print(time.time() - t0)

# plt.pcolor(th_grid, ph_grid, pos_array[:, :, 1])
# plt.colorbar()
# plt.show()
to_pix1 = to_pixel(cube_im, 0)
to_pix2 = to_pixel(church_im, np.pi)
im = np.vectorize(
    lambda p: pos_to_pixel(to_pix1, to_pix2, p),
    signature='(d)->(c)')(pos_array)
plt.imshow(im)
plt.show()
