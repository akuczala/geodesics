import cProfile
import itertools
import time
from pstats import SortKey
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from geodesics.coordinate_map_library import SPHERICAL_MAPPING, INVERSE_SPHERICAL_MAPPING
from geodesics.geodesic_generator import TerminationCondition, GeodesicGenerator
from geodesics.metric_library import morris_thorne_wormhole_generator, flat, flat_polar, sc_metric_generator
from geodesics.numba_geodesic_generator import NumbaGeodesicGenerator
from geodesics.sphere_projection import intfloorclip, perspective_to_sphere, sphere_to_equirect_pixel
from geodesics.tangent_vector import TangentVector, TangentVectorType

print('generating metric')
metric = morris_thorne_wormhole_generator(4, b0_val=3.0)
#metric = sc_metric_generator(4, 1.0)
#metric = flat_polar(4)
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


def get_ray_final(x0: np.ndarray, th: float, ph: float) -> Tuple[np.ndarray, np.ndarray]:
    direction = SPHERICAL_MAPPING.tangent_inverse_map(
        domain_pos=x0[1:],
        image_vec=np.array([np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)])
    )
    u0 = np.array([1.0, 0.1 * direction[0], 0.1 * direction[1], 0.1 * direction[2]])
    geo = gg.calc_geodesic(*make_null_geo_args(TangentVector(x=x0, u=u0)))
    tv = geo.tv[-1]
    return tv.x, tv.u


# def to_pixel(im: np.ndarray, ph0: float) -> np.ndarray:
#     ni = im.shape[0]
#     nj = im.shape[1]
#
#     def f(th: float, ph: float):
#         i = intfloorclip(ni, ((th) % np.pi) / np.pi * ni)
#         j = intfloorclip(nj, ((ph + ph0) % (2 * np.pi)) / (2 * np.pi) * nj)
#         return im[i, j]
#
#     return f


def pos_to_pixel(to_pixel_pos, to_pixel_neg, pos, u):
    cartesian_u = SPHERICAL_MAPPING.tangent_map(TangentVector(x=pos[1:], u=u[1:])).u
    th, ph = INVERSE_SPHERICAL_MAPPING.eval(cartesian_u)[1:]
    th = th - np.pi/2
    if pos[1] > 0:
        return to_pixel_pos(ph, th)
    else:
        return to_pixel_neg(ph, -th)


#cube_im = plt.imread('test-equirectangular.png')[:, :, :3]  # rm alpha
cube_im = plt.imread('/Users/kook/Pictures/rheingauer.jpeg')[:, :, :3]
church_im = plt.imread('/Users/kook/Pictures/aquaduct.png')[:, :, :3]
cube_im = cube_im / np.max(cube_im)
church_im = church_im / np.max(church_im)

x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))
#th_range = np.linspace(np.pi / 4, 3 * np.pi / 4, 160)
#ph_range = np.linspace(np.pi - np.pi / 4, np.pi + np.pi / 4, 160)

#th_grid, ph_grid = np.meshgrid(th_range, ph_range, indexing='ij')
ph_grid, lat_grid = np.vectorize(perspective_to_sphere, signature='(),()->(),()')(x_grid, y_grid)
ph_grid = ph_grid + np.pi
th_grid = np.pi/2 + lat_grid


def calc_xu_array():
    x0 = np.array([0, 8.0, np.pi / 2, 0], dtype=float)
    return np.vectorize(
        lambda th, ph: get_ray_final(x0, th, ph),
        signature='(),()->(d),(d)'
    )(th_grid, ph_grid)


# cProfile.run("calc_pos_array()",sort=SortKey.CUMULATIVE)

t0 = time.time()
#pos_array, u_array = calc_xu_array()
cProfile.run("pos_array, u_array = calc_xu_array()",sort=SortKey.CUMULATIVE)
print(time.time() - t0)

# plt.pcolor(th_grid, ph_grid, pos_array[:, :, 1])
# plt.colorbar()
# plt.show()
to_pix1 = sphere_to_equirect_pixel(cube_im)
to_pix2 = sphere_to_equirect_pixel(church_im)
t0 = time.time()
im = np.vectorize(
    lambda x, u: pos_to_pixel(to_pix1, to_pix2, x, u),
    signature='(d),(d)->(c)')(pos_array, u_array)
print(time.time()- t0)
# igrid, jgrid = np.meshgrid(np.linspace(-1, 1, 160), np.linspace(-1, 1, 160))
# im = np.vectorize(
#     get_perspective_pixel_fn(
#         sphere_im,
#         ph_range=(-np.pi / 4, np.pi / 4),
#         th_range=(-np.pi / 4, np.pi / 4)
#     ),
#     signature='(),()->(k)'
#)(igrid, jgrid)
plt.imshow(im)
plt.show()
