import cProfile
import itertools
import pstats
import time
from pstats import SortKey
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from geodesics.coordinate_map_library import SPHERICAL_MAPPING, INVERSE_SPHERICAL_MAPPING
from geodesics.geodesic_generator import TerminationCondition
from geodesics.metric_library import morris_thorne_wormhole_generator
from geodesics.numba_geodesic_generator import NumbaGeodesicGenerator
from geodesics.sphere_projection import perspective_to_sphere, sphere_to_equirect_pixel
from geodesics.tangent_vector import TangentVector

TAU_RANGE = np.linspace(0, 80, 100)


# todo make faster: null geodesic takes ~ 1/3 times runtime of calc_geodesic
def make_null_geo_args(timelike_tv: TangentVector) -> np.ndarray:
    tv = timelike_tv
    return metric.calc_null_tangent_fast(
        np.array([tv.u[0], 0, 0, 0], dtype=float),
        np.array([0, tv.u[1], tv.u[2], tv.u[3]], dtype=float), tv.x, check=False)


def get_ray_final(x0: np.ndarray, th: float, ph: float) -> Tuple[np.ndarray, np.ndarray]:
    direction = SPHERICAL_MAPPING.tangent_inverse_map(
        domain_pos=x0[1:],
        image_vec=np.array([np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)], dtype=float)
    )
    u0 = np.array([1.0, 0.1 * direction[0], 0.1 * direction[1], 0.1 * direction[2]], dtype=float)
    tv0 = TangentVector(x=x0, u=u0)
    tv0.u = make_null_geo_args(tv0)
    geo = gg.calc_geodesic(tv0, TAU_RANGE)
    tv = geo.tv[-1]
    # tv = tv0
    return tv.x, tv.u


def pos_to_pixel(to_pixel_pos, to_pixel_neg, pos, u):
    cartesian_u = SPHERICAL_MAPPING.tangent_map(TangentVector(x=pos[1:], u=u[1:])).u
    th, ph = INVERSE_SPHERICAL_MAPPING.eval(cartesian_u)[1:]
    th = th - np.pi / 2
    if pos[1] > 0:
        return to_pixel_pos(ph, th)
    else:
        return to_pixel_neg(ph, -th)


print('generating metric')
metric = morris_thorne_wormhole_generator(4, b0_val=3.0)
# metric = sc_metric_generator(4, 1.0)
# metric = flat_polar(4)
print('calculating connections')
gg = NumbaGeodesicGenerator(metric, termination_condition=TerminationCondition.none())
metric.eval_g(np.array([1.0, 2.0, 3.0, 4.0]))

# cube_im = plt.imread('test-equirectangular.png')[:, :, :3]  # rm alpha
cube_im = plt.imread('/Users/kook/Pictures/rheingauer.jpeg')[:, :, :3]
church_im = plt.imread('/Users/kook/Pictures/aquaduct.png')[:, :, :3]
cube_im = cube_im / np.max(cube_im)
church_im = church_im / np.max(church_im)

x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))
ph_grid, lat_grid = np.vectorize(perspective_to_sphere, signature='(),()->(),()')(x_grid, y_grid)
ph_grid = ph_grid + np.pi
th_grid = np.pi / 2 + lat_grid


def calc_xu_array():
    x0 = np.array([0, 8.0, np.pi / 2, 0], dtype=float)
    return np.vectorize(
        lambda th, ph: get_ray_final(x0, th, ph),
        signature='(),()->(d),(d)'
    )(th_grid, ph_grid)


# t0 = time.time()
# pos_array, u_array = calc_xu_array()
# print(time.time() - t0)
cProfile.run("pos_array, u_array = calc_xu_array()", sort=SortKey.CUMULATIVE, filename='raytace-stats')

# plt.pcolor(th_grid, ph_grid, pos_array[:, :, 1])
# plt.colorbar()
# plt.show()
to_pix1 = sphere_to_equirect_pixel(cube_im)
to_pix2 = sphere_to_equirect_pixel(church_im)
t0 = time.time()
im = np.vectorize(
    lambda x, u: pos_to_pixel(to_pix1, to_pix2, x, u),
    signature='(d),(d)->(c)')(pos_array, u_array)
print(time.time() - t0)

plt.imshow(im)
plt.show()
