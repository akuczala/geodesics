import os
import time
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from geodesics.coordinate_map_library import SPHERICAL_MAPPING, INVERSE_SPHERICAL_MAPPING, get_bispherical_mapping
from geodesics.metric_library import morris_thorne_wormhole_generator, double_wormhole_generator
from geodesics.numba_geodesic_generator import NumbaGeodesicGenerator
from geodesics.numpy_geodesic_transport import NumpyGeodesicTransport
from geodesics.scipy_geodesic_generator import ScipyGeodesicGenerator
from geodesics.sphere_projection import perspective_to_sphere, sphere_to_equirect_pixel
from geodesics.tangent_vector import TangentVector

TAU_RANGE = np.linspace(0, 80, 100)

START_ANI = int(os.environ['START_ANI'])
END_ANI = int(os.environ['END_ANI'])
N_FRAMES = int(os.environ['N_FRAMES'])
RESOLUTION = int(os.environ['RESOLUTION'])
#OUT_PATH = os.environ['OUT_PATH']

# START_ANI = 0
# END_ANI = 6
# N_FRAMES = 6
# RESOLUTION = 80

# todo make faster: null geodesic takes ~ 1/3 times runtime of calc_geodesic
def make_null_geo_args(timelike_tv: TangentVector) -> np.ndarray:
    tv = timelike_tv
    return metric.calc_null_tangent_fast(
        np.array([tv.u[0], 0, 0, 0], dtype=float),
        np.array([0, tv.u[1], tv.u[2], tv.u[3]], dtype=float), tv.x, check=False)


def get_ray_final(x0: np.ndarray, frame: np.ndarray, th: float, ph: float) -> Tuple[np.ndarray, np.ndarray]:
    # rho = np.sqrt(b0_val ** 2 + x0[1] ** 2)
    # direction = np.array([
    #     np.cos(ph) * np.sin(th),
    #     -np.cos(th) / rho,
    #     np.sin(ph) * np.sin(th) / (rho * np.abs(np.sin(x0[2])))
    # ])
    direction = np.cos(ph) * np.sin(th) * frame[1] + np.cos(th) * frame[2] + np.sin(ph) * np.sin(th) * frame[3]
    u0 = np.array([1.0, 0.1 * direction[1], 0.1 * direction[2], 0.1 * direction[3]], dtype=float)
    tv0 = TangentVector(x=x0, u=u0)
    tv0.u = make_null_geo_args(tv0)
    geo = gg.calc_geodesic(tv0, TAU_RANGE)
    tv = geo.tv[-1]
    # tv = tv0
    return tv.x, tv.u


def pos_to_pixel(to_pixel_pos, to_pixel_neg, pos, u):
    # todo: this needs to be bispherical coordinates
    cartesian_u = bispherical_mapping.tangent_map(TangentVector(x=pos[1:], u=u[1:])).u
    th, ph = INVERSE_SPHERICAL_MAPPING.eval(cartesian_u)[1:]
    th = th - np.pi / 2
    if np.sin(pos[1]) > 0:
        return to_pixel_pos(ph, th)
    else:
        return to_pixel_neg(ph, -th)


def get_frame0(x0):
    ps, s, ph = x0[1], x0[2], x0[3]
    tau = - beta_val*np.cos(ps)
    b = a_val / (np.cosh(tau) - np.cos(s))
    vt = 0
    vps = 1/np.sqrt((b * beta_val * np.sin(ps)) ** 2 + np.cos(ps)**2)
    vs = -1/np.abs(b)
    vph = 1 / np.abs(b * np.sin(s))
    return np.diag([vt, vps, vs, vph])


def get_frames(tv0: TangentVector, frame0, tau_range) -> List[NumpyGeodesicTransport]:
    sgg = ScipyGeodesicGenerator(metric)
    return [
        sgg.calc_parallel_transport(
            tv0=tv0,
            t_range=tau_range,
            v0=v0,
            raise_on_fail=True
        )
        for v0 in frame0
    ]


def calc_xu_array(x0, frame, th_grid, ph_grid):
    return np.vectorize(
        lambda th, ph: get_ray_final(x0, frame, th, ph),
        signature='(),()->(d),(d)'
    )(th_grid, ph_grid)


def render_frame(pos_to_pixel, x0, frame, th_grid, ph_grid):
    pos_array, u_array = calc_xu_array(x0, frame, th_grid, ph_grid)

    im = np.vectorize(
        pos_to_pixel,
        signature='(d),(d)->(c)')(pos_array, u_array)
    return im


print('generating metric')
a_val = 2.0
beta_val = 2.0
metric = double_wormhole_generator(4, a_val=a_val, beta_val=beta_val)
# metric = sc_metric_generator(4, 1.0)
# metric = flat_polar(4)
print('calculating connections')
gg = NumbaGeodesicGenerator(metric)

print('calculating mapping')
bispherical_mapping = get_bispherical_mapping(a=a_val, beta=beta_val, calc_inverse=False)

cube_im = plt.imread('rheingauer.jpeg')[:, :, :3]  # rm alpha
church_im = plt.imread('aquaduct.png')[:, :, :3]  # rm alpha
cube_im = cube_im / np.max(cube_im)
church_im = church_im / np.max(church_im)

x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, RESOLUTION), np.linspace(-1, 1, RESOLUTION))
ph_grid, lat_grid = np.vectorize(perspective_to_sphere, signature='(),()->(),()')(x_grid, y_grid)
ph_grid = ph_grid + np.pi
th_grid = np.pi / 2 + lat_grid

camera_tv0 = TangentVector(
    x=np.array([0, np.pi*0.4, np.pi*1.002, 0]),
    u=np.array([1.0, -0.005, 0.0, 0.0])
)
CAMERA_TAU_RANGE = np.linspace(0,1100,N_FRAMES)

frame_geos = get_frames(camera_tv0, get_frame0(camera_tv0.x), CAMERA_TAU_RANGE)
to_pix1 = sphere_to_equirect_pixel(cube_im)
to_pix2 = sphere_to_equirect_pixel(church_im)

for i in list(range(len(frame_geos[0].tau)))[START_ANI: END_ANI]:
    print(f'rendering frame {i}')
    im = render_frame(
        pos_to_pixel=lambda x, u: pos_to_pixel(to_pix1, to_pix2, x, u),
        x0=frame_geos[0].x[i],
        frame=np.array([geo.v[i] for geo in frame_geos]),
        th_grid=th_grid,
        ph_grid=ph_grid
    )
    plt.imsave(f'out/out_{i:03d}.png', im)
    #plt.imshow(im)
    #plt.show()
print('done')