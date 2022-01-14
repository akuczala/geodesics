import cProfile
from pstats import SortKey

import numpy as np
from matplotlib import pyplot as plt

# im = plt.imread('examples/test-equirectangular.png')
from geodesics.sphere_projection import equirect_to_perspective, sphere_to_equirect_pixel, perspective_to_sphere

im = plt.imread('/Users/kook/Pictures/aquaduct.png')
x_grid, y_grid = np.meshgrid(np.linspace(-1.5, 1.5, 800), np.linspace(-1, 1, 400))
print(im.shape)
# cProfile.run("testim = np.vectorize(get_perspective_pixel_fn(im), signature='(),()->(k)')(igrid, jgrid)",
#              sort=SortKey.CUMULATIVE)
sphere_to_pix = sphere_to_equirect_pixel(im)
testim = np.vectorize(
    lambda x,y: sphere_to_pix(*perspective_to_sphere(x,y)),
    signature='(),()->(k)'
)(x_grid, y_grid)

# test = np.vectorize(lambda x, y: perspective_to_equirect(x, y, 1))(igrid, jgrid)

def plot_latlon(ax=None):
    if ax is None:
        ax = plt.gca()
    ph_range = np.linspace(-np.pi / 4, np.pi / 4, 20)
    for lat in np.linspace(-np.pi / 4, np.pi / 4, 5):
        ax.plot(*np.array([equirect_to_perspective(ph, lat) for ph in ph_range]).T)
    lat_range = np.linspace(-np.pi / 4, np.pi / 4, 20)
    for ph in np.linspace(-np.pi / 4, np.pi / 4, 5):
        ax.plot(*np.array([equirect_to_perspective(ph, lat) for lat in lat_range]).T)


# plt.colorbar()
# plt.contourf(igrid, jgrid, test[0])
# plot_latlon()
# plt.colorbar()
# plt.show()
plt.imshow(testim)
plt.show()
