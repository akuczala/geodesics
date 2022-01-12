import numpy as np
from matplotlib import pyplot as plt


def perspective_to_equirect(x, y, R):
    ph = R * np.arctan(x)
    return R * np.arctan(x), R * np.arctan(np.cos(ph) * y)

def equirect_to_perspective(ph, lat):
    return np.tan(ph), np.tan(lat)/np.cos(ph)
def get_perspective_pixel_fn(im):
    ni = im.shape[0]
    nj = im.shape[1]

    def f(i, j):
        ph, th = perspective_to_equirect(i, j, 1)
        j = int(np.floor(np.clip((ph / (np.pi) + 1) / 2 * nj, 0, nj - 1)))
        i = int(np.floor(np.clip((th / (np.pi / 2) + 1) / 2 * ni, 0, ni - 1)))
        return im[i, j]

    return f


#im = plt.imread('examples/test-equirectangular.png')
im = plt.imread('/Users/kook/Pictures/rheingauer.jpeg')
igrid, jgrid = np.meshgrid(np.linspace(-1.5, 1.5, 800), np.linspace(-1, 1, 400))
print(im.shape)
testim = np.vectorize(get_perspective_pixel_fn(im), signature='(),()->(k)')(igrid, jgrid)
test = np.vectorize(lambda x, y: perspective_to_equirect(x, y, 1))(igrid, jgrid)

def plot_latlon(ax=None):
    if ax is None:
        ax = plt.gca()
    ph_range = np.linspace(-np.pi/4,np.pi/4,20)
    for lat in np.linspace(-np.pi/4,np.pi/4,5):
        ax.plot(*(np.array([equirect_to_perspective(ph, lat) for ph in ph_range]).T))
    lat_range = np.linspace(-np.pi/4,np.pi/4,20)
    for ph in np.linspace(-np.pi/4,np.pi/4,5):
        ax.plot(*(np.array([equirect_to_perspective(ph, lat) for lat in lat_range]).T))
#plt.colorbar()
#plt.contourf(igrid, jgrid, test[0])
#plot_latlon()
#plt.colorbar()
#plt.show()
plt.imshow(testim)
plt.show()
