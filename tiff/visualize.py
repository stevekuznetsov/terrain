from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy


def parcel(data):
    """
    Plot the top and bottom surfaces for a parcel.
    :param data: top and bottom surface data for a parcel, as 2D arrays
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ls = LightSource(270, 45)
    cmap = plt.cm.gist_earth
    z = data[0]
    x, y = numpy.meshgrid(numpy.arange(z.shape[1]), numpy.arange(z.shape[0]))
    rgb = ls.shade(z, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False, shade=False, facecolors=rgb)
    ax2 = fig.gca(projection='3d')
    z2 = data[1]
    x2, y2 = numpy.meshgrid(numpy.arange(z2.shape[1]), numpy.arange(z2.shape[0]))
    rgb2 = ls.shade(z2, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    ax2.plot_surface(x2, y2, z2, cmap=cmap, linewidth=0, antialiased=False, shade=False, facecolors=rgb2)
    plt.show()
