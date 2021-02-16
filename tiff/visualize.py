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
    z = numpy.ma.masked_invalid(data[0])
    x, y = numpy.meshgrid(numpy.arange(z.shape[1]), numpy.arange(z.shape[0]))
    rgb = ls.shade(z, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    ax.set_zlim(numpy.nanmin(z), numpy.nanmax(z))
    ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False, shade=False, facecolors=rgb,
                    vmin=numpy.nanmin(z), vmax=numpy.nanmax(z))
    ax2 = fig.gca(projection='3d')
    z2 = numpy.ma.masked_invalid(data[1])
    x2, y2 = numpy.meshgrid(numpy.arange(z2.shape[1]), numpy.arange(z2.shape[0]))
    rgb2 = ls.shade(z2, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    ax.set_zlim(numpy.nanmin(z2), numpy.nanmax(z2))
    ax2.plot_surface(x2, y2, z2, cmap=cmap, linewidth=0, antialiased=False, shade=False, facecolors=rgb2,
                     vmin=numpy.nanmin(2), vmax=numpy.nanmax(2))
    plt.show()


def support(data):
    """
    Plot the support densities for a parcel.
    :param data: support densities for a parcel, as a 3D array
    """
    density = numpy.ma.masked_where(data <= 1e-10, data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = numpy.mgrid[:density.shape[0], :density.shape[1], :density.shape[2]]
    img = ax.scatter(X, Y, Z, c=density.ravel(), cmap=plt.viridis())
    fig.colorbar(img)
    plt.show()