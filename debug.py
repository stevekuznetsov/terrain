#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

import config.load
import config.debug
from tiff.support import generate_support_for_surface


def elemental_density():
    def rho_e_for(mu, beta):
        return 1 - numpy.exp(- beta * mu) + mu * numpy.exp(-beta)

    mu_bounds = numpy.arange(0, 1, .001)
    beta_bounds = numpy.arange(1, 100, .1)
    mu, beta = numpy.meshgrid(mu_bounds, beta_bounds)
    rho_values = numpy.array(rho_e_for(numpy.ravel(mu), numpy.ravel(beta)))
    rho_e = rho_values.reshape(mu.shape)

    return mu, beta, rho_e


def plot_elemental_density_surface():
    mu, beta, rho_e = elemental_density()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Elemental Density ρ as a Function of Neighborhood Density and Heaviside Aggressiveness")
    ax.set_ylabel("Regularized Heaviside Parameter β")
    ax.set_xlabel("Neighborhood Density μ")
    ax.set_zlabel("Elemental Density ρ")
    img = ax.plot_surface(mu, beta, rho_e, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.colorbar(img)
    plt.show()


def plot_elemental_density_lines():
    mu, _, rho_e = elemental_density()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Elemental Density ρ as a Function of Neighborhood Density and Heaviside Aggressiveness")
    ax.set_xlabel("Neighborhood Density μ")
    ax.set_ylabel("Elemental Density ρ")
    for beta in [1, 5, 10, 20, 40, 80]:
        beta_index = int(beta / .1)
        ax.plot(mu[0, :], rho_e[beta_index, :], label=beta)
    leg = plt.legend(loc="lower right", title="Regularized Heavi-\nside Parameter β")
    leg._legend_box.align = "left"
    plt.show()


def plot_support_density():
    bounds = numpy.arange(0, 1, .001)
    mu, threshold = numpy.meshgrid(bounds, bounds)
    beta_t = 20

    fig = plt.figure()

    def rho_s_for(t, m):
        return (numpy.tanh(beta_t * t) + numpy.tanh(beta_t * (m - t))) / \
               (numpy.tanh(beta_t * t) + numpy.tanh(beta_t * (1 - t)))

    rho_values = numpy.array(rho_s_for(numpy.ravel(mu), numpy.ravel(threshold)))
    rho_s = rho_values.reshape(threshold.shape)

    ax = fig.gca()
    ax.set_title("Nodal Support Density ρ for β={}".format(beta_t))
    ax.set_xlabel("Neighborhood Support μ")
    ax.set_ylabel("Nodal Support Density ρ")
    for t in [.1, .2, .4, .6, .8]:
        t_index = int(t / .001)
        ax.plot(mu[0, :], rho_s[:, t_index], label=t)
    leg = plt.legend(loc="lower right", title="Heavisideside\n Parameter T")
    leg._legend_box.align = "left"
    plt.show()


def plot_support_densities():
    bounds = numpy.arange(0, 1, .001)
    mu, threshold = numpy.meshgrid(bounds, bounds)

    fig = plt.figure()
    fig.suptitle(
        "Nodal Support Density ρ as a Function of Neighborhood Support"
        " and Threshold For Various Heaviside Aggressiveness β"
    )
    ix = 1
    axes = []
    for beta_t in [1, 5, 10, 20, 40, 80]:
        def rho_s_for(t, m):
            return (numpy.tanh(beta_t * t) + numpy.tanh(beta_t * (m - t))) / \
                   (numpy.tanh(beta_t * t) + numpy.tanh(beta_t * (1 - t)))

        rho_values = numpy.array(rho_s_for(numpy.ravel(mu), numpy.ravel(threshold)))
        rho_s = rho_values.reshape(threshold.shape)

        ax = fig.add_subplot(2, 3, ix)
        ix += 1
        ax.set_title("Nodal Support Density ρ for β={}".format(beta_t))
        ax.set_xlabel("Neighborhood Support μ")
        ax.set_ylabel("Threshold T")
        img = ax.imshow(rho_s, extent=[0, 1, 1, 0])
        axes.append(ax)
    plt.colorbar(img, ax=axes)
    plt.show()


def generate_supports(configuration, debug_configuration, logger):
    def fun(x, y):
        return abs((x - 1) * (configuration["model"]["support"]["minimum_feature_radius_millimeters"] / 1e3) / 4)

    x = y = numpy.arange(0, 5, .2)
    X, Y = numpy.meshgrid(x, y)
    zs = numpy.array(fun(numpy.ravel(X), numpy.ravel(Y)))
    Z = zs.reshape(X.shape)
    # Z /= Z.max()/((configuration["model"]["support"]["minimum_feature_radius_millimeters"] / 1e3) * 7)
    surface = Z

    generate_support_for_surface(configuration, debug_configuration["support"], surface, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug things.")
    parser.add_argument("--configuration", help="Path to the configuration.", required=True)
    parser.add_argument("--debug-configuration", help="Path to the debug configuration.")
    parser.add_argument("--cache", help="Cache base directory.", default=str(Path.home().joinpath("terrain")))
    parser.add_argument("--loglevel", help="Logging verbosity level.", default="INFO")
    args = parser.parse_args()
    logger = logging.getLogger("terrain")
    logger.setLevel(args.loglevel.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    conf, hash = config.load.load(args.configuration)
    cache_dir = Path(args.cache).joinpath(hash)
    logger.info("Initializing cache to " + str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    conf["meta"] = {"cache": cache_dir, "logger": logger}

    debug_conf = config.debug.load(args.debug_configuration)
    generate_supports(conf, debug_conf, logger)
