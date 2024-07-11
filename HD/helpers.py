import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Circle
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300

def enthalpy(rho, p, E):
    return (E + p) / rho

def get_cons(gamma, U):
    """
        Returns:
            rho, rho * u, rho * v, E
    """
    return U[:, :, 0], U[:, :, 1], U[:, :, 2], U[:, :, 3]

def add_ghost_cells(arr, num):
    # add ghost cells to the second coordinate direction (y if cartesian, theta if polar)
    arr = np.hstack((np.repeat(arr[:, :1, :], num, axis=1), arr, np.repeat(
        arr[:, :1, :], num, axis=1)))

    # add ghost cells to the first coordinate direction (x if cartesian, r if polar)
    arr = np.vstack((np.repeat(arr[:1, :, :], num, axis=0), arr, np.repeat(
        arr[:1, :, :], num, axis=0)))

    return arr


def minmod(x, y, z):
    return 0.25 * np.absolute(np.sign(x) + np.sign(y)) * (np.sign(x) + np.sign(z)) * np.minimum(np.minimum(np.absolute(x), np.absolute(y)), np.absolute(z))


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

# def plot_sheer(gamma, U, t=0, extent=[0, 1], label=""):
#     res_x, res_y = U.shape[0], U.shape[1]
#     rho, u, v, p = get_prims(gamma, U)
#     v_mid = v[:, res_y // 2]
#     x = np.linspace(extent[0], extent[1], num=res_x, endpoint=False)
#     plt.plot(x, v_mid, label=label)
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$v$")
#     plt.legend()
#     plt.title(f"Velocity Shear Test at t = {t:.2f}")

def plot_grid(matrix, label, coords="cartesian", x1=None, x2=None, vmin=None, vmax=None):
    extent = [x1[0], x1[-1], x2[0], x2[-1]]

    if coords == "cartesian":
        fig, ax = plt.subplots()
        if vmin is None:
            vmin, vmax = np.min(matrix), np.max(matrix)
        c = ax.imshow(np.transpose(matrix), cmap="magma", interpolation='nearest',
                    origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    elif coords == "polar":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        if vmin is None:
            vmin, vmax = np.min(matrix), np.max(matrix)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, np.max(x1))
        ax.set_facecolor("black")
        circle = Circle((0, 0), radius=np.min(x1), transform=ax.transData._b, color='blue', fill=False, linewidth=1)
        ax.add_patch(circle)
        R, Theta = np.meshgrid(x1, x2, indexing="ij")
        c = ax.pcolormesh(Theta, R, matrix, shading='auto', cmap="magma", vmin=vmin, vmax=vmax)
    
    cb = plt.colorbar(c, ax=ax, label=label)
        
    return fig, ax, c, cb

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                        (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # print new line on complete
    if iteration == total:
        print()