import numpy as np
import matplotlib.pyplot as plt

def c_s(gamma, P, rho):
    return np.sqrt(gamma * P / rho)


def enthalpy(rho, p, E):
    return (E + p) / rho


def E(gamma, rho, p, u, v):
    return (p / (gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))


def P(gamma, rho, u, v, E):
    p = (gamma - 1) * (E - (0.5 * rho * (u ** 2 + v ** 2)))
    # p[p <= 0] = 1e-6
    return p

def get_prims(gamma, U):
    """
        Returns:
            rho, u, v, p
    """
    U = np.copy(U)
    rho = U[:, :, 0]
    u, v = U[:, :, 1] / rho, U[:, :, 2] / rho
    E = U[:, :, 3]
    p = P(gamma, rho, u, v, E)
    return rho, u, v, p


def get_cons(gamma, U):
    """
        Returns:
            rho, rho * u, rho * v, E
    """
    return U[:, :, 0], U[:, :, 1], U[:, :, 2], U[:, :, 3]


def U_from_prim(gamma, prims):
    rho, u, v, p = prims
    e = E(gamma, rho, p, u, v)
    U = np.array([rho, rho * u, rho * v, e]).transpose((1, 2, 0))
    return U


def F_from_prim(gamma, prims, x=True):
    rho, u, v, p = prims
    e = E(gamma, rho, p, u, v)
    if x:
        F = np.array([
            rho * u,
            rho * (u ** 2) + p,
            rho * u * v,
            (e + p) * u
        ]).transpose((1, 2, 0))
    else:
        F = np.array([
            rho * v,
            rho * u * v,
            rho * (v ** 2) + p,
            (e + p) * v
        ]).transpose((1, 2, 0))

    return F


def minmod(x, y, z):
    return 0.25 * np.absolute(np.sign(x) + np.sign(y)) * (np.sign(x) + np.sign(z)) * np.minimum(np.minimum(np.absolute(x), np.absolute(y)), np.absolute(z))


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    return (r * np.cos(theta), r * np.sin(theta))

def plot_sheer(gamma, U, t=0, extent=[0, 1], label=""):
    res_x, res_y = U.shape[0], U.shape[1]
    rho, u, v, p = get_prims(gamma, U)
    v_mid = v[:, res_y // 2]
    x = np.linspace(extent[0], extent[1], num=res_x, endpoint=False)
    plt.plot(x, v_mid, label=label)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$v$")
    plt.legend()
    plt.title(f"Velocity Shear Test at t = {t:.2f}")

def plot_grid(gamma, U, t=0, plot="density", extent=[0, 1, 0, 1], vmin=None, vmax=None):
    rho, u, v, p = get_prims(gamma, U)
    E = U[:, :, 3]
    labels = {"density": r"$\rho$", "u": r"$u$",
              "v": r"$v$", "pressure": r"$P$", "energy": r"$E$", }

    plt.cla()
    if plot == "density":
        if vmin is None:
            vmin, vmax = np.min(rho), np.max(rho)
        c = plt.imshow(np.transpose(rho), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    elif plot == "u":
        if vmin is None:
            vmin, vmax = np.min(u), np.max(u)
        c = plt.imshow(np.transpose(u), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    elif plot == "v":
        if vmin is None:
            vmin, vmax = np.min(v), np.max(v)
        c = plt.imshow(np.transpose(v), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    elif plot == "pressure":
        if vmin is None:
            vmin, vmax = np.min(p), np.max(p)
        c = plt.imshow(np.transpose(p), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent, vmin=0, vmax=vmax)
    elif plot == "energy":
        if vmin is None:
            vmin, vmax = np.min(E), np.max(E)
        c = plt.imshow(np.transpose(E), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent, vmin=vmin, vmax=vmax)

    plt.colorbar(c, label=labels[plot])
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title(f"t = {t:.2f}")
