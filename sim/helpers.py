import numpy as np
import matplotlib.pyplot as plt


def E(gamma, rho, p, u, v):
    return (p / (gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))


def P(gamma, rho, u, v, E):
    return (gamma - 1) * (E - (0.5 * rho * (u ** 2 + v ** 2)))

# returns rho, u, v, p


def get_prims(gamma, U):
    rho = U[:, :, 0]
    u, v = U[:, :, 1] / rho, U[:, :, 2] / rho
    E = U[:, :, 3]
    p = P(gamma, rho, u, v, E)
    return rho, u, v, p

# returns rho, rho * u, rho * v, E


def get_cons(gamma, U):
    return U[:, :, 0], U[:, :, 1], U[:, :, 2], U[:, :, 3]


def U_from_prim(gamma, prims):
    rho, u, v, p = prims[0], prims[1], prims[2], prims[3]
    e = E(gamma, rho, p, u, v)
    U = np.array([rho, rho * u, rho * v, e]).transpose((1, 2, 0))
    return U


def F_from_prim(gamma, prims, x=True):
    rho, u, v, p = prims[0], prims[1], prims[2], prims[3]
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


def enthalpy(rho, p, E):
    return (E + p) / rho


def minmod(x, y, z):
    return 0.25 * np.absolute(np.sign(x) + np.sign(y)) * (np.sign(x) + np.sign(z)) * np.minimum(np.absolute(x), np.minimum(np.absolute(y), np.absolute(z)))


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return (r, theta)


def polar_to_cartesian(r, theta):
    return (r * np.cos(theta), r * np.sin(theta))


def c_s(gamma, P, rho):
    return np.sqrt(gamma * P / rho)


def plot_grid(gamma, U, t=0, plot="density", extent=[0, 1, 0, 1]):
    rho = U[:, :, 0]
    u, v, E = U[:, :, 1] / rho, U[:, :, 2] / rho, U[:, :, 3]
    p = P(gamma, rho, u, v, E)
    labels = {"density": r"$\rho$", "u": r"$u$",
              "v": r"$v$", "pressure": r"$P$", "E": r"$E$", }

    plt.cla()
    if plot == "density":
        # plot density matrix (excluding ghost cells)
        c = plt.imshow(np.transpose(rho), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent)
    elif plot == "u":
        c = plt.imshow(np.transpose(u), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent)
    elif plot == "v":
        c = plt.imshow(np.transpose(u), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent)
    elif plot == "pressure":
        c = plt.imshow(np.transpose(p), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent)
    elif plot == "energy":
        c = plt.imshow(np.transpose(E), cmap="plasma", interpolation='nearest',
                       origin='lower', extent=extent)

    plt.colorbar(c, label=labels[plot])
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title(f"t = {t:.2f}")
