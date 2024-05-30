import numpy as np
import matplotlib.pyplot as plt


def E(gamma, rho, p, u, v):
    return (p / (gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))


def P(gamma, rho, u, v, E):
    return (gamma - 1) * (E - (0.5 * rho * (u ** 2 + v ** 2)))


def enthalpy(rho, p, E):
    return (E + p) / rho


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return (r, theta)


def polar_to_cartesian(r, theta):
    return (r * np.cos(theta), r * np.sin(theta))


def c_s(gamma, P, rho):
    return np.sqrt(gamma * P / rho)

def plot_grid(gamma, U, plot="density", extent=[0, 1, 0, 1]):
    rho = U[:, :, 0]
    u, v, E = U[:, :, 1] / rho, U[:, :, 2] / rho, U[:, :, 3]
    p = P(gamma, rho, u, v, E)

    plt.cla()
    if plot == "density":
        # plot density matrix (excluding ghost cells)
        c = plt.imshow(np.transpose(rho[1:-1, 1:-1]), cmap="plasma", interpolation='nearest',
                        origin='lower', extent=extent)
    elif plot == "pressure":
        c = plt.imshow(np.transpose(p[1:-1, 1:-1]), cmap="plasma", interpolation='nearest',
                        origin='lower', extent=extent)
    elif plot == "energy":
        c = plt.imshow(np.transpose(E[1:-1, 1:-1]), cmap="plasma", interpolation='nearest',
                        origin='lower', extent=extent)

    plt.colorbar(c)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(plot)