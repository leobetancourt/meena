import numpy as np
import matplotlib.pyplot as plt


def c_s(gamma, P, rho):
    return np.sqrt(gamma * P / rho)


def c_A(rho, Bx, By, Bz):
    """
        Returns:
            Alfven speed
    """
    return np.sqrt((Bx ** 2 + By ** 2 + Bz ** 2) / rho)


def c_fm(gamma, P, rho, Bx, By, Bz):
    """
        Returns:
            Fast magnetosonic speed
    """
    c_s2 = c_s(gamma, P, rho) ** 2
    c_A2 = c_A(rho, Bx, By, Bz) ** 2
    c_Ax2 = (Bx ** 2) / rho

    return np.sqrt(0.5 * (c_s2 + c_A2) + 0.5 * np.sqrt((c_s2 + c_A2) ** 2 - 4 * c_s2 * c_Ax2))


def c_sm(gamma, P, rho, Bx, By, Bz):
    """
        Returns:
            Slow magnetosonic speed
    """
    c_s2 = c_s(gamma, P, rho) ** 2
    c_A2 = c_A(rho, Bx, By, Bz) ** 2
    c_Ax2 = (Bx ** 2) / rho

    return np.sqrt(0.5 * (c_s2 + c_A2) - 0.5 * np.sqrt((c_s2 + c_A2) ** 2 - 4 * c_s2 * c_Ax2))


def E(gamma, rho, u, v, w, p, Bx, By, Bz):
    """
        Returns:
            Total energy
    """
    u2 = u ** 2 + v ** 2 + w ** 2
    B2 = Bx ** 2 + By ** 2 + Bz ** 2
    return p / (gamma - 1) + 0.5 * rho * u2 + 0.5 * B2


def P(gamma, rho, u, v, w, E, Bx, By, Bz):
    """
        Returns:
            Thermal? pressure
    """
    u2 = u ** 2 + v ** 2 + w ** 2
    B2 = Bx ** 2 + By ** 2 + Bz ** 2
    return (gamma - 1) * (E - (0.5 * rho * u2) - 0.5 * B2)

def get_prims(gamma, U):
    """
        Returns:
            rho, u, v, w, p, Bx, By, Bz
    """
    U = np.copy(U)
    rho = U[..., 0]
    u, v, w = U[..., 1] / rho, U[..., 2] / rho, U[..., 3] / rho
    Bx, By, Bz = U[..., 4], U[..., 5], U[..., 6]
    En = U[..., 7]
    p = P(gamma, rho, u, v, w, En, Bx, By, Bz)
    return rho, u, v, w, p, Bx, By, Bz


def get_cons(gamma, U: np.ndarray):
    """
        Returns:
            rho, rho * u, rho * v, rho * w, E, Bx, By, Bz
    """
    return tuple(U[:, :, i] for i in range(U.shape[-1]))


def U_from_prim(gamma, prims):
    rho, u, v, w, p, Bx, By, Bz = prims
    En = E(gamma, rho, u, v, w, p, Bx, By, Bz)
    U = np.array([
        rho,
        rho * u,
        rho * v,
        rho * w,
        Bx,
        By,
        Bz,
        En
    ]).transpose((1, 2, 3, 0))
    return U


def F_from_prim(gamma, prims, dir="x"):
    rho, u, v, w, p, Bx, By, Bz = prims
    B2 = Bx ** 2 + By ** 2 + Bz ** 2
    En = E(gamma, rho, u, v, w, p, Bx, By, Bz)

    if dir == "x":
        F = np.array([
            rho * u,
            rho * (u ** 2) + p + 0.5 * B2 - Bx**2,
            rho * u * v - Bx * By,
            rho * u * w - Bx * Bz,
            np.zeros_like(rho),
            By * u - v * Bx,
            Bz * u - w * Bx,
            u * (En + p + 0.5 * B2) - Bx * (u * Bx + v * By + w * Bz)
        ]).transpose((1, 2, 3, 0))
    else:
        pass  # implement flux in y and z

    return F


def plot_1d(gamma, x, U, t=0, plot="density"):
    rho, u, v, w, p, Bx, By, Bz = get_prims(gamma, U)
    labels = {"density": r"$\rho$", "u": r"$u$",
              "v": r"$v$", "w": r"$w$", "pressure": r"$P$", "energy": r"$E$", "Bx": r"$B_x$", "By": r"$B_y$", "Bz": r"$B_z$"}
    plt.cla()
    if plot == "density":
        plt.plot(x, rho[:, 0, 0], label=labels[plot])
    elif plot == "u":
        plt.plot(x, u[:, 0, 0], label=labels[plot])
    elif plot == "v":
        plt.plot(x, v[:, 0, 0], label=labels[plot])
    elif plot == "w":
        plt.plot(x, w[:, 0, 0], label=labels[plot])
    elif plot == "pressure":
        plt.plot(x, p[:, 0, 0], label=labels[plot])
    elif plot == "By":
        plt.plot(x, By[:, 0, 0], label=labels[plot])
    elif plot == "Bz":
        plt.plot(x, Bz[:, 0, 0], label=labels[plot])

    plt.xlabel("x")
    plt.legend()
    plt.title(f"t = {t:.2f}")
