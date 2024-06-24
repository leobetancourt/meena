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


def c_fm(gamma, P, rho, Bx, By, Bz, dir="x"):
    """
        Returns:
            Fast magnetosonic speed
    """
    c_s2 = c_s(gamma, P, rho) ** 2
    c_A2 = c_A(rho, Bx, By, Bz) ** 2
    if dir == "x": c_Ax2 = (Bx ** 2) / rho
    elif dir == "y": c_Ax2 = (By ** 2) / rho
    elif dir == "z": c_Ax2 = (Bz ** 2) / rho

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
    elif dir == "y":
        F = np.array([
            rho * v,
            rho * v * u - By * Bx,
            rho * (v ** 2) + p + 0.5 * B2 - By**2,
            rho * v * w - By * Bz,
            Bx * v - u * By,
            np.zeros_like(rho),
            Bz * v - w * By,
            v * (En + p + 0.5 * B2) - By * (u * Bx + v * By + w * Bz)
        ]).transpose((1, 2, 3, 0))
    elif dir == "z":
        F = np.array([
            rho * w,
            rho * w * u - Bz * Bx,
            rho * w * v - Bz * By,
            rho * (w ** 2) + p + 0.5 * B2 - Bz**2,
            Bx * w - u * Bz,
            By * w - v * Bz,
            np.zeros_like(rho),
            w * (En + p + 0.5 * B2) - Bz * (u * Bx + v * By + w * Bz)
        ]).transpose((1, 2, 3, 0))

    return F

def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta

def divergence(Bx, By, Bz, dx, dy, dz):
    """
        Returns:
            Divergence of magnetic field, dBx/dx + dBy/dy + dBz/dz
    """
    dBx = (Bx[1:, :-1, :] - Bx[:-1, :-1, :]) / dx
    dBy = (By[:-1, 1:, :] - By[:-1, :-1, :]) / dy
    dBz = 0
    return dBx + dBy + dBz

def plot_grid(gamma, U, t=0, plot="density", x=None, extent=None):
    rho, u, v, w, p, Bx, By, Bz = get_prims(gamma, U)
    labels = {"density": r"$\rho$", "u": r"$u$",
              "v": r"$v$", "w": r"$w$", "pressure": r"$P$", "energy": r"$E$", "Bx": r"$B_x$", "By": r"$B_y$", "Bz": r"$B_z$", "div": r"div $B$"}
    plt.cla()
    if extent:  # plot in 2D
        if plot == "density":
            c = plt.imshow(np.transpose(rho[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "u":
            c = plt.imshow(np.transpose(u[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "v":
            c = plt.imshow(np.transpose(v[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "w":
            c = plt.imshow(np.transpose(v[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "pressure":
            c = plt.imshow(np.transpose(p[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "Bx":
            c = plt.imshow(np.transpose(Bx[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "By":
            c = plt.imshow(np.transpose(By[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "Bz":
            c = plt.imshow(np.transpose(By[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent)
        elif plot == "div":
            res_x, res_y, res_z = Bx.shape
            dx, dy, dz = (extent[1] - extent[0]) / res_x, (extent[3] - extent[2]) / res_y, 2
            div = divergence(Bx, By, Bz, dx, dy, dz)
            c = plt.imshow(np.transpose(div[:, :, 0]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=extent, vmin=-10, vmax=10)
            
        plt.colorbar(c, label=labels[plot])

    else:  # plot in 1D (x)
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
        elif plot == "Bx":
            plt.plot(x, Bx[:, 0, 0], label=labels[plot])
        elif plot == "By":
            plt.plot(x, By[:, 0, 0], label=labels[plot])
        elif plot == "Bz":
            plt.plot(x, Bz[:, 0, 0], label=labels[plot])

        plt.xlabel("x")
        plt.legend()

    plt.title(f"t = {t:.2f}")
