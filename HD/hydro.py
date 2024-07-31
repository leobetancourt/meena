import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, Array

from functools import partial
from dataclasses import dataclass
from abc import ABC

import os
import matplotlib.pyplot as plt

from helpers import Coords, get_prims, linspace_cells, logspace_cells, print_progress_bar, plot_grid, append_diagnostics, create_diagnostics_file, save_to_h5
from flux import interface_flux

type Primitives = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
type Conservatives = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
type BoundaryCondition = tuple[str, str]


class Lattice:
    def __init__(self, coords: str, bc_x1: BoundaryCondition, bc_x2: BoundaryCondition, nx1: int, nx2: int, x1_range: tuple[float, float], x2_range: tuple[float, float], num_g: int = 2, log_x1: bool = False, log_x2: bool = False):
        self.coords = coords
        self.num_g = num_g
        self.bc_x1 = bc_x1
        self.bc_x2 = bc_x2
        self.nx1, self.nx2 = nx1, nx2
        self.x1_min, self.x1_max = x1_range
        self.x2_min, self.x2_max = x2_range

        if log_x1:
            self.x1, self.x1_intf = logspace_cells(
                self.x1_min, self.x1_max, num=nx1)
        else:
            self.x1, self.x1_intf = linspace_cells(
                self.x1_min, self.x1_max, num=nx1)
        if log_x2:
            self.x2, self.x2_intf = logspace_cells(
                self.x2_min, self.x2_max, num=nx2)
        else:
            self.x2, self.x2_intf = linspace_cells(
                self.x2_min, self.x2_max, num=nx2)
        self.X1, self.X2 = jnp.meshgrid(self.x1, self.x2, indexing="ij")
        self.X1_INTF, _ = jnp.meshgrid(self.x1_intf, self.x2, indexing="ij")
        _, self.X2_INTF = jnp.meshgrid(self.x1, self.x2_intf, indexing="ij")
        self.dX1 = self.X1_INTF[1:, :] - self.X1_INTF[:-1, :]
        self.dX2 = self.X2_INTF[:, 1:] - self.X2_INTF[:, :-1]


@dataclass(frozen=True)
class Hydro(ABC):
    gamma: float = 5/3
    nu: float = 1e-3
    cfl: float = 0.4
    dt: float = None
    coords: str = "cartesian"

    def E(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho, u, v, p = prims
        return jnp.sqrt(self.gamma * p / rho)

    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        pass

    def P(self, cons: Conservatives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho, u, v, e = cons
        return (self.gamma - 1) * (e - (0.5 * rho * (u ** 2 + v ** 2)))

    def source(self, U: ArrayLike, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        return jnp.zeros_like(U)

    def check_U(self, lattice: Lattice, U: ArrayLike) -> Array:
        return U


def cartesian_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    c_s = hydro.c_s((rho, u, v, p), lattice.X1, lattice.X2, t)
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + c_s))
    dt2 = jnp.min(lattice.dX2 / (jnp.abs(v) + c_s))
    return hydro.cfl * jnp.minimum(dt1, dt2)


def polar_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    c_s = hydro.c_s((rho, u, v, p), lattice.X1, lattice.X2, t)
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + c_s))
    dt2 = jnp.min(lattice.X1 * lattice.dX2 / (jnp.abs(v) + c_s))
    return hydro.cfl * jnp.minimum(dt1, dt2)


def compute_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    if lattice.coords == Coords.CARTESIAN:
        return cartesian_timestep(hydro, lattice, U, t)
    elif lattice.coords == Coords.POLAR:
        return polar_timestep(hydro, lattice, U, t)


def solve_cartesian(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> tuple[Array, Array, Array, Array]:
    F_l, F_r, G_l, G_r = interface_flux(hydro, lattice, U, t)
    L = - ((F_r - F_l) / lattice.dX1[..., jnp.newaxis]) - \
        ((G_r - G_l) / lattice.dX2[..., jnp.newaxis])
    flux = F_l, F_r, G_l, G_r
    return L, flux


def solve_polar(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> tuple[Array, Array, Array, Array]:
    F_l, F_r, G_l, G_r = interface_flux(hydro, lattice, U, t)
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)

    S = jnp.array([
        jnp.zeros_like(rho),
        (p / lattice.X1) + (rho * v ** 2) / lattice.X1,
        - rho * u * v / lattice.X1,
        jnp.zeros_like(rho)
    ]).transpose(1, 2, 0)

    dX1 = lattice.dX1[..., jnp.newaxis]
    dX2 = lattice.dX2[..., jnp.newaxis]
    X1_l, X1_r = lattice.X1_INTF[:-1, :,
                                 jnp.newaxis], lattice.X1_INTF[1:, :, jnp.newaxis]
    X1 = lattice.X1[..., jnp.newaxis]

    L = - ((X1_r * F_r - X1_l * F_l) / (X1 * dX1)) - \
        ((G_r - G_l) / (X1 * dX2)) + S
    flux = F_l, F_r, G_l, G_r
    return L, flux


def solve(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> tuple[Array, Array, Array, Array]:
    if lattice.coords == Coords.CARTESIAN:
        return solve_cartesian(hydro, lattice, U, t)
    elif lattice.coords == Coords.POLAR:
        return solve_polar(hydro, lattice, U, t)


@partial(jit, static_argnames=["hydro", "lattice"])
def first_order_step(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> tuple[Array, float]:
    if hydro.dt is None:
        dt = compute_timestep(hydro, lattice, U, t)
    else:
        dt = hydro.dt
    L, flux = solve(hydro, lattice, U, t)
    U = U + L * dt + hydro.source(U, lattice.X1, lattice.X2, t) * dt
    # U = hydro.check_U(U)
    return U, flux, dt


def run(hydro, lattice, U, t=0, T=1, plot=None, out="./out", save_interval=0.1, diagnostics: ArrayLike = []):
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$",
              "v": r"$v$", "pressure": r"$P$", "energy": r"$E$", }

    os.makedirs(f"{out}/checkpoints", exist_ok=True)
    diag_file = f"{out}/diagnostics.csv"
    if not os.path.isfile(diag_file):
        create_diagnostics_file(diagnostics, diag_file)

    next_checkpoint = 0

    if plot:
        rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)
        e = U[:, :, 3]
        vmin, vmax = None, None
        if plot == "density":
            matrix = rho
        elif plot == "log density":
            matrix = jnp.log10(rho)
            vmin, vmax = -3, 0.5
        elif plot == "u":
            matrix = u
        elif plot == "v":
            matrix = v
        elif plot == "pressure":
            matrix = p
        elif plot == "energy":
            matrix = e
        fig, ax, c, cb = plot_grid(
            matrix, label=labels[plot], coords=lattice.coords, x1=lattice.x1, x2=lattice.x2, vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {t:.2f}")

    while t < T:
        U_, flux, dt = first_order_step(hydro, lattice, U, t)

        # save diagnostics
        diag_values = [get_val(hydro, lattice, U, flux, t)
                       for _, get_val in diagnostics]
        append_diagnostics(diag_file, t, dt, diag_values)

        # at each checkpoint, save the conserved variables in every zone
        if t >= next_checkpoint:
            filename = f"{out}/checkpoints/out_{t:.2f}.h5"
            save_to_h5(filename, t, U, lattice.coords,
                       hydro.gamma, lattice.x1, lattice.x2)
            next_checkpoint += save_interval

        U = U_

        if plot:
            rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)
            e = U[:, :, 3]
            vmin, vmax = None, None
            if plot == "density":
                matrix = rho
            elif plot == "log density":
                matrix = jnp.log10(rho)
                vmin, vmax = -3, 0.5
            elif plot == "u":
                matrix = u
            elif plot == "v":
                matrix = v
            elif plot == "pressure":
                matrix = p
            elif plot == "energy":
                matrix = e
            if lattice.coords == "cartesian":
                c.set_data(matrix)
            elif lattice.coords == "polar":
                c.set_array(matrix.ravel())
            if vmin is None:
                vmin, vmax = jnp.min(matrix), jnp.max(matrix)
            c.set_clim(vmin=vmin, vmax=vmax)
            cb.update_normal(c)
            ax.set_title(f"t = {t:.2f}")
            fig.canvas.draw()
            plt.pause(0.001)

        t = t + dt if (t + dt <= T) else T

        print_progress_bar(t, T, suffix="complete", length=50)
