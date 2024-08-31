from __future__ import annotations

from functools import partial
import os

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, Array
import matplotlib.pyplot as plt


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hydrocode import Hydro, Lattice
    
from ..common.log import Logger
from ..common.helpers import get_prims, plot_grid, append_row_csv, create_csv_file, save_to_h5
from .flux import interface_flux


def cartesian_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    c_s = hydro.c_s((rho, u, v, p), lattice.X1, lattice.X2, t)
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + c_s))
    dt2 = jnp.min(lattice.dX2 / (jnp.abs(v) + c_s))
    return hydro.cfl() * jnp.minimum(dt1, dt2)


def polar_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    c_s = hydro.c_s((rho, u, v, p), lattice.X1, lattice.X2, t)
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + c_s))
    dt2 = jnp.min(lattice.X1 * lattice.dX2 / (jnp.abs(v) + c_s))
    return hydro.cfl() * jnp.minimum(dt1, dt2)


def compute_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    if lattice.coords == "cartesian":
        return cartesian_timestep(hydro, lattice, U, t)
    elif lattice.coords == "polar":
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
    if lattice.coords == "cartesian":
        return solve_cartesian(hydro, lattice, U, t)
    elif lattice.coords == "polar":
        return solve_polar(hydro, lattice, U, t)


@partial(jit, static_argnames=["hydro", "lattice"])
def first_order_step(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> tuple[Array, float]:
    dt = compute_timestep(hydro, lattice, U, t)
    L, flux = solve(hydro, lattice, U, t)
    U = U + L * dt + hydro.source(U, lattice.X1, lattice.X2, t) * dt
    return U, flux, dt


def get_matrix_to_plot(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float, plot: str):
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    e = U[:, :, 3]
    if plot == "density":
        matrix = rho
    elif plot == "log density":
        matrix = jnp.log10(rho)
    elif plot == "u":
        matrix = u
    elif plot == "v":
        matrix = v
    elif plot == "pressure":
        matrix = p
    elif plot == "energy":
        matrix = e
        
    return matrix

def run(hydro, lattice, U, t=0, T=1, N=None, plot=None, plot_range=None, out="./out", save_interval=None, diagnostics: ArrayLike = []):
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$",
              "v": r"$v$", "pressure": r"$P$", "energy": r"$E$", }

    saving = save_interval is not None

    if saving or len(diagnostics) > 0:
        os.makedirs(out, exist_ok=True)

    if saving:
        os.makedirs(f"{out}/checkpoints", exist_ok=True)

    if len(diagnostics) > 0:
        diag_file = f"{out}/diagnostics.csv"
        if not os.path.isfile(diag_file):
            headers = ["t", "dt"]
            headers.extend([name for name, _ in diagnostics])
            create_csv_file(diag_file, headers)

    if plot:
        matrix = get_matrix_to_plot(hydro, lattice, U, t, plot)
        fig, ax, c, cb = plot_grid(
            matrix, label=labels[plot], coords=lattice.coords, x1=lattice.x1, x2=lattice.x2, vmin=None, vmax=None)
        ax.set_title(f"t = {t:.2f}")
        
    with Logger() as logger:
        n = 1
        next_checkpoint = t
        while (N is None and t < T) or (N is not None and n < N):
            U_, flux, dt = first_order_step(hydro, lattice, U, t)

            if len(diagnostics) > 0:
                # save diagnostics
                diag_values = [get_val(hydro, lattice, U, flux, t)
                               for _, get_val in diagnostics]
                values = [t, dt]
                values.extend(diag_values)
                append_row_csv(diag_file, values)
    
            # at each checkpoint, save the conserved variables in every zone
            if saving and t >= next_checkpoint:
                filename = f"{out}/checkpoints/out_{t:.2f}.h5"
                save_to_h5(filename, t, U, hydro, lattice)
                next_checkpoint += save_interval

            U = U_

            if plot:
                matrix = get_matrix_to_plot(hydro, lattice, U, t, plot)
                if plot_range:
                    vmin, vmax = plot_range[0], plot_range[1]
                else:
                    vmin, vmax = jnp.min(matrix), jnp.max(matrix)
                if lattice.coords == "cartesian":
                    c.set_data(jnp.transpose(matrix))
                elif lattice.coords == "polar":
                    c.set_array(matrix.ravel())
                c.set_clim(vmin=vmin, vmax=vmax)
                cb.update_normal(c)
                ax.set_title(f"t = {t:.2f}")
                fig.canvas.draw()
                plt.pause(0.001)

            t = t + dt if (t + dt <= T) else T
            n = n + 1

            logger.update_logs(lattice, n, t, dt)
