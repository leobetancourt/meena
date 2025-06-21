from __future__ import annotations

from functools import partial
import os

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, Array, debug
import matplotlib.pyplot as plt


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from meena import Hydro, Lattice
    
from ..common.log import Logger
from ..common.helpers import plot_grid, append_row_csv, create_csv_file, save_to_h5, cartesian_to_polar
from .hd import U_from_prim, get_prims, interface_flux

def gravity_mesh(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> Array:
    """
        Implements self-gravity by solving Poisson's equation on a mesh using the Fast-Fourier-Transform
    """
    rho = U[..., 0]
    u, v, = U[..., 1] / rho, U[..., 2] / rho
    rho_hat = jnp.fft.fft2(rho)
    dx1, dx2 = lattice.x1[1] - lattice.x1[0], lattice.x2[1] - lattice.x2[0]
    kx = jnp.fft.fftfreq(lattice.nx1, d=dx1) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(lattice.nx2, d=dx2) * 2 * jnp.pi
    kx, ky = jnp.meshgrid(kx, ky, indexing='ij')

    # Avoid division by zero at (kx, ky) = (0, 0)
    k_squared = kx**2 + ky**2
    k_squared = k_squared.at[0, 0].set(1.0)  # Temporarily set to avoid division by zero

    # Calculate the potential in Fourier space
    phi_hat = -4 * jnp.pi * hydro.G() * rho_hat / k_squared
    phi_hat = phi_hat.at[0, 0].set(0)
    
    phi = jnp.fft.ifft2(phi_hat).real
    gx1, gx2 = jnp.gradient(phi, dx1, dx2)
    gx1, gx2 = -gx1, -gx2
    
    return jnp.array([
        jnp.zeros_like(rho),
        gx1,
        gx2,
        (u * gx1 + v * gx2)
    ]).transpose((1, 2, 0))

def cartesian_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    prims = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    c_s = hydro.c_s(prims, lattice.X1, lattice.X2, t)
    u, v = prims[..., 1], prims[..., 2]
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + c_s))
    dt2 = jnp.min(lattice.dX2 / (jnp.abs(v) + c_s))
    return hydro.cfl() * jnp.minimum(dt1, dt2)


def polar_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> float:
    prims = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    c_s = hydro.c_s(prims, lattice.X1, lattice.X2, t)
    u, v = prims[..., 1], prims[..., 2]
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
    prims = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    rho, u, v, p = prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]

    S = jnp.array([
        jnp.zeros_like(rho),
        (p / lattice.X1) + (rho * v ** 2) / lattice.X1,
        - rho * u * v / lattice.X1,
        jnp.zeros_like(rho)
    ]).transpose((1, 2, 0))

    dX1 = lattice.dX1[..., jnp.newaxis]
    dX2 = lattice.dX2[..., jnp.newaxis]
    X1_l, X1_r = lattice.X1_INTF[:-1, :,
                                 jnp.newaxis], lattice.X1_INTF[1:, :, jnp.newaxis]
    X1 = lattice.X1[..., jnp.newaxis]

    L = - ((X1_r * F_r - X1_l * F_l) / (X1 * dX1)) - \
        ((G_r - G_l) / (X1 * dX2)) + S
    flux = F_l, F_r, G_l, G_r
    
    if hydro.inflow():
        F_l = F_l.at[0, :, 0].set(jnp.minimum(F_l[0, :, 0], 0.0))
        F_l = F_l.at[0, :, 1].set(jnp.minimum(F_l[0, :, 1], 0.0))
    
    return L, flux


def solve(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> tuple[Array, Array, Array, Array]:
    if lattice.coords == "cartesian":
        return solve_cartesian(hydro, lattice, U, t)
    elif lattice.coords == "polar":
        return solve_polar(hydro, lattice, U, t)

@partial(jit, static_argnames=["hydro", "lattice"])
def step(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float) -> tuple[Array, float]:
    if hydro.timestep():
        dt = hydro.timestep()
    else:
        dt = compute_timestep(hydro, lattice, U, t)

    L1, flux = solve(hydro, lattice, U, t)
    S1 = hydro.source(U, lattice.X1, lattice.X2, t)
    if hydro.time_order() == 1: # forward Euler
        U = U + L1 * dt + S1 * dt
    elif hydro.time_order() == 2: # RK2
        U2 = U + (L1 * dt / 2) + (S1 * dt / 2)
        L2, flux = solve(hydro, lattice, U2, t + (dt / 2))
        S2 = hydro.source(U2, lattice.X1, lattice.X2, t + (dt / 2))
        U = U + L2 * dt + S2 * dt
        
    U = hydro.check_U(lattice, U, t)

    return U, flux, dt


def get_matrix_to_plot(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float, plot: str):
    prims = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    rho, u, v, p = prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]
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
    elif plot == "dt":
        c_s = hydro.c_s(prims, lattice.X1, lattice.X2, t)
        matrix = jnp.log(jnp.minimum(lattice.dX1 / (jnp.abs(u) + c_s), lattice.dX2 / (jnp.abs(v) + c_s)))
        
    return matrix

def run(hydro, lattice, prims, t=0, T=1, N=None, plot=None, save_plots=False, plot_range=None, out="./out", save_interval=None, diagnostics: ArrayLike = []):
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$",
              "v": r"$v$", "pressure": r"$P$", "energy": r"$E$", "dt": r"$\log dt$"}

    saving = save_interval is not None

    if saving or len(diagnostics) > 0:
        os.makedirs(out, exist_ok=True)
    
    if saving:
        os.makedirs(f"{out}/checkpoints", exist_ok=True)
    
    if save_plots:
        os.makedirs(f"{out}/plots", exist_ok=True)

    if len(diagnostics) > 0:
        diag_file = f"{out}/diagnostics.csv"
        if not os.path.isfile(diag_file):
            headers = ["t", "dt"]
            headers.extend([name for name, _ in diagnostics])
            create_csv_file(diag_file, headers)

    U = U_from_prim(hydro, prims, lattice.X1, lattice.X2, t)
    if plot:
        matrix = get_matrix_to_plot(hydro, lattice, U, t, plot)
        fig, ax, c, cb = plot_grid(
            matrix, label=labels[plot], coords=lattice.coords, x1=lattice.x1, x2=lattice.x2, vmin=None, vmax=None)
        ax.set_title(f"t = {t:.2f}")
        
    with Logger() as logger:
        n = 1
        next_checkpoint = t
        while (N is None and t < T) or (N is not None and n < N):
            U_, flux, dt = step(hydro, lattice, U, t)

            if len(diagnostics) > 0:
                # save diagnostics
                diag_values = [get_val(hydro, lattice, U, flux, t)
                               for _, get_val in diagnostics]
                values = [t, dt]
                values.extend(diag_values)
                append_row_csv(diag_file, values)
    
            # at each checkpoint, save the conserved variables in every zone
            if saving and t >= next_checkpoint:
                prims = get_prims(hydro, U, lattice.X1, lattice.X2, t)
                save_to_h5(out, t, prims, hydro, lattice, save_plots)
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
