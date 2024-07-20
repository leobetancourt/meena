import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, lax, Array
import matplotlib.pyplot as plt
import h5py

from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod

from helpers import Coords, get_prims, linspace_cells, logspace_cells, print_progress_bar, plot_grid
from flux import interface_flux

type Primitives = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
type Conservatives = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
type BoundaryCondition = tuple[str, str]


class Lattice:
    def __init__(self, coords: str, num_g: int, bc_x1: BoundaryCondition, bc_x2: BoundaryCondition, nx1: int, nx2: int, x1_range: tuple[float, float], x2_range: tuple[float, float], log_x1: bool = False, log_x2: bool = False):
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

    @abstractmethod
    def E(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike) -> Array:
        pass

    @abstractmethod
    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike) -> Array:
        pass

    @abstractmethod
    def P(self, cons: Conservatives, X1: ArrayLike, X2: ArrayLike) -> Array:
        pass


class IdealGas(Hydro):
    def E(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike) -> Array:
        rho, u, v, p = prims
        return (p / (self.gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))

    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike) -> Array:
        rho, u, v, p = prims
        return jnp.sqrt(self.gamma * p / rho)

    def P(self, cons: Conservatives, X1: ArrayLike, X2: ArrayLike) -> Array:
        rho, u, v, e = cons
        return (self.gamma - 1) * (e - (0.5 * rho * (u ** 2 + v ** 2)))


def cartesian_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> float:
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2)
    c_s = hydro.c_s((rho, u, v, p), lattice.X1, lattice.X2)
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + c_s))
    dt2 = jnp.min(lattice.dX2 / (jnp.abs(v) + c_s))
    return hydro.cfl * jnp.minimum(dt1, dt2)


def polar_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> float:
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2)
    c_s = hydro.c_s((rho, u, v, p), lattice.X1, lattice.X2)
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + c_s))
    dt2 = jnp.min(lattice.X1 * lattice.dX2 / (jnp.abs(v) + c_s))
    return hydro.cfl * jnp.minimum(dt1, dt2)


def compute_timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> float:
    if lattice.coords == Coords.CARTESIAN:
        return cartesian_timestep(hydro, lattice, U)
    elif lattice.coords == Coords.POLAR:
        return polar_timestep(hydro, lattice, U)


def solve_cartesian(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> Array:
    F_l, F_r, G_l, G_r = interface_flux(hydro, lattice, U)
    return - ((F_r - F_l) / lattice.dX1[..., jnp.newaxis]) - ((G_r - G_l) / lattice.dX2[..., jnp.newaxis])


def solve_polar(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> Array:
    F_l, F_r, G_l, G_r = interface_flux(hydro, lattice, U)
    rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2)

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

    return - ((X1_r * F_r - X1_l * F_l) / (X1 * dX1)) - ((G_r - G_l) / (X1 * dX2)) + S


def solve(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> Array:
    if lattice.coords == Coords.CARTESIAN:
        return solve_cartesian(hydro, lattice, U)
    elif lattice.coords == Coords.POLAR:
        return solve_polar(hydro, lattice, U)


@partial(jit, static_argnames=["hydro", "lattice"])
def first_order_step(hydro: Hydro, lattice: Lattice, U: ArrayLike) -> tuple[Array, float]:
    dt = compute_timestep(hydro, lattice, U)
    L = solve(hydro, lattice, U)
    U = U + L * dt
    # # add source terms
    # for s in self.S:
    #     u += s(u) * self.dt
    # self.U[g:-g, g:-g, :] = u
    return U, dt

# resizes h5py dataset and saves d


def save_to_dset(dset, d):
    dset.resize(dset.shape[0] + 1, axis=0)
    dset[-1] = d


def run(hydro, lattice, U, T=1, plot=None, out="./out", save_interval=0.1):
    PATH = f"{out}/out.h5"
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$",
              "v": r"$v$", "pressure": r"$P$", "energy": r"$E$", }

    t = 0
    next_checkpoint = 0

    if plot:
        rho, u, v, p = get_prims(hydro, U, lattice.X1, lattice.X2)
        matrix = rho
        fig, ax, c, cb = plot_grid(
            matrix, label=labels[plot], coords=lattice.coords, x1=lattice.x1, x2=lattice.x2)
        ax.set_title(f"t = {t:.2f}")

    with h5py.File(PATH, "w") as f:
        # metadata
        f.attrs["coords"] = lattice.coords
        f.attrs["gamma"] = hydro.gamma
        f.attrs["x1"] = lattice.x1
        f.attrs["x2"] = lattice.x2

        # create h5 datasets for time, conserved variables and diagnostics
        max_shape = (None, lattice.nx1, lattice.nx2)
        dset_t = f.create_dataset("t", (0,), maxshape=(
            None,), chunks=True, dtype="float64")  # simulation times
        dset_tc = f.create_dataset("tc", (0,), maxshape=(
            None,), chunks=True, dtype="float64")  # checkpoint times
        dset_rho = f.create_dataset(
            "rho", (0, lattice.nx1, lattice.nx2), maxshape=max_shape, chunks=True, dtype="float64")
        dset_momx1 = f.create_dataset(
            "momx1", (0, lattice.nx1, lattice.nx2), maxshape=max_shape, chunks=True, dtype="float64")
        dset_momx2 = f.create_dataset(
            "momx2", (0, lattice.nx1, lattice.nx2), maxshape=max_shape, chunks=True, dtype="float64")
        dset_E = f.create_dataset(
            "E", (0, lattice.nx1, lattice.nx2), maxshape=max_shape, chunks=True, dtype="float64")

        while t < T:
            save_to_dset(dset_t, t)
            # at each checkpoint, save the current state excluding the ghost cells
            if t >= next_checkpoint:
                save_to_dset(dset_tc, t)
                save_to_dset(dset_rho, U[..., 0])
                save_to_dset(dset_momx1, U[..., 1])
                save_to_dset(dset_momx2, U[..., 2])
                save_to_dset(dset_E, U[..., 3])

                next_checkpoint += save_interval

            U, dt = first_order_step(hydro, lattice, U)

            if plot:
                matrix = U[:, :, 0]
                if lattice.coords == "cartesian":
                    c.set_data(matrix)
                elif lattice.coords == "polar":
                    c.set_array(matrix.ravel())
                c.set_clim(vmin=jnp.min(matrix), vmax=jnp.max(matrix))
                cb.update_normal(c)
                ax.set_title(f"t = {t:.2f}")
                fig.canvas.draw()
                plt.pause(0.0001)

            t = t + dt if (t + dt <= T) else T

            print_progress_bar(t, T, suffix="complete", length=50)
