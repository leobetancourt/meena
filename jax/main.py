import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, lax, Array
import matplotlib.pyplot as plt

from hydro import Hydro, IdealGas, Lattice, Coords, run
from helpers import Boundary, cartesian_to_polar, plot_grid

jax.config.update('jax_log_compiles', True)

def sedov(hydro: Hydro, lattice: Lattice, radius: float = 0.1) -> Array:
    if lattice.coords == Coords.CARTESIAN:
        r, _ = cartesian_to_polar(lattice.X1, lattice.X2)
    else:
        r, _ = lattice.X1, lattice.X2
    U = jnp.zeros((*r.shape, 4))
    U = U.at[r < radius].set(jnp.array([1, 0, 0, 10]))
    U = U.at[r >= radius].set(jnp.array([1, 0, 0, (1e-4 / (hydro.gamma - 1))]))
    return U


def main():
    hydro = IdealGas(gamma=5/3, nu=1e-3, cfl=0.4)
    lattice = Lattice(
        coords="cartesian",
        num_g=2,
        bc_x1=(Boundary.OUTFLOW, Boundary.OUTFLOW),
        bc_x2=(Boundary.OUTFLOW, Boundary.OUTFLOW),
        nx1=500,
        nx2=500,
        x1_range=(-1, 1),
        x2_range=(-1, 1)
    )

    U = sedov(hydro, lattice, radius=0.1)

    run(hydro, lattice, U, T=1, plot="density")


if __name__ == "__main__":
    main()
