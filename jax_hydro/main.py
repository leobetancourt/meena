import jax.numpy as jnp
from jax import Array

from hydro import Hydro, Lattice, Coords, run
from helpers import Boundary, cartesian_to_polar

# jax.config.update('jax_log_compiles', True)


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
    hydro = Hydro(gamma=5/3, nu=1e-3, cfl=0.4, coords="polar")
    lattice = Lattice(
        coords="polar",
        bc_x1=(Boundary.OUTFLOW, Boundary.OUTFLOW),
        bc_x2=(Boundary.PERIODIC, Boundary.PERIODIC),
        nx1=50,
        nx2=300,
        x1_range=(0.05, 1),
        x2_range=(0, 2 * jnp.pi),
        log_x1=True
    )

    U = sedov(hydro, lattice, radius=0.1)

    run(hydro, lattice, U, T=1, out="../output/sedov", save_interval=0.05)


if __name__ == "__main__":
    main()
