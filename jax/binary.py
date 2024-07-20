import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, lax, Array
import matplotlib.pyplot as plt

from hydro import Hydro, IdealGas, Lattice, Coords, run, Primitives, Conservatives
from helpers import Boundary, cartesian_to_polar, plot_grid

class Isothermal(Hydro):
    def E(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike) -> Array:
        rho, u, v, p = prims
        return (p / (self.gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))

    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike) -> Array:
        rho, u, v, p = prims
        return jnp.sqrt(self.gamma * p / rho)

    def P(self, cons: Conservatives, X1: ArrayLike, X2: ArrayLike) -> Array:
        rho, u, v, e = cons
        return rho * self.c_s(cons, X1, X2) ** 2

if __name__ == "__main__":
    pass