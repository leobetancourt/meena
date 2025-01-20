from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition

@dataclass(frozen=True)
class RayleighTaylor(Hydro):
    g: float = -0.1
    gamma_ad: float = 1.4
    nx: int = 500
    
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        x, y = X1, X2
        # velocity perturbation is 5% of characteristic sound speed
        v = 0.01 * (1 + jnp.cos(4 * jnp.pi * x)) * (1 + jnp.cos(3 * jnp.pi * y)) / 4
        rho = jnp.zeros_like(x)
        rho = rho.at[y > 0].set(2)
        rho = rho.at[y <= 0].set(1)
        p = 2.5 + self.g * rho * y

        return jnp.array([
            rho,
            jnp.zeros_like(x),
            v,
            p
        ]).transpose((1, 2, 0))

    def gamma(self) -> float:
        return self.gamma_ad

    def t_end(self) -> float:
        return 20

    def time_order(self) -> int:
        return 2

    def solver(self) -> str:
        return "hllc"

    def PLM(self) -> float:
        return True

    def save_interval(self) -> float:
        return 0.1
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-0.25, 0.25), (-0.75, 0.75))

    def resolution(self) -> tuple[int, int]:
        return (self.nx, self.nx * 3)

    def bc_x1(self) -> BoundaryCondition:
        return ("periodic", "periodic")

    def bc_x2(self) -> BoundaryCondition:
        return ("reflective", "reflective")

    def source(self, U: ArrayLike, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho = U[..., 0]
        v = U[..., 2] / rho
        zero = jnp.zeros_like(rho)
        
        return jnp.array([
            zero,
            zero,
            rho * self.g,
            rho * (self.g * v)
        ]).transpose((1, 2, 0))