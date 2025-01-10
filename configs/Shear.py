from dataclasses import dataclass

import os
import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition

@dataclass(frozen=True)
class Shear(Hydro):
    res: int = 300
    gamma_ad: float = 5.0 / 3.0
    
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        x, y = X1, X2
        
        rho = jnp.ones_like(x)
        u = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        p = jnp.ones_like(x)

        u = jnp.where(y >= 0, 0.5, u)
        u = jnp.where(y < 0, -0.5, u)

        return jnp.array([
            rho,
            rho * u,
            rho * v,
            self.E((rho, u, v, p))
        ]).transpose((1, 2, 0))

    def theta_PLM(self) -> float:
        return 1.5

    def gamma(self) -> float:
        return self.gamma_ad

    def nu(self) -> float:
        return 1e-3

    def t_end(self) -> float:
        return 10

    def solver(self) -> str:
        return "hll"

    def PLM(self) -> float:
        return True
 
    def save_interval(self) -> float:
        return 1

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-0.5, 0.5), (-0.5, 0.5))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)

    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("outflow", "outflow")
