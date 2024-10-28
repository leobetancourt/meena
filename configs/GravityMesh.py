from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition
from src.common.helpers import cartesian_to_polar

@dataclass(frozen=True)
class GravityMesh(Hydro):
    res: int = 200
    gamma_ad: float = 5.0 / 3.0
    
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        x, y = X1, X2
        r, theta = cartesian_to_polar(x, y)
        
        a = 0.2
        rho = np.exp(-(r ** 2) / (2 * a))
        u = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        p = jnp.ones_like(x) * 1

        return jnp.array([
            rho,
            rho * u,
            rho * v,
            self.E((rho, u, v, p))
        ]).transpose((1, 2, 0))
        
    def self_gravity(self) -> bool:
        return True

    def theta_PLM(self) -> float:
        return 1.5

    def gamma(self) -> float:
        return self.gamma_ad

    def t_end(self) -> float:
        return 5

    def solver(self) -> str:
        return "hll"

    def PLM(self) -> float:
        return True

    def save_interval(self) -> float:
        return 0.1

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-1, 1), (-1, 1))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)

    def bc_x1(self) -> BoundaryCondition:
        return ("periodic", "periodic")

    def bc_x2(self) -> BoundaryCondition:
        return ("periodic", "periodic")
