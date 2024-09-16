from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from hydrocode import Hydro, BoundaryCondition

@dataclass(frozen=True)
class RT(Hydro):
    g: float = -0.1
    
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        t = 0
        x, y = X1, X2
        cs = self.c_s((2, 0, 0, 2.5), X1, X2, t)
        # velocity perturbation is 5% of characteristic sound speed
        v = (cs * 0.05) * (1 - jnp.cos(4 * jnp.pi * x)) * (1 - jnp.cos(4 * jnp.pi * y / 3))
        rho = jnp.zeros_like(x)
        rho = rho.at[y >= 0.75].set(2)
        rho = rho.at[y < 0.75].set(1)
        p = 2.5 + self.g * rho * (y - 0.75)
        
        return jnp.array([
            rho,
            jnp.zeros_like(x),
            rho * v,
            self.E((rho, 0, v, p), X1, X2, t)
        ]).transpose((1, 2, 0))
        
    def t_end(self) -> float:
        return 18
    
    def solver(self) -> str:
        return "hllc"
    
    def PLM(self) -> float:
        return True
    
    def save_interval(self) -> float:
        return 0.1
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0, 0.5), (0, 1.5))
        
    def resolution(self) -> tuple[int, int]:
        return (300, 750)
    
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