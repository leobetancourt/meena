from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition
from src.common.helpers import cartesian_to_polar

@dataclass(frozen=True)
class SedovBlast(Hydro):    
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        radius = 0.1
        # r, _ = X1, X2
        x, y = X1, X2
        r, _ = cartesian_to_polar(x, y)
        U = jnp.zeros(shape=(X1.shape[0], X1.shape[1], 4))
        
        U = U.at[r < radius].set(jnp.array([1, 0, 0, 10]))
        U = U.at[r >= radius].set(jnp.array([1, 0, 0, self.E((1, 1e-4, 0, 0))]))
        
        return U
        
    def t_end(self) -> float:
        return 10
    
    def PLM(self) -> float:
        return True
    
    def solver(self) -> float:
        return "hll"
    
    def coords(self) -> str:
        return "cartesian"
    
    def save_interval(self) -> float:
        return 0.1
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0, 1), (0, 1))
        
    def resolution(self) -> tuple[int, int]:
        return (300, 300)
    
    def bc_x1(self) -> BoundaryCondition:
        return ("reflective", "reflective")

    def bc_x2(self) -> BoundaryCondition:
        return ("reflective", "reflective")