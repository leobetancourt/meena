from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from hydrocode import Hydro, BoundaryCondition
from src.common.helpers import cartesian_to_polar

@dataclass(frozen=True)
class SedovBlast(Hydro):    
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        radius = 0.2
        r, _ = X1, X2
        # r, _ = cartesian_to_polar(x, y)
        U = jnp.zeros(shape=(X1.shape[0], X1.shape[1], 4))
        
        U = U.at[r < radius].set(jnp.array([1, 0, 0, 10]))
        U = U.at[r >= radius].set(jnp.array([1, 0, 0, self.E((1, 1e-4, 0, 0))]))
        
        return U
        
    def t_end(self) -> float:
        return 10
    
    def PLM(self) -> float:
        return True
    
    def coords(self) -> str:
        return "polar"
    
    def save_interval(self) -> float:
        return 0.1
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0.1, 1), (0, 2 * jnp.pi))
        
    def resolution(self) -> tuple[int, int]:
        return (100, 600)
    
    def log_x1(self) -> bool:
        return True
    
    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("periodic", "periodic")