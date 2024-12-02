from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, Lattice, BoundaryCondition
from src.common.helpers import cartesian_to_polar

@dataclass(frozen=True)
class SedovBlast(Hydro):    
    def initialize(self, lattice: Lattice) -> Array:
        radius = 0.1
        x, y = lattice.X1, lattice.X2
        r, _ = cartesian_to_polar(x, y)
        U = jnp.zeros(shape=(*x.shape, 4))
        
        U = U.at[r < radius].set(jnp.array([1, 0, 0, 10]))
        U = U.at[r >= radius].set(jnp.array([1, 0, 0, self.E((1, 1e-4, 0, 0))]))
        
        return U
        
    def t_end(self) -> float:
        return 10
    
    def PLM(self) -> float:
        return True
    
    def time_order(self) -> int:
        return 2
    
    def solver(self) -> float:
        return "hll"
    
    def coords(self) -> str:
        return "cartesian"
    
    def save_interval(self) -> float:
        return 0.5
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0, 1), (0, 1))
        
    def resolution(self) -> tuple[int, int]:
        return (1000, 1000)
    
    def bc_x1(self) -> BoundaryCondition:
        return ("reflective", "reflective")

    def bc_x2(self) -> BoundaryCondition:
        return ("reflective", "reflective")