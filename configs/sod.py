from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition, Primitives, Conservatives

@dataclass(frozen=True)
class SodShockTube(Hydro):    
    def initialize(self, X1: ArrayLike) -> Array:
        x = X1
                
        left_state = jnp.array([1, 0, 1])
        right_state = jnp.array([0.125, 0, 0.1])
        
        return jnp.where(x[:, None] < 0.5, left_state, right_state)
    
    def dim(self) -> int:
        return 1
    
    def E(self, prims: Primitives, X1: ArrayLike, t: float) -> Array:
        rho, u, p = prims[..., 0], prims[..., 1], prims[..., 2]
        return p / (self.gamma() - 1) + 0.5 * rho * (u ** 2)

    def c_s(self, prims: Primitives, X1: ArrayLike, t: float) -> Array:
        rho, p = prims[..., 0], prims[..., 2]
        return jnp.sqrt(self.gamma() * p / rho)

    def P(self, cons: Conservatives, X1: ArrayLike, t: float) -> Array:
        rho, mom_x, e = cons[..., 0], cons[..., 1], cons[..., 2]
        u = mom_x / rho
        return (self.gamma() - 1) * (e - (0.5 * rho * (u ** 2)))
        
    def t_end(self) -> float:
        return 0.21
    
    def gamma(self) -> float:
        return 1.4
    
    def PLM(self) -> float:
        return False
    
    def time_order(self) -> int:
        return 2
    
    def solver(self) -> float:
        return "hll"
    
    def coords(self) -> str:
        return "cartesian"
    
    def save_interval(self) -> float:
        return 0.01
        
    def range(self) -> tuple[float, float]:
        return (0, 1)
        
    def resolution(self) -> int:
        return 300
    
    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")