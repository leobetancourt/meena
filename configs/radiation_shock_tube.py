from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition, Primitives, Conservatives

@dataclass(frozen=True)
class RadiationShockTube(Hydro):    
    def initialize(self, X1: ArrayLike) -> Array:
        x = X1
        
        density = jnp.where(x < 0.5, 1, 0.125)
        velocity = jnp.zeros_like(x)
        pressure = jnp.where(x < 0.5, 1, 0.1)
        e_rad = jnp.ones_like(x) * 0.1
        
        return jnp.array([density, velocity, pressure, e_rad]).T
    
    def dim(self) -> int:
        return 1
    
    def radiation(self) -> bool:
        return True
    
    def c(self) -> float:
        return 1
    
    def kappa(self) -> float:
        return 1e3
    
    def E(self, prims: Primitives, X1: ArrayLike, t: float) -> Array:
        rho, u, p, e_rad = prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]
        return p / (self.gamma() - 1) + 0.5 * rho * (u ** 2) + e_rad

    def c_s(self, prims: Primitives, X1: ArrayLike, t: float) -> Array:
        rho, p = prims[..., 0], prims[..., 2]
        return jnp.sqrt(self.gamma() * p / rho)

    def P(self, cons: Conservatives, X1: ArrayLike, t: float) -> Array:
        rho, mom_x, E, e_rad = cons[..., 0], cons[..., 1], cons[..., 2], cons[..., 3]
        u = mom_x / rho
        return (self.gamma() - 1) * (E - 0.5 * rho * (u ** 2) - e_rad)
    
    def cfl(self) -> float:
        return 0.01
    
    def t_end(self) -> float:
        return 10
    
    def gamma(self) -> float:
        return (5.0 / 3.0)
    
    def PLM(self) -> float:
        return False
    
    def time_order(self) -> int:
        return 1
    
    def solver(self) -> float:
        return "finite-difference"
    
    def coords(self) -> str:
        return "cartesian"
    
    def save_interval(self) -> float:
        return 0.1
        
    def range(self) -> tuple[float, float]:
        return (0, 1)
        
    def resolution(self) -> int:
        return 500
    
    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")