from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition, Primitives, Conservatives

@dataclass(frozen=True)
class RadiationTest(Hydro):    
    def initialize(self, X1: ArrayLike) -> Array:
        x = X1
        
        density = jnp.ones_like(x)
        velocity = jnp.zeros_like(x)
        pressure = jnp.ones_like(x)
        p_rad = jnp.exp(-(x) ** 2 / (2 * 0.1 ** 2))
        
        return jnp.array([density, velocity, pressure, p_rad]).T
    
    def dim(self) -> int:
        return 1
    
    def radiation(self) -> bool:
        return True
    
    def c(self) -> float:
        return 1
    
    def kappa(self) -> float:
        return 1e2
    
    def E(self, prims: Primitives, X1: ArrayLike, t: float) -> Array:
        rho, u, p, p_rad = prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]
        return p / (self.gamma() - 1) + 0.5 * rho * (u ** 2) + (3 * p_rad)

    def c_s(self, prims: Primitives, X1: ArrayLike, t: float) -> Array:
        rho, p = prims[..., 0], prims[..., 2]
        return jnp.sqrt(self.gamma() * p / rho)

    def P(self, cons: Conservatives, X1: ArrayLike, t: float) -> Array:
        rho, mom_x, E, e_rad = cons[..., 0], cons[..., 1], cons[..., 2], cons[..., 3]
        u = mom_x / rho
        return (self.gamma() - 1) * (E - 0.5 * rho * (u ** 2) - (e_rad / 3))
    
    def cfl(self) -> float:
        return 0.01
    
    def t_end(self) -> float:
        return 1
    
    def gamma(self) -> float:
        return 2
    
    def PLM(self) -> float:
        return False
    
    def time_order(self) -> int:
        return 1
    
    def solver(self) -> float:
        return "hll"
    
    def coords(self) -> str:
        return "cartesian"
    
    def save_interval(self) -> float:
        return 0.01
        
    def range(self) -> tuple[float, float]:
        return (0, 1)
        
    def resolution(self) -> int:
        return 100
    
    def bc_x1(self) -> BoundaryCondition:
        return ("reflective", "reflective")