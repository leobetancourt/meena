from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, Lattice, Prims, Cons, BoundaryCondition
from src.common.helpers import cartesian_to_spherical

@dataclass(frozen=True)
class BlastWave(Hydro):
    L: float = 1
      
    def initialize(self, lattice: Lattice) -> Array:
        radius = 0.1
        x, y, z = lattice.X1, lattice.X2, lattice.X3
        r, theta, phi = cartesian_to_spherical(x, y, z)
        
        U = jnp.zeros(shape=(x.shape[0], x.shape[1], x.shape[2], 4))
        
        rho_0 = 1
        v_0 = 0
        p_0 = 0.1
        p_b = 100
        B = (1/jnp.sqrt(2), 1/jnp.sqrt(2), 0)
        prims_0 = (1, *v_0, p_0, *B)
        prims_b = (1, *v_0, p_b, *B)
        U = U.at[r < radius].set(jnp.array([rho_0, v_0, v_0, v_0, self.E(prims_0), B[0], B[1], B[2]]))
        U = U.at[r < radius].set(jnp.array([rho_0, v_0, v_0, v_0, self.E(prims_b), B[0], B[1], B[2]]))
        
        return U
    
    def E(self, prims: Prims) -> Array:
        rho, u, v, w, p, Bx, By, Bz = prims
        u2 = u**2 + v**2 + w**2
        B2 = Bx**2 + By**2 + Bz**2
        return p / (self.gamma() - 1) + 0.5 * (rho * u2 + B2)

    def c_s(self, prims: Prims) -> Array:
        rho, _, _, _, p, _, _, _ = prims
        return jnp.sqrt(self.gamma() * p / rho)

    def P(self, cons: Cons) -> Array:
        rho, momx1, momx2, momx3, e, Bx, By, Bz = cons
        u, v, w = momx1/rho, momx2/rho, momx3/rho
        u2 = u**2 + v**2 + w**2
        B2 = Bx**2 + By**2 + Bz**2
        return (self.gamma() - 1) * (e - 0.5 * (rho * u2 + B2))
        
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
        return ((0, self.L), (0, 1.5*self.L), (0, self.L))
        
    def resolution(self) -> tuple[int, int]:
        return (200, 300, 200)
    
    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("outflow", "outflow")
    
    def bc_x3(self) -> BoundaryCondition:
        return ("outflow", "outflow")