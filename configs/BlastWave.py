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
        x_intf, y_intf, z_intf = lattice.X1_INTF, lattice.X2_INTF, lattice.X3_INTF
        r = jnp.sqrt(x**2 + y**2 + z**2)
            
        rho_0 = jnp.ones_like(x)
        v_0 = jnp.zeros_like(x)
        p_0 = 0.1 * jnp.ones_like(x)
        p_b = 100 * jnp.ones_like(x)
        Bx = jnp.ones_like(x_intf) * 1/jnp.sqrt(2)
        By = jnp.ones_like(y_intf) * 1/jnp.sqrt(2)
        Bz = jnp.zeros_like(z_intf)
        bx, by, bz = 0.5 * (Bx[:-1] + Bx[1:]), 0.5 * (By[:, :-1] + Bx[: 1:]), 0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])
        
        prims_0 = (rho_0, v_0, v_0, v_0, p_0, bx, by, bz)
        prims_b = (rho_0, v_0, v_0, v_0, p_b, bx, by, bz)
        e_0, e_b = self.E(prims_0), self.E(prims_b)
        e = jnp.where(r < radius, e_b, e_0)
        
        U = jnp.array([
            rho_0,
            v_0,
            v_0,
            v_0,
            e,
            bx,
            by,
            bz
        ]).transpose(1, 2, 3, 0)
        
        return U, (Bx, By, Bz)
    
    def E(self, prims: Prims, *args) -> Array:
        rho, u, v, w, p, Bx, By, Bz = prims
        u2 = u**2 + v**2 + w**2
        B2 = Bx**2 + By**2 + Bz**2
        return p / (self.gamma() - 1) + 0.5 * (rho * u2 + B2)

    def c_s(self, prims: Prims, *args) -> Array:
        rho, _, _, _, p, _, _, _ = prims
        return jnp.sqrt(self.gamma() * p / rho)

    def P(self, cons: Cons, *args) -> Array:
        rho, momx1, momx2, momx3, e, Bx, By, Bz = cons
        u, v, w = momx1/rho, momx2/rho, momx3/rho
        u2 = u**2 + v**2 + w**2
        B2 = Bx**2 + By**2 + Bz**2
        return (self.gamma() - 1) * (e - 0.5 * (rho * u2 + B2))
    
    def cfl(self) -> float:
        return 0.4
        
    def t_end(self) -> float:
        return 10
    
    def regime(self) -> str:
        return "MHD"
    
    def solver(self) -> float:
        return "hll"
    
    def coords(self) -> str:
        return "cartesian"
    
    def save_interval(self) -> float:
        return 0.1
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-0.5 * self.L, 0.5 * self.L), (-0.75 * self.L, 0.75*self.L), (-0.5 * self.L, 0.5 * self.L))
        
    def resolution(self) -> tuple[int, int]:
        return (100, 150, 100)
    
    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("outflow", "outflow")
    
    def bc_x3(self) -> BoundaryCondition:
        return ("outflow", "outflow")