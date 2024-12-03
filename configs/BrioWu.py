from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from meena import Hydro, Lattice, Prims, Cons, BoundaryCondition
from src.common.helpers import cartesian_to_spherical

@dataclass(frozen=True)
class BrioWu(Hydro):
      
    def initialize(self, lattice: Lattice) -> Array:
        x = lattice.X1
        x_intf, y_intf, z_intf = lattice.X1_INTF, lattice.X2_INTF, lattice.X3_INTF

        rho = jnp.where(x < 0.5, 1, 0.125)
        v = jnp.zeros_like(x)
        p = jnp.where(x < 0.5, 1, 0.1)
        Bx = jnp.ones_like(x_intf) * 0.75
        By = jnp.ones_like(y_intf) * jnp.where(x < 0.5, 1, -1)
        Bz = jnp.ones_like(z_intf)
        bx, by, bz = 0.5 * (Bx[:-1] + Bx[1:]), 0.5 * (By[:, :-1] + Bx[: 1:]), 0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])
        
        e = self.E((rho, v, v, v, p, bx, by, bz))
        
        U = jnp.array([
            rho,
            v,
            v,
            v,
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
    
    def gamma(self) -> float:
        return 2
    
    def cfl(self) -> float:
        return 0.4
        
    def t_end(self) -> float:
        return 0.1
    
    def regime(self) -> str:
        return "MHD"
    
    def solver(self) -> float:
        return "hll"
    
    def coords(self) -> str:
        return "cartesian"
    
    def save_interval(self) -> float:
        return 0.01
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (((0, 1), (0, 1), (0, 1)))
        
    def resolution(self) -> tuple[int, int]:
        return (500, 1, 1)
    
    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")