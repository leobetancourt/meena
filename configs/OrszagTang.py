from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from meena import Hydro, Lattice, Prims, Cons, BoundaryCondition

@dataclass(frozen=True)
class OrszagTang(Hydro):
    cfl_num: float = 0.4
      
    def initialize(self, lattice: Lattice) -> Array:
        x, y = lattice.X1, lattice.X2
        x_intf, y_intf, z_intf = lattice.X1_INTF, lattice.X2_INTF, lattice.X3_INTF

        rho = jnp.ones_like(x)
        v_0 = 1
        u, v, w = v_0 * -jnp.sin(2 * jnp.pi * y), v_0 * jnp.sin(2 * jnp.pi * x), jnp.zeros_like(x)
        p = jnp.ones_like(x) / self.gamma()
        B_0 = 1 / self.gamma()
        x_repl = jnp.concatenate([x[:, 0:1], x], axis=1)
        y_repl = jnp.concatenate([y[0:1], y], axis=0)
        Bx = B_0 * -jnp.sin(2 * jnp.pi * y_repl)
        By = B_0 * jnp.sin(4 * jnp.pi * x_repl)
        Bz = jnp.zeros_like(z_intf)
        bx, by, bz = 0.5 * (Bx[:-1] + Bx[1:]), 0.5 * (By[:, :-1] + By[:, 1:]), 0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])
        
        e = self.E((rho, v, v, v, p, bx, by, bz))
        
        U = jnp.array([
            rho,
            u,
            v,
            w,
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
        u, v, w = momx1 / rho, momx2 / rho, momx3 / rho
        u2 = u**2 + v**2 + w**2
        B2 = Bx**2 + By**2 + Bz**2
        return (self.gamma() - 1) * (e - 0.5 * (rho * u2 + B2))
    
    def gamma(self) -> float:
        return 5.0 / 3.0
    
    def cfl(self) -> float:
        return self.cfl_num
        
    def t_end(self) -> float:
        return 1
    
    def regime(self) -> str:
        return "MHD"
    
    def solver(self) -> float:
        return "hll"
    
    def save_interval(self) -> float:
        return 0.01
        
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0, 1), (0, 1))
        
    def resolution(self) -> tuple[int, int]:
        return (400, 400)
    
    def bc_x1(self) -> BoundaryCondition:
        return ("periodic", "periodic")
    
    def bc_x2(self) -> BoundaryCondition:
        return ("periodic", "periodic")