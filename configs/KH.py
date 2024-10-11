from dataclasses import dataclass

import os
import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from hydrocode import Hydro, BoundaryCondition

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
@dataclass(frozen=True)
class KH(Hydro):
    res: int = 1000
    
    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        x, y = X1, X2
        # w0 = 0.1
        # sigma = 0.05 / jnp.sqrt(2)
        # v_pert = w0*jnp.sin(4*jnp.pi*x) * (jnp.exp(-(y-0.25)**2 /
        #                                            (2 * sigma**2)) + jnp.exp(-(y-0.75)**2/(2*sigma**2)))

        rho = jnp.zeros_like(x)
        u = jnp.zeros_like(x)
        v = jnp.ones_like(x)
        p = jnp.ones_like(x) * 2.5

        rho = jnp.where((jnp.abs(y) > 0.25), 1, rho)
        u = jnp.where((jnp.abs(y) > 0.25), 0.5, u)
        
        rho = jnp.where((jnp.abs(y) <= 0.25), 2, rho)
        u = jnp.where((jnp.abs(y) <= 0.25), -0.5, u)
        
        rng = np.random.default_rng(1000)
        sin_pert = 0.01 * jnp.sin(2 * jnp.pi * x[:, 0])
        u_rand = rng.choice(sin_pert, size=u.shape)
        v_rand = rng.choice(sin_pert, size=v.shape)

        u += u_rand
        v += v_rand

        return jnp.array([
            rho,
            rho * u,
            rho * v,
            self.E((rho, u, v, p))
        ]).transpose((1, 2, 0))

    def gamma(self) -> float:
        return 5.0 / 3.0

    def t_end(self) -> float:
        return 30

    def solver(self) -> str:
        return "hllc"

    def PLM(self) -> float:
        return False

    def save_interval(self) -> float:
        return 0.05

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-0.5, 0.5), (-0.5, 0.5))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)

    def bc_x1(self) -> BoundaryCondition:
        return ("periodic", "periodic")

    def bc_x2(self) -> BoundaryCondition:
        return ("periodic", "periodic")
