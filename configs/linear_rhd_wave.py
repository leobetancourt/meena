from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from meena import Hydro, BoundaryCondition, Primitives, Conservatives


def newton_raphson_T(
    rho, P_total, a_rad, *,
    k_B=1.3807e-16,
    m_p=1.6726e-24,
    mu=1/2,
    max_iter=20,
    tol=1e-6
):
    # Initial guess: ideal gas only
    T_init = (P_total / rho) * (mu * m_p / k_B)

    def f(T, rho, P):
        return rho * (k_B / (mu * m_p)) * T + (1/3) * a_rad * T**4 - P

    def df(T, rho):
        return rho * (k_B / (mu * m_p)) + (4/3) * a_rad * T**3

    def body_fn(state):
        T, i, converged = state
        f_val = f(T, rho, P_total)
        df_val = df(T, rho)
        T_new = T - f_val / df_val

        # Convergence check
        err = jnp.abs(T_new - T) / (T + 1e-10)
        new_converged = jnp.logical_or(converged, err < tol)
        return (jnp.where(new_converged, T, T_new), i + 1, new_converged)

    def cond_fn(state):
        _, i, converged = state
        return jnp.logical_and(i < max_iter, jnp.any(~converged))

    # Run Newton iterations
    T_final, _, _ = lax.while_loop(
        cond_fn,
        body_fn,
        (T_init, 0, jnp.zeros_like(T_init, dtype=bool))
    )

    return T_final

@dataclass(frozen=True)
class LinearRHDWave(Hydro):
    L: float = 1
    sigma_B: float = 5.670374419e-5
    a_rad: float = 4 * sigma_B
    
    def initialize(self, X1: ArrayLike) -> Array:
        x = X1
        prims_0 = jnp.array([jnp.ones_like(x), jnp.zeros_like(x), jnp.ones_like(x), 0.1 * jnp.ones_like(x)]).T
        c_s = self.c_s(prims_0, X1, 0)
        delta = 1e-2 * c_s
        
        density = jnp.ones_like(x) + delta * jnp.cos(2 * jnp.pi * x / self.L)
        velocity = delta * c_s * jnp.sin(2 * jnp.pi * x / self.L)
        pressure = jnp.ones_like(x) + delta * self.gamma() * jnp.cos(2 * jnp.pi * x / self.L)
        p_rad = 0 * jnp.ones_like(x)
        
        return jnp.array([density, velocity, pressure, p_rad]).T
    
    def dim(self) -> int:
        return 1
    
    def radiation(self) -> bool:
        return True
    
    def c(self) -> float:
        return 1
    
    def kappa(self) -> float:
        return 1000
    
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

    def source(self, U: ArrayLike, lattice, t: float) -> Array:
        rho = U[..., 0]
        p = self.P(U, lattice.X1, t)
        e_rad = U[..., 3]
        p_tot = p + (1/3) * e_rad
        
        T = newton_raphson_T(rho, p_tot, self.a_rad)
        zero = jnp.zeros_like(rho)
        
        return jnp.array([
            zero,
            zero,
            -self.c() * self.kappa() * (self.a_rad * T**4 - e_rad),
            self.c() * self.kappa() * (self.a_rad * T**4 - e_rad),
        ]).transpose((1, 0))
    
    def cfl(self) -> float:
        return 0.01
    
    def t_end(self) -> float:
        return 20
    
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
        return 0.05
        
    def range(self) -> tuple[float, float]:
        return (0, self.L)
        
    def resolution(self) -> int:
        return 200
    
    def bc_x1(self) -> BoundaryCondition:
        return ("reflective", "reflective")