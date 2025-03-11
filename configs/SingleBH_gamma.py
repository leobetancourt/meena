from dataclasses import dataclass
from functools import partial

from jax.typing import ArrayLike
from jax import Array, jit
import jax.numpy as jnp

from scipy.special import iv

from meena import Hydro, Lattice, Primitives, Conservatives, BoundaryCondition

from src.common.helpers import cartesian_to_polar


@partial(jit, static_argnames=["hydro", "lattice"])
def get_accr_rate(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dA = lattice.dX1 * lattice.dX2
    x1, x2 = hydro.get_BH_position(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x1, x2)
    m_dot = (sink_source[..., 0] * dA)
    return jnp.sum(m_dot)

def pringle(x, tau, m, R0):
    # Clip tau to avoid numerical issues (ensures it's not too small)
    tau = jnp.clip(tau, 1e-6, None)
    
    # Clip x to prevent extreme values in exponentiation and power functions
    x = jnp.clip(x, 1e-6, None)
    
    # Compute the argument for the Bessel function with clipping
    iv_arg = jnp.clip(2 * x / tau, 1e-6, 1e2)
    
    # Use asymptotic expansion for large arguments of iv(0.25, z)
    iv_val = jnp.where(
        iv_arg > 10,  # Use asymptotic form for large arguments
        jnp.exp(iv_arg) / jnp.sqrt(2 * jnp.pi * iv_arg),
        iv(0.25, iv_arg)  # Otherwise, use the standard Bessel function
    )
    
    # Compute the exponent term, ensuring stability
    exp_term = jnp.exp(-jnp.clip((1 + x**2) / tau, 1e-6, 1e2))
    
    # Compute the full function
    result = (m / (jnp.pi * R0**2)) * (tau**-1) * (x**-0.25) * exp_term * iv_val
    
    return result

@dataclass(frozen=True)
class SingleBH(Hydro):
    G: float = 1
    M: float = 1
    mach: float = 10
    a: float = 1
    omega_B: float = 1
    nu_vis: float = 1e-3 * (a ** 2) * omega_B
    sink_rate: float = 10 * omega_B
    sink_prescription: str = "acceleration-free"
    m: float = 1
    Sigma_floor: float = 1e-4
    R_0: float = 2 * a
    sigma: float = 0.1 * a
    
    size: float = 10
    res: int = 1000
    eps: float = 0.05 * a
    cfl_num: float = 0.1

    retrograde: bool = 0
    
    tau_0: float = 0.032
    
    def tau_to_t(self, tau: float) -> float:
        return tau / (12 * self.nu_vis * self.R_0**-2)
    
    def t_to_tau(self, t: float) -> float:
        return 12 * t * self.nu_vis * self.R_0**-2

    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        r, theta = cartesian_to_polar(X1, X2)

        # surface density
        Sigma_pringle = jnp.nan_to_num(pringle(r / self.R_0, self.tau_0, self.m, self.R_0), nan=self.Sigma_floor)
        Sigma = jnp.maximum(self.Sigma_floor, Sigma_pringle)
        Sigma = jnp.where(Sigma > 10, self.Sigma_floor, Sigma)

        v_r = jnp.zeros_like(X1)
        v_theta = jnp.sqrt(self.G * self.M / r) # keplerian velocity
        if self.retrograde:
            v_theta *= -1
        u = v_r * jnp.cos(theta) - v_theta * jnp.sin(theta)
        v = v_r * jnp.sin(theta) + v_theta * jnp.cos(theta)
        
        h = 0.1
        p = jnp.ones_like(X1) * 1e-2 # h**2 * v_theta**2 * Sigma / self.gamma()
        return jnp.array([
            Sigma,
            u,
            v,
            p
        ]).transpose((1, 2, 0))

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-self.size/2, self.size/2), (-self.size/2, self.size/2))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)
    
    def t_start(self) -> float:
        return self.tau_to_t(self.tau_0)

    def t_end(self) -> float:
        return self.tau_to_t(0.002) * 1025
    
    def save_interval(self):
        return 0.2

    def PLM(self) -> bool:
        return True
    
    def theta_PLM(self) -> float:
        return 1.5
    
    def time_order(self) -> int:
        return 1
    
    def cfl(self) -> float:
        return self.cfl_num

    def coords(self) -> str:
        return "cartesian"

    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def nu(self) -> float:
        return self.nu_vis
    
    def gamma(self) -> float:
        return 5 / 3
    
    def get_BH_position(self, t):
        return 0, 0

    def BH_gravity(self, U, x, y, x_bh, y_bh):
        dx, dy = x - x_bh, y - y_bh
        r = jnp.sqrt(dx ** 2 + dy ** 2)
        
        g_acc = - self.G * self.M / (r ** 2 + self.eps ** 2)
        g_x, g_y = g_acc * dx / (r + self.eps), g_acc * dy / (r + self.eps)
        rho = U[..., 0]
        u, v = U[..., 1] / rho, U[..., 2] / rho

        return jnp.array([
            jnp.zeros_like(rho),
            rho * g_x,
            rho * g_y,
            rho * (u * g_x + v * g_y)
        ]).transpose((1, 2, 0))

    def BH_sink(self, U, x, y, x_bh, y_bh):
        rho = U[..., 0]
        pres = self.P(U)
        eps = pres / rho / (self.gamma() - 1)
        dx, dy = x - x_bh, y - y_bh
        r = jnp.sqrt(dx ** 2 + dy ** 2)
        r_sink = self.eps
        r_soft = self.eps
        s_rate = jnp.where(r < (4 * r_sink), self.sink_rate * jnp.exp(-jnp.pow(r / r_sink, 4)), jnp.zeros_like(r))
        
        fgrav_num = rho * self.M * jnp.pow(r**2 + r_soft**2, -1.5)
        fx, fy = -fgrav_num * dx, -fgrav_num * dy
        mdot = rho * s_rate * -1
        
        S = jnp.zeros_like(U)
        u, v = U[..., 1] / rho, U[..., 2] / rho
          
        if self.sink_prescription == "acceleration-free":
            S = S.at[..., 0].set(mdot)
            S = S.at[..., 1].set(mdot * u)
            S = S.at[..., 2].set(mdot * v)
            S = S.at[..., 3].set(mdot * eps + 0.5 * mdot * (u*u + v*v) + (fx * u + fy * v))
        elif self.sink_prescription == "torque-free":
            u_bh, v_bh = 0, 0
            rhatx = dx / (r + 1e-12)
            rhaty = dy / (r + 1e-12)
            dvdotrhat = (u - u_bh) * rhatx + (v - v_bh) * rhaty
            ustar = dvdotrhat * rhatx + u_bh
            vstar = dvdotrhat * rhaty + v_bh
            S = S.at[..., 0].set(mdot)
            S = S.at[..., 1].set(mdot * ustar)
            S = S.at[..., 2].set(mdot * vstar)
            S = S.at[..., 3].set((mdot * eps + 0.5 * mdot * (ustar * ustar + vstar * vstar)) + (fx * u + fy * v))
        else:
            raise AttributeError("Invalid sink prescription")

        return S

    def source(self, U: ArrayLike, X1, X2, t: float) -> Array:
        S = jnp.zeros_like(U)
        x1, x2 = self.get_BH_position(t)
 
        # gravity
        S += self.BH_gravity(U, X1, X2, x1, x2)

        # sink
        S += self.BH_sink(U, X1, X2, x1, x2)
        
        # buffer
        S += self.buffer(U, X1, X2, t)
        
        return S

    # Buffer implementation adapted from Westernacher-Schneider et al. 2022
    def buffer(self, U: ArrayLike, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        D = self.size / 2
        r, theta = cartesian_to_polar(X1, X2)

        # f(r) increases linearly from 0 at r = D - 0.1a to 1000 at r = D (and otherwise 0)
        def f(r):
            linear = (r - (D - 0.1 * self.a)) * (1000 / (0.1 * self.a))
            return jnp.where((r >= D - 0.1 * self.a) & (r <= D), linear, 0)
        
        Sigma_0 = jnp.ones_like(X1) * self.Sigma_floor
        v_r = jnp.zeros_like(X1)
        v_theta = jnp.sqrt(self.G * self.M / r) # keplerian velocity
        u_0 = v_r * jnp.cos(theta) - v_theta * jnp.sin(theta)
        v_0 = v_r * jnp.sin(theta) + v_theta * jnp.cos(theta)
        p_0 = jnp.ones_like(X1) * 1e-2
        prims = jnp.array([
            Sigma_0,
            u_0,
            v_0,
            p_0
        ]).transpose((1, 2, 0))
        E_0 = self.E(prims, X1, X2, t)
        U_0 = jnp.array([
            Sigma_0,
            Sigma_0 * u_0,
            Sigma_0 * v_0,
            E_0
        ]).transpose((1, 2, 0))
        omega_naught = jnp.sqrt((self.G * self.M / (D ** 3 + self.eps ** 3))
                                * (1 - (1 / (self.mach ** 2))))
        omega_D = ((omega_naught ** -4) + (self.omega_B ** -4)) ** (-1/4)
        
        return - f(r)[..., jnp.newaxis] * omega_D * (U - U_0)

    def diagnostics(self):
        diagnostics = []
        diagnostics.append(("m_dot", get_accr_rate))
        return diagnostics