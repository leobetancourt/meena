from dataclasses import dataclass
from functools import partial

from jax.typing import ArrayLike
from jax import Array, jit
import jax.numpy as jnp

from meena import Hydro, Lattice, Primitives, Conservatives, BoundaryCondition
from src.common.helpers import cartesian_to_polar


@partial(jit, static_argnames=["hydro", "lattice"])
def get_accr_rate(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dA = lattice.dX1 * lattice.dX2
    x1, x2 = hydro.get_BH_position(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x1, x2)
    m_dot = (sink_source[..., 0] * dA)
    return jnp.sum(m_dot)

def get_eccentricity(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    x, y = lattice.X1, lattice.X2
    r = jnp.sqrt(x ** 2 + y ** 2)

    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    j = x * v - y * u
    e_x = (j * v / (hydro.G * hydro.M)) - (x / r)
    e_y = -(j * u / (hydro.G * hydro.M)) - (y / r)
    dA = lattice.dX1 * lattice.dX2

    bounds = jnp.logical_and(r >= hydro.a, r <= 6 * hydro.a)
    ec_x = jnp.where(bounds, e_x * rho * dA, 0).sum() / \
        (35 * jnp.pi * hydro.Sigma_0 * (hydro.a ** 2))
    ec_y = jnp.where(bounds, e_y * rho * dA, 0).sum() / \
        (35 * jnp.pi * hydro.Sigma_0 * (hydro.a ** 2))
    return ec_x, ec_y

@partial(jit, static_argnames=["hydro", "lattice"])
def get_eccentricity_x(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    return get_eccentricity(hydro, lattice, U, flux, t)[0]


@partial(jit, static_argnames=["hydro", "lattice"])
def get_eccentricity_y(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    return get_eccentricity(hydro, lattice, U, flux, t)[1]


@dataclass(frozen=True)
class SingleBH(Hydro):
    G: float = 1
    M: float = 1
    mach: float = 10
    a: float = 1
    m: float = 1
    eps: float = 0.05 * a
    omega_B: float = 1
    R_0: float = 1 * a
    sigma: float = 0.1 * a
    nu_vis: float = 1e-3 * (a ** 2) * omega_B
    cfl_num: float = 0.1
    size: float = 5
    res: int = 500
    retrograde: bool = 0
    
    sink_rate: float = 10 * omega_B
    sink_prescription: str = "acceleration-free"


    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        r, theta = cartesian_to_polar(X1, X2)

        # surface density
        A = self.m / (2 * jnp.pi * self.R_0 * jnp.sqrt(2 * jnp.pi * self.sigma**2))
        gaussian = A * jnp.exp(- ((r - self.R_0) ** 2) / (2 * (self.sigma ** 2)))
        
        floor = 1e-3
        rho = jnp.maximum(floor, gaussian)

        v_r = jnp.zeros_like(rho)
        v_theta = jnp.sqrt(self.G * self.M / r) # keplerian velocity
        if self.retrograde:
            v_theta *= -1
        u = v_r * jnp.cos(theta) - v_theta * jnp.sin(theta)
        v = v_r * jnp.sin(theta) + v_theta * jnp.cos(theta)
        
        p = jnp.ones_like(X1) * 1

        return jnp.array([
            rho,
            u,
            v,
            p
        ]).transpose((1, 2, 0))

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-self.size/2, self.size/2), (-self.size/2, self.size/2))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)

    def t_end(self) -> float:
        return (0.002 / (12 * self.nu_vis * self.R_0**-2)) * 513
    
    def save_interval(self):
        return 0.002 / (12 * self.nu_vis * self.R_0**-2)
    
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
        dx, dy = x - x_bh, y - y_bh
        r = jnp.sqrt(dx ** 2 + dy ** 2)
        r_sink = self.eps
        s_rate = jnp.where(r < (4 * r_sink), self.sink_rate * jnp.exp(-jnp.pow(r / r_sink, 4)), jnp.zeros_like(r))
        mdot = jnp.where(s_rate > 0, -s_rate * rho, -s_rate)
        
        S = jnp.zeros_like(U)
        u, v = U[..., 1] / rho, U[..., 2] / rho
          
        if self.sink_prescription == "acceleration-free":
            S = S.at[..., 0].set(mdot)
            S = S.at[..., 1].set(mdot * u)
            S = S.at[..., 2].set(mdot * v)
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
        r, _ = cartesian_to_polar(X1, X2)

        # f(r) increases linearly from 0 at r = D - 0.1a to 1000 at r = D (and otherwise 0)
        def f(r):
            linear = (r - (D - 0.1 * self.a)) * (1000 / (0.1 * self.a))
            return jnp.where((r >= D - 0.1 * self.a) & (r <= D), linear, 0)
        
        prims_0 = self.initialize(X1, X2)
        rho_0, u_0, v_0 = prims_0[..., 0], prims_0[..., 1], prims_0[..., 2]
        E_0 = self.E(prims_0, X1, X2, t)
        U_0 = jnp.array([
            rho_0,
            rho_0 * u_0,
            rho_0 * v_0,
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