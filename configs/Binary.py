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
    x1_1, x2_1, x1_2, x2_2 = hydro.get_positions(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x1_1, x2_1) + \
            hydro.BH_sink(U, lattice.X1, lattice.X2, x1_2, x2_2)
    m_dot = (sink_source[..., 0] * dA)
    return jnp.sum(m_dot)


@partial(jit, static_argnames=["hydro", "lattice"])
def get_torque1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    x1_1, x2_1, x1_2, x2_2 = hydro.get_positions(t)
    rho = U[..., 0]
    x_bh, y_bh = x1_1, x2_1
    dA = lattice.dX1 * lattice.dX2

    r, theta = cartesian_to_polar(lattice.X1, lattice.X2)
    r_bh, theta_bh = jnp.sqrt(x_bh ** 2 + y_bh ** 2), jnp.arctan2(y_bh, x_bh)
    delta_theta = theta - theta_bh
    dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                    r_bh * jnp.cos(delta_theta))
    Fg = hydro.G * (hydro.M / 2) * rho * dA / \
        (dist ** 2 + hydro.eps ** 2)
    Fg_theta = Fg * (r * jnp.sin(delta_theta)) / dist
    T = jnp.sum((hydro.a / 2) * Fg_theta)
    return T


@partial(jit, static_argnames=["hydro", "lattice"])
def get_torque2(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    x1_1, x2_1, x1_2, x2_2 = hydro.get_positions(t)
    rho = U[..., 0]
    x_bh, y_bh = x1_2, x2_2
    dA = lattice.dX1 * lattice.dX2

    r, theta = cartesian_to_polar(lattice.X1, lattice.X2)
    r_bh, theta_bh = jnp.sqrt(x_bh ** 2 + y_bh ** 2), jnp.arctan2(y_bh, x_bh)
    delta_theta = theta - theta_bh
    dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                    r_bh * jnp.cos(delta_theta))
    Fg = hydro.G * (hydro.M / 2) * rho * dA / \
        (dist ** 2 + hydro.eps ** 2)
    Fg_theta = Fg * (r * jnp.sin(delta_theta)) / dist
    T = jnp.sum((hydro.a / 2) * Fg_theta)
    return T


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
class Binary(Hydro):
    G: float = 1
    M: float = 1
    mach: float = 10
    a: float = 1
    R_cav: float = 2.5 * a
    R_out: float = 10 * a
    delta_0: float = 1e-5
    Sigma_0: float = 1
    eps: float = 0.05 * a
    omega_B: float = 1
    t_sink: float = 1 / (10 * omega_B)
    cfl_num: float = 0.3
    size: float = 20
    res: float = 2000
    plm: float = 1.8

    def initialize(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        t = 0
        r, theta = cartesian_to_polar(X1, X2)

        def f(r):
            return 1 # - 1 / (1 + jnp.exp(-2 * (r - self.R_out) / self.a))
        # surface density
        rho = self.Sigma_0 * \
            ((1 - self.delta_0) * jnp.exp(-(self.R_cav /
                                            (r + self.eps)) ** 12) + self.delta_0) * f(r)

        # radial velocity 'kick'
        v_naught = 1e-4 * self.omega_B * self.a
        v_r = v_naught * jnp.sin(theta) * (r / self.a) * \
            jnp.exp(-(r / (3.5 * self.a)) ** 6)
        # angular frequency of gas in the disk
        omega_naught = jnp.sqrt((self.G * self.M / (r ** 3))
                                * (1 - (1 / (self.mach ** 2))))
        omega = ((omega_naught ** -4) + (self.omega_B ** -4)) ** (-1/4)
        v_theta = omega * r
        
        u = v_r * jnp.cos(theta) - v_theta * jnp.sin(theta)
        v = v_r * jnp.sin(theta) + v_theta * jnp.cos(theta)

        return jnp.array([
            rho,
            rho * u,
            rho * v,
            self.E((rho, u, v, jnp.zeros_like(rho)), X1, X2, t)
        ]).transpose((1, 2, 0))

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-self.size/2, self.size/2), (-self.size/2, self.size/2))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)

    def t_end(self) -> float:
        return 800 * 2 * jnp.pi
    
    def PLM(self) -> bool:
        return True
    
    def theta_PLM(self) -> float:
        return self.plm
    
    def time_order(self) -> int:
        return 2
    
    def cfl(self) -> float:
        return self.cfl_num

    def coords(self) -> str:
        return "cartesian"

    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def nu(self) -> float:
        return 1e-3 * (self.a ** 2) * self.omega_B

    def E(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho, u, v, p = prims
        e_internal = rho * (self.c_s(prims, X1, X2, t) ** 2)
        e_kinetic = 0.5 * rho * (u ** 2 + v ** 2)
        return e_internal + e_kinetic

    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        x1_1, x2_1, x1_2, x2_2 = self.get_positions(t)
        r, theta = cartesian_to_polar(X1, X2)
        r1, theta1 = cartesian_to_polar(x1_1, x2_1)
        r2, theta2 = cartesian_to_polar(x1_2, x2_2)

        dist1 = jnp.sqrt(r ** 2 + r1 ** 2 - 2 * r *
                         r1 * jnp.cos(theta - theta1))
        dist2 = jnp.sqrt(r ** 2 + r2 ** 2 - 2 * r *
                         r2 * jnp.cos(theta - theta2))

        cs = jnp.sqrt(((self.G * (self.M / 2) / jnp.sqrt(dist1 ** 2 + self.eps ** 2)) +
                       (self.G * (self.M / 2) / jnp.sqrt(dist2 ** 2 + self.eps ** 2))) / (self.mach ** 2))
        return cs

    def P(self, cons: Conservatives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho, _, _, _ = cons
        return rho * self.c_s(cons, X1, X2, t) ** 2
    
    def get_positions(self, t):
        delta = jnp.pi
        x1_1, x2_1 = (self.a / 2) * jnp.cos(self.omega_B *
                                            t), (self.a / 2) * jnp.sin(self.omega_B * t)
        x1_2, x2_2 = (self.a / 2) * jnp.cos(self.omega_B * t +
                                            delta), (self.a / 2) * jnp.sin(self.omega_B * t + delta)
        return x1_1, x2_1, x1_2, x2_2

    def BH_gravity(self, U, x, y, x_bh, y_bh):
        dx, dy = x - x_bh, y - y_bh
        r = jnp.sqrt(dx ** 2 + dy ** 2)
        
        
        g_acc = - self.G * (self.M / 2) / (r ** 2 + self.eps ** 2)
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
        sink = jnp.exp(-((r / r_sink) ** 6)) * (self.t_sink ** -1) * rho
        S = jnp.zeros_like(U).at[:, :, 0].set(-sink)

        return S

    def source(self, U: ArrayLike, X1, X2, t: float) -> Array:
        S = jnp.zeros_like(U)
        x1_1, x2_1, x1_2, x2_2 = self.get_positions(t)
 
        # gravity
        S += self.BH_gravity(U, X1, X2, x1_1, x2_1) + \
            self.BH_gravity(U, X1, X2, x1_2, x2_2)

        # sinks
        S += self.BH_sink(U, X1, X2, x1_1, x2_1) + self.BH_sink(U, X1, X2, x1_2, x2_2)
        
        # buffer
        S += self.buffer(U, X1, X2)
        
        return S

    # Buffer implementation adapted from Westernacher-Schneider et al. 2022
    def buffer(self, U: ArrayLike, X1, X2) -> Array:
        D = self.size / 2
        r, _ = cartesian_to_polar(X1, X2)

        # f(r) increases linearly from 0 at r = D - 0.1a to 1000 at r = D (and otherwise 0)
        def f(r):
            linear = (r - (D - 0.1 * self.a)) * (1000 / (0.1 * self.a))
            return jnp.where((r >= D - 0.1 * self.a) & (r <= D), linear, 0)
        
        U_0 = self.initialize(X1, X2)
        omega_naught = jnp.sqrt((self.G * self.M / (D ** 3 + self.eps ** 3))
                                * (1 - (1 / (self.mach ** 2))))
        omega_D = ((omega_naught ** -4) + (self.omega_B ** -4)) ** (-1/4)
        
        return - f(r)[..., jnp.newaxis] * omega_D * (U - U_0)

    def diagnostics(self):
        diagnostics = []
        diagnostics.append(("m_dot", get_accr_rate))
        diagnostics.append(("torque_1", get_torque1))
        diagnostics.append(("torque_2", get_torque2))
        diagnostics.append(("e_x", get_eccentricity_x))
        diagnostics.append(("e_y", get_eccentricity_y))
        return diagnostics

    def save_interval(self):
        return 1