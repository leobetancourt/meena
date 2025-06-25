from dataclasses import dataclass
from functools import partial

from jax.typing import ArrayLike
from jax import Array, jit
import jax.numpy as jnp

from meena import Hydro, Lattice, Primitives, Conservatives, BoundaryCondition
from src.common.helpers import cartesian_to_polar

@partial(jit, static_argnames=["hydro", "lattice"])
def get_accr_rate(hydro: Hydro, lattice: Lattice, prims: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dA = lattice.dX1 * lattice.dX2
    x1_1, x2_1, x1_2, x2_2 = hydro.get_bh_positions(t)
    u_1, v_1, u_2, v_2 = hydro.get_bh_velocities(t)
    
    sink_source = hydro.BH_sink(prims, lattice.X1, lattice.X2, x1_1, x2_1, u_1, v_1) + \
            hydro.BH_sink(prims, lattice.X1, lattice.X2, x1_2, x2_2, u_2, v_2)
    m_dot = (sink_source[..., 0] * dA)
    return jnp.sum(m_dot)


@partial(jit, static_argnames=["hydro", "lattice"])
def get_torque1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    x1_1, x2_1, x1_2, x2_2 = hydro.get_bh_positions(t)
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
    x1_1, x2_1, x1_2, x2_2 = hydro.get_bh_positions(t)
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
    sink_rate: float = 10 * omega_B
    sink_prescription: str = "torque-free"
    T: float = 1.1
    
    cadence: float = 0.5
    cfl_num: float = 0.4
    size: float = 10
    res: float = 1000
    plm: bool = 1
    plm_theta: float = 1.5
    t_order: int = 2
    buff: bool = 1

    def initialize(self, lattice) -> Array:
        X1, X2 = lattice.X1, lattice.X2
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
            u,
            v,
            rho * (self.c_s(None, X1, X2, 0) ** 2)
        ]).transpose((1, 2, 0))

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-self.size/2, self.size/2), (-self.size/2, self.size/2))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)

    def t_end(self) -> float:
        return self.T * 2 * jnp.pi
    
    def PLM(self) -> bool:
        return self.plm
    
    def theta_PLM(self) -> float:
        return self.plm_theta
    
    def time_order(self) -> int:
        return self.t_order
    
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
        rho, u, v = prims[..., 0], prims[..., 1], prims[..., 2]
        e_internal = rho * (self.c_s(prims, X1, X2, t) ** 2)
        e_kinetic = 0.5 * rho * (u ** 2 + v ** 2)
        return e_internal + e_kinetic

    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        x1_1, x2_1, x1_2, x2_2 = self.get_bh_positions(t)
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
        rho = cons[..., 0]
        return rho * (self.c_s(cons, X1, X2, t) ** 2)
    
    def get_bh_positions(self, t):
        delta = jnp.pi
        x1_1, x2_1 = (self.a / 2) * jnp.cos(self.omega_B *
                                            t), (self.a / 2) * jnp.sin(self.omega_B * t)
        x1_2, x2_2 = (self.a / 2) * jnp.cos(self.omega_B * t +
                                            delta), (self.a / 2) * jnp.sin(self.omega_B * t + delta)
        return x1_1, x2_1, x1_2, x2_2
    
    def get_bh_velocities(self, t):
        delta = jnp.pi
        u_1, v_1 = -self.omega_B * (self.a / 2) * jnp.sin(self.omega_B * t), self.omega_B * (self.a / 2) * jnp.cos(self.omega_B * t)
        u_2, v_2 = -self.omega_B * (self.a / 2) * jnp.sin(self.omega_B * t + delta), self.omega_B * (self.a / 2) * jnp.cos(self.omega_B * t + delta)
        
        return u_1, v_1, u_2, v_2

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

    # Sink prescription in Sailfish code
    def BH_sink(self, U, x, y, x_bh, y_bh, u_bh, v_bh):
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
        x1_1, x2_1, x1_2, x2_2 = self.get_bh_positions(t)
        u_1, v_1, u_2, v_2 = self.get_bh_velocities(t)
        
        # gravity
        S += self.BH_gravity(U, X1, X2, x1_1, x2_1) + \
            self.BH_gravity(U, X1, X2, x1_2, x2_2)

        # sinks
        S += self.BH_sink(U, X1, X2, x1_1, x2_1, u_1, v_1) + self.BH_sink(U, X1, X2, x1_2, x2_2, u_2, v_2)
        
        # buffer
        if self.buff:
            S += self.buffer(U, X1, X2, t)
        
        return S

    # Buffer implementation adapted from Sailfish
    def buffer(self, U: ArrayLike, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        x, y = X1, X2
        r, _ = cartesian_to_polar(x, y)

        surface_density = jnp.ones_like(x) * self.delta_0
        driving_rate = 100
        outer_radius = self.size / 2
        onset_width = 1
        onset_radius = outer_radius - onset_width
        
        v_kep = jnp.sqrt(self.G * self.M / r)
        u, v = (-y / r) * v_kep, (x / r) * v_kep
        px = surface_density * u
        py = surface_density * v
        prims = jnp.array([
            surface_density,
            px,
            py,
            jnp.zeros_like(x)
        ]).transpose((1, 2, 0))
        U_0 = jnp.array([
            surface_density,
            px,
            py,
            self.E(prims, X1, X2, t)
        ]).transpose((1, 2, 0))
        omega_outer = jnp.sqrt(self.M * jnp.pow(onset_radius, -3.0))
        buffer_rate = driving_rate * omega_outer * (r - onset_radius) / (outer_radius - onset_radius)
        buffer_rate = jnp.where(r > onset_radius, buffer_rate, 0)
        
        return -(U - U_0) * buffer_rate[..., jnp.newaxis]

    def diagnostics(self):
        diagnostics = []
        diagnostics.append(("m_dot", get_accr_rate))
        diagnostics.append(("torque_1", get_torque1))
        diagnostics.append(("torque_2", get_torque2))
        diagnostics.append(("e_x", get_eccentricity_x))
        diagnostics.append(("e_y", get_eccentricity_y))
        return diagnostics

    def save_interval(self):
        return self.cadence * 2 * jnp.pi