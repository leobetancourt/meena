from dataclasses import dataclass
from functools import partial

from jax.typing import ArrayLike
from jax import Array, jit
import jax.numpy as jnp

from meena import Hydro, Lattice, Primitives, Conservatives, BoundaryCondition
from src.common.helpers import cartesian_to_polar


@partial(jit, static_argnames=["hydro", "lattice"])
def get_m_dot_1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dA = lattice.dX1 * lattice.dX2
    x, y, _, _ = hydro.get_bh_positions(t)
    u, v, _, _ = hydro.get_bh_velocities(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x, y, u, v)
    m_dot = (-sink_source[..., 0] * dA)
    return jnp.sum(m_dot)

@partial(jit, static_argnames=["hydro", "lattice"])
def get_m_dot_2(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dA = lattice.dX1 * lattice.dX2
    _, _, x, y = hydro.get_bh_positions(t)
    _, _, u, v = hydro.get_bh_velocities(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x, y, u, v)
    m_dot = (-sink_source[..., 0] * dA)
    return jnp.sum(m_dot)

@partial(jit, static_argnames=["hydro", "lattice"])
def get_L_dot_1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dA = lattice.dX1 * lattice.dX2
    x, y, _, _ = hydro.get_bh_positions(t)
    u, v, _, _ = hydro.get_bh_velocities(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x, y, u, v)
    m_dot_density = -sink_source[..., 0]  # mass accretion rate density
    
    rho = U[..., 0]
    v_x = U[..., 1] / rho
    v_y = U[..., 2] / rho
    dx = lattice.X1 - x
    dy = lattice.X2 - y

    specific_L = dx * v_y - dy * v_x
    L_dot_density = specific_L * m_dot_density
    L_dot = jnp.sum(L_dot_density * dA)
    
    return L_dot

@partial(jit, static_argnames=["hydro", "lattice"])
def get_L_dot_2(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dA = lattice.dX1 * lattice.dX2
    _, _, x, y = hydro.get_bh_positions(t)
    _, _, u, v = hydro.get_bh_velocities(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x, y, u, v)
    m_dot_density = -sink_source[..., 0]  # mass accretion rate density
    
    rho = U[..., 0]
    v_x = U[..., 1] / rho
    v_y = U[..., 2] / rho
    dx = lattice.X1 - x
    dy = lattice.X2 - y

    specific_L = dx * v_y - dy * v_x
    L_dot_density = specific_L * m_dot_density
    L_dot = jnp.sum(L_dot_density * dA)
    
    return L_dot

@partial(jit, static_argnames=["hydro", "lattice"])
def get_torque_1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    x1_1, x2_1, x1_2, x2_2 = hydro.get_bh_positions(t)
    rho = U[..., 0]
    x_bh, y_bh = x1_1, x2_1
    dA = lattice.dX1 * lattice.dX2

    r, theta = cartesian_to_polar(lattice.X1, lattice.X2)
    r_bh, theta_bh = jnp.sqrt(x_bh ** 2 + y_bh ** 2), jnp.arctan2(y_bh, x_bh)
    delta_theta = theta - theta_bh
    dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                    r_bh * jnp.cos(delta_theta))
    Fg = hydro.G() * (hydro.M / 2) * rho * dA / \
        (dist ** 2 + hydro.eps ** 2)
    Fg_theta = Fg * (r * jnp.sin(delta_theta)) / dist
    T = jnp.sum((hydro.a / 2) * Fg_theta)
    return T


@partial(jit, static_argnames=["hydro", "lattice"])
def get_torque_2(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    x1_1, x2_1, x1_2, x2_2 = hydro.get_bh_positions(t)
    rho = U[..., 0]
    x_bh, y_bh = x1_2, x2_2
    dA = lattice.dX1 * lattice.dX2

    r, theta = cartesian_to_polar(lattice.X1, lattice.X2)
    r_bh, theta_bh = jnp.sqrt(x_bh ** 2 + y_bh ** 2), jnp.arctan2(y_bh, x_bh)
    delta_theta = theta - theta_bh
    dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                    r_bh * jnp.cos(delta_theta))
    Fg = hydro.G() * (hydro.M / 2) * rho * dA / \
        (dist ** 2 + hydro.eps ** 2)
    Fg_theta = Fg * (r * jnp.sin(delta_theta)) / dist
    T = jnp.sum((hydro.a / 2) * Fg_theta)
    return T

# cavity eccentricity calculation from Duffel et al. 2024
def get_cavity_eccentricity(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    x, y = lattice.X1, lattice.X2
    r = jnp.sqrt(x ** 2 + y ** 2)

    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    j = x * v - y * u
    e_x = (j * v / (hydro.G() * hydro.M)) - (x / r)
    e_y = -(j * u / (hydro.G() * hydro.M)) - (y / r)
    dA = lattice.dX1 * lattice.dX2

    bounds = jnp.logical_and(r >= hydro.a, r <= 6 * hydro.a)
    # note: i removed the normalization constant for now
    ec_x = jnp.where(bounds, e_x * rho * dA, 0).sum()
    ec_y = jnp.where(bounds, e_y * rho * dA, 0).sum() 
    return ec_x, ec_y


@partial(jit, static_argnames=["hydro", "lattice"])
def get_cavity_eccentricity_x(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    return get_cavity_eccentricity(hydro, lattice, U, flux, t)[0]


@partial(jit, static_argnames=["hydro", "lattice"])
def get_cavity_eccentricity_y(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    return get_cavity_eccentricity(hydro, lattice, U, flux, t)[1]

# work on binary components
@partial(jit, static_argnames=["hydro", "lattice"])
def get_E_dot_1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    rho = U[..., 0]
    dA = lattice.dX1 * lattice.dX2
    
    # BH position and velocity
    x, y, _, _ = hydro.get_bh_positions(t)
    u, v, _, _ = hydro.get_bh_velocities(t)

    # gravitational force by gas on BH
    dx = lattice.X1 - x
    dy = lattice.X2 - y
    dist = jnp.sqrt(dx**2 + dy**2 + hydro.eps**2)
    Fg_mag = hydro.G() * (hydro.M / 2) * rho * dA / dist**2
    Fg_x = Fg_mag * dx / dist
    Fg_y = Fg_mag * dy / dist
    
    # dot product F * v gives the rate of work done
    E_dot = jnp.sum(Fg_x*u + Fg_y*v)
    
    return E_dot

@partial(jit, static_argnames=["hydro", "lattice"])
def get_E_dot_2(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    rho = U[..., 0]
    dA = lattice.dX1 * lattice.dX2
    
    # BH position and velocity
    _, _, x, y = hydro.get_bh_positions(t)
    _, _, u, v = hydro.get_bh_velocities(t)

    # gravitational force by gas on BH
    dx = lattice.X1 - x
    dy = lattice.X2 - y
    dist = jnp.sqrt(dx**2 + dy**2 + hydro.eps**2)
    Fg_mag = hydro.G() * (hydro.M / 2) * rho * dA / dist**2
    Fg_x = Fg_mag * dx / dist
    Fg_y = Fg_mag * dy / dist
    
    # dot product F * v gives the rate of work done
    E_dot = jnp.sum(Fg_x*u + Fg_y*v)
    
    return E_dot

@partial(jit, static_argnames=["hydro", "lattice"])
def get_E_dot_accr_1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    dA = lattice.dX1 * lattice.dX2
    x, y, _, _ = hydro.get_bh_positions(t)
    u, v, _, _ = hydro.get_bh_velocities(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x, y, u, v)
    m_dot_density = -sink_source[..., 0]  # mass accretion rate density
    
    rho = U[..., 0]
    v_x = U[..., 1] / rho
    v_y = U[..., 2] / rho
    
    # dot product of BH velocity with local fluid velocity
    dot_v = u*v_x + v*v_y
    return jnp.sum(m_dot_density * dot_v * dA)

@partial(jit, static_argnames=["hydro", "lattice"])
def get_E_dot_accr_2(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    dA = lattice.dX1 * lattice.dX2
    _, _, x, y = hydro.get_bh_positions(t)
    _, _, u, v = hydro.get_bh_velocities(t)
    sink_source = hydro.BH_sink(U, lattice.X1, lattice.X2, x, y, u, v)
    m_dot_density = -sink_source[..., 0]  # mass accretion rate density
    
    rho = U[..., 0]
    v_x = U[..., 1] / rho
    v_y = U[..., 2] / rho
    
    # dot product of BH velocity with local fluid velocity
    dot_v = u*v_x + v*v_y
    return jnp.sum(m_dot_density * dot_v * dA)

@partial(jit, static_argnames=["hydro", "lattice"])
def get_m_dot_buffer(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    X1, X2 = lattice.X1, lattice.X2
    S_buffer = hydro.buffer(U, X1, X2, t) # buffer source term
    rho_dot = S_buffer[..., 0] # rate at which density is removed by buffer
    dA = lattice.dX1 * lattice.dX2
    return jnp.sum(rho_dot * dA) # m_dot_buffer

@partial(jit, static_argnames=["hydro", "lattice"])
def get_L_dot_buffer(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    X1, X2 = lattice.X1, lattice.X2
    S_buffer = hydro.buffer(U, X1, X2, t)  # shape (Nx, Ny, 4)

    px = S_buffer[..., 1]
    py = S_buffer[..., 2]
    x = X1
    y = X2

    L_dot = x * py - y * px  # angular momentum density removal rate
    dA = lattice.dX1 * lattice.dX2

    return jnp.sum(L_dot * dA)

@dataclass(frozen=True)
class BinaryRing(Hydro):
    M: float = 1
    mach: float = 10
    a: float = 1
    eps: float = 0.05 * a
    omega_B: float = 1
    t_sink: float = 1 / (10 * omega_B)
    sink_rate: float = 10 * omega_B
    sink_prescription: str = "torque-free"
    R_0: float = 1 * a
    retrograde: bool = 0
    T: float = 1000
    
    cadence: float = 1
    CFL_num: float = 0.3
    size: float = 40
    res: int = 2000
    density_floor: float = 1e-6

    def initialize(self, lattice) -> Array:
        t = 0
        X1, X2 = lattice.X1, lattice.X2
        r, theta = cartesian_to_polar(X1, X2)
        dA = (X1[1, 0] - X1[0, 0]) * (X2[0, 1] - X2[0, 0])

        # surface density (normalized so that M_ring = 1)
        sigma = self.R_0 / 10
        ring = jnp.exp(- ((r - self.R_0) ** 2) / (2 * (sigma ** 2)))
        M_ring = jnp.sum(ring * dA)
        Sigma = ring / M_ring
        Sigma = jnp.maximum(self.density_floor, Sigma)

        v_r = jnp.zeros_like(Sigma)
        v_theta = jnp.sqrt(self.G() * self.M / r) # keplerian velocity
        if self.retrograde:
            v_theta *= -1
        u = v_r * jnp.cos(theta) - v_theta * jnp.sin(theta)
        v = v_r * jnp.sin(theta) + v_theta * jnp.cos(theta)
        p = Sigma * (self.c_s(None, X1, X2, 0) ** 2)

        return jnp.array([
            Sigma,
            u,
            v,
            p,
        ]).transpose((1, 2, 0))

    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-self.size/2, self.size/2), (-self.size/2, self.size/2))

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res)

    def t_end(self) -> float:
        return self.T * (2 * jnp.pi)
    
    def PLM(self) -> bool:
        return True
    
    def theta_PLM(self) -> bool:
        return 1.5
    
    def time_order(self) -> int:
        return 2
    
    def cfl(self) -> float:
        return self.CFL_num

    def coords(self) -> str:
        return "cartesian"

    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def nu(self) -> float:
        return 1e-3 * (self.a ** 2) * self.omega_B
    
    def G(self) -> float:
        return 1

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

        cs = jnp.sqrt(((self.G() * (self.M / 2) / jnp.sqrt(dist1 ** 2 + self.eps ** 2)) +
                       (self.G() * (self.M / 2) / jnp.sqrt(dist2 ** 2 + self.eps ** 2))) / (self.mach ** 2))
        return cs

    def P(self, cons: Conservatives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho = cons[..., 0]
        return rho * (self.c_s(cons, X1, X2, t) ** 2)

    def check_U(self, lattice: Lattice, U: ArrayLike, t: float) -> Array:
        rho = U[..., 0]
        apply_floor = rho < self.density_floor

        rho_new = jnp.where(apply_floor, self.density_floor, rho)

        # scale momenta to preserve velocity: p_new = v * rho_new = p_old * (rho_new / rho_old)
        factor = jnp.where(apply_floor, rho_new / rho, 1.0)
        mom_x_new = U[..., 1] * factor
        mom_y_new = U[..., 2] * factor

        U_new = U.at[..., 0].set(rho_new)
        U_new = U_new.at[..., 1].set(mom_x_new)
        U_new = U_new.at[..., 2].set(mom_y_new)
        
        return U_new

    def BH_gravity(self, U, x, y, x_bh, y_bh):
        dx, dy = x - x_bh, y - y_bh
        r = jnp.sqrt(dx ** 2 + dy ** 2)
        
        g_acc = - self.G() * (self.M / 2) / (r ** 2 + self.eps ** 2)
        g_x, g_y = g_acc * dx / (r + self.eps), g_acc * dy / (r + self.eps)
        rho = U[..., 0]
        u, v = U[..., 1] / rho, U[..., 2] / rho

        return jnp.array([
            jnp.zeros_like(rho),
            rho * g_x,
            rho * g_y,
            rho * (u * g_x + v * g_y)
        ]).transpose((1, 2, 0))

    def BH_sink(self, U, x, y, x_bh, y_bh, u_bh, v_bh):
        rho = U[..., 0]
        dx, dy = x - x_bh, y - y_bh
        r = jnp.sqrt(dx ** 2 + dy ** 2)
        r_sink = self.eps
        s_rate = jnp.where(r < (4 * r_sink), self.sink_rate * jnp.exp(-jnp.pow(r / r_sink, 4)), jnp.zeros_like(r))
        mdot = rho * s_rate * -1
        
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
        elif self.sink_prescription == "force-free":
            S = S.at[..., 0].set(mdot)
        else:
            raise AttributeError("Invalid sink prescription")

        return S

    def source(self, U: ArrayLike, X1, X2, t: float) -> Array:
        S = jnp.zeros_like(U)
        x_1, y_1, x_2, y_2 = self.get_bh_positions(t)
        u_1, v_1, u_2, v_2 = self.get_bh_velocities(t)
 
        # gravity
        S += self.BH_gravity(U, X1, X2, x_1, y_1) + \
            self.BH_gravity(U, X1, X2, x_2, y_2)

        # sinks
        S += self.BH_sink(U, X1, X2, x_1, y_1, u_1, v_1) + \
            self.BH_sink(U, X1, X2, x_2, y_2, u_2, v_2)
            
        # buffer
        S += self.buffer(U, X1, X2, t)

        return S

    # Buffer implementation from Sailfish
    def buffer(self, U: ArrayLike, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        x, y = X1, X2
        r, _ = cartesian_to_polar(x, y)

        surface_density = jnp.ones_like(x) * self.density_floor
        driving_rate = 100
        outer_radius = self.size / 2
        onset_width = 1
        onset_radius = outer_radius - onset_width
        
        v_kep = jnp.sqrt(self.G() * self.M / r)
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

    def get_bh_positions(self, t):
        delta = jnp.pi
        x1_1, x2_1 = (self.a / 2) * jnp.cos(self.omega_B * t), \
                     (self.a / 2) * jnp.sin(self.omega_B * t)
        x1_2, x2_2 = (self.a / 2) * jnp.cos(self.omega_B * t + delta), \
                     (self.a / 2) * jnp.sin(self.omega_B * t + delta)
        return x1_1, x2_1, x1_2, x2_2

    def get_bh_velocities(self, t):
        delta = jnp.pi
        u_1, v_1 = -(self.a / 2) * self.omega_B * jnp.sin(self.omega_B * t), \
                    (self.a / 2) * self.omega_B * jnp.cos(self.omega_B * t)
        u_2, v_2 = -(self.a / 2) * self.omega_B * jnp.sin(self.omega_B * t + delta), \
                    (self.a / 2) * self.omega_B * jnp.cos(self.omega_B * t + delta)
        return u_1, v_1, u_2, v_2

    def diagnostics(self):
        diagnostics = []
        diagnostics.append(("m_dot_1", get_m_dot_1))
        diagnostics.append(("m_dot_2", get_m_dot_2))
        diagnostics.append(("L_dot_1", get_L_dot_1))
        diagnostics.append(("L_dot_2", get_L_dot_2))
        diagnostics.append(("torque_1", get_torque_1))
        diagnostics.append(("torque_2", get_torque_2))
        diagnostics.append(("e_cav_x", get_cavity_eccentricity_x))
        diagnostics.append(("e_cav_y", get_cavity_eccentricity_y))
        diagnostics.append(("E_dot_1", get_E_dot_1))
        diagnostics.append(("E_dot_2", get_E_dot_2))
        diagnostics.append(("E_dot_accr_1", get_E_dot_accr_1))
        diagnostics.append(("E_dot_accr_2", get_E_dot_accr_2))
        diagnostics.append(("m_dot_buffer", get_m_dot_buffer))
        diagnostics.append(("L_dot_buffer", get_L_dot_buffer))
        
        return diagnostics

    def save_interval(self):
        return self.cadence * (2 * jnp.pi)
