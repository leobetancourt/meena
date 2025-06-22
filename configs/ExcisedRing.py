from dataclasses import dataclass
from functools import partial

from jax.typing import ArrayLike
from jax import Array, jit
import jax.numpy as jnp

from meena import Hydro, Lattice, Primitives, Conservatives, BoundaryCondition, Coords

@partial(jit, static_argnames=["hydro", "lattice"])
def get_accr_rate(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dr = lattice.x1_intf[1] - lattice.x1_intf[0]
    dtheta = lattice.x2_intf[1] - lattice.x2_intf[0]
    dA = lattice.x1[0] * dr * dtheta
    F_l, F_r, G_l, G_r = flux
    m_dot = -(F_l[0, :, 0] / dr) * dA
    return jnp.sum(m_dot)


@partial(jit, static_argnames=["hydro", "lattice"])
def get_angular_mom_rate(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    dr = lattice.x1_intf[1] - lattice.x1_intf[0]
    dtheta = lattice.x2_intf[1] - lattice.x2_intf[0]
    dA = lattice.x1[0] * dr * dtheta
    F_l, F_r, G_l, G_r = flux
    m_dot = -(F_l[0, :, 0] / dr) * dA
    vtheta = U[0, :, 2] / U[0, :, 0]
    return jnp.sum(m_dot * lattice.x1[0] * vtheta)



@partial(jit, static_argnames=["hydro", "lattice"])
def get_torque1(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    x1_1, x2_1, x1_2, x2_2 = hydro.get_positions(t)
    rho = U[..., 0]
    r, theta = lattice.X1, lattice.X2
    r_bh, theta_bh = x1_1, x2_1
    R_interf, _ = jnp.meshgrid(lattice.x1_intf, lattice.x2, indexing="ij")
    x1_l, x1_r = R_interf[:-1, :], R_interf[1:, :]
    dx1 = x1_r - x1_l
    dx2 = lattice.x2[1] - lattice.x2[0]
    dA = r * dx1 * dx2
    
    delta_theta = theta - theta_bh
    dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                    r_bh * jnp.cos(delta_theta))
    Fg = hydro.G() * (hydro.M / 2) * rho * dA / \
        (dist ** 2 + hydro.eps ** 2)
    Fg_theta = Fg * (r * jnp.sin(delta_theta)) / dist
    T = jnp.sum((hydro.a / 2) * Fg_theta)
    return T


@partial(jit, static_argnames=["hydro", "lattice"])
def get_torque2(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float) -> float:
    x1_1, x2_1, x1_2, x2_2 = hydro.get_positions(t)
    rho = U[..., 0]
    r, theta = lattice.X1, lattice.X2
    r_bh, theta_bh = x1_2, x2_2
    R_interf, _ = jnp.meshgrid(lattice.x1_intf, lattice.x2, indexing="ij")
    x1_l, x1_r = R_interf[:-1, :], R_interf[1:, :]
    dx1 = x1_r - x1_l
    dx2 = lattice.x2[1] - lattice.x2[0]
    dA = r * dx1 * dx2  # dA = rdrdtheta

    delta_theta = theta - theta_bh
    dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                    r_bh * jnp.cos(delta_theta))
    Fg = hydro.G() * (hydro.M / 2) * rho * dA / \
        (dist ** 2 + hydro.eps ** 2)
    Fg_theta = Fg * (r * jnp.sin(delta_theta)) / dist
    T = jnp.sum((hydro.a / 2) * Fg_theta)

    return T

@dataclass(frozen=True)
class ExcisedRing(Hydro):
    M: float = 1
    mach: float = 10
    a: float = 1
    eps: float = 0.05 * a
    omega_B: float = 1
    R_0: float = 4 * a
    retrograde: bool = 0
    
    cadence: float = 10
    T: float = 2000
    CFL_num: float = 0.3
    r_min: float = 1
    r_max: float = 100 * a
    res: int = 1200
    density_floor: float = 1e-6

    def initialize(self, lattice) -> Array:
        t = 0
        r, theta = lattice.X1, lattice.X2
        dA = lattice.X1 * lattice.dX1 * lattice.dX2

        # surface density
        sigma = self.R_0 / 10
        ring = jnp.exp(- ((r - self.R_0) ** 2) / (2 * (sigma ** 2)))
        M_ring = jnp.sum(ring * dA)
        Sigma = ring / M_ring
        Sigma = jnp.maximum(self.density_floor, Sigma)

        v_r = jnp.zeros_like(Sigma)
        v_theta = jnp.sqrt(self.G() * self.M / r) # keplerian velocity
        if self.retrograde:
            v_theta *= -1
        
        p = Sigma * (self.c_s(None, r, theta, 0) ** 2)

        return jnp.array([
            Sigma,
            v_r,
            v_theta,
            p,
        ]).transpose((1, 2, 0))
                
    def range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.r_min, self.r_max), (0, 2 * jnp.pi))

    def log_x1(self) -> bool:
        return True

    def resolution(self) -> tuple[int, int]:
        return (self.res, self.res * 6)

    def t_end(self) -> float:
        return self.T * 2 * jnp.pi
    
    def PLM(self) -> bool:
        return True
    
    def theta_PLM(self):
        return 1.5
    
    def time_order(self):
        return 2
    
    def cfl(self) -> float:
        return self.CFL_num
    
    def nu(self) -> float:
        return 1e-3

    def coords(self) -> str:
        return "polar"

    def bc_x1(self) -> BoundaryCondition:
        return ("outflow", "outflow")

    def bc_x2(self) -> BoundaryCondition:
        return ("periodic", "periodic")
    
    def inflow(self) -> bool:
        return True

    def E(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho, u, v = prims[..., 0], prims[..., 1], prims[..., 2]
        e_internal = rho * (self.c_s(prims, X1, X2, t) ** 2)
        e_kinetic = 0.5 * rho * (u ** 2 + v ** 2)
        return e_internal + e_kinetic

    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        x1_1, x2_1, x1_2, x2_2 = self.get_positions(t)
        r, theta = X1, X2
        r1, theta1 = x1_1, x2_1
        r2, theta2 = x1_2, x2_2

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
    
    def G(self) -> float:
        return 1

    def BH_gravity(self, U, r, theta, r_bh, theta_bh):
        delta_theta = theta - theta_bh
        dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                        r_bh * jnp.cos(delta_theta))
        g_acc = -self.G() * (self.M / 2) / (dist ** 2)

        g_r = g_acc * (r - r_bh * jnp.cos(delta_theta)) / dist
        g_theta = g_acc * (r_bh * jnp.sin(delta_theta)) / dist
        rho = U[..., 0]
        u, v = U[..., 1] / rho, U[..., 2] / rho

        return jnp.array([
            jnp.zeros_like(rho),
            rho * g_r,
            rho * g_theta,
            rho * (u * g_r + v * g_theta)
        ]).transpose((1, 2, 0))

    def source(self, U: ArrayLike, X1, X2, t: float) -> Array:
        x1_1, x2_1, x1_2, x2_2 = self.get_positions(t)
        S = jnp.zeros_like(U)
        S = S + self.BH_gravity(U, X1, X2, x1_1, x2_1)
        S = S + self.BH_gravity(U, X1, X2, x1_2, x2_2)
        return S

    def get_positions(self, t):
        delta = jnp.pi
        x1_1, x2_1 = (self.a / 2), (self.omega_B * t) % (2 * jnp.pi)
        x1_2, x2_2 = (self.a / 2), (self.omega_B * t + delta) % (2 * jnp.pi)
        # x1_1, x2_1 = 0, 0
        # x1_2, x2_2 = 0, 0

        return x1_1, x2_1, x1_2, x2_2

    # assumes prims with ghost cells
    def check_U(self, lattice: Lattice, U: ArrayLike, t: float) -> Array:
        # density floor
        rho = U[..., 0]
        apply_floor = rho < self.density_floor

        rho_new = jnp.where(apply_floor, self.density_floor, rho)

        # scale momenta to preserve velocity: p_new = v * rho_new = p_old * (rho_new / rho_old)
        factor = jnp.where(apply_floor, rho_new / rho, 1.0)
        mom_r_new = U[..., 1] * factor
        mom_theta_new = U[..., 2] * factor

        U_new = U.at[..., 0].set(rho_new)
        U_new = U_new.at[..., 1].set(mom_r_new)
        U_new = U_new.at[..., 2].set(mom_theta_new)
        
        return U_new

    def diagnostics(self):
        diagnostics = []
        diagnostics.append(("m_dot", get_accr_rate))
        diagnostics.append(("L_dot", get_angular_mom_rate))
        diagnostics.append(("torque_1", get_torque1))
        diagnostics.append(("torque_2", get_torque1))
        return diagnostics

    def save_interval(self):
        return self.cadence * (2 * jnp.pi)
