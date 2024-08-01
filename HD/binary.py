import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit
from functools import partial
from dataclasses import dataclass

from hydro import Hydro, Lattice, Coords, run, Primitives, Conservatives
from helpers import Boundary, cartesian_to_polar, load_U, read_csv


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
    nu: float = 1e-3 * (a ** 2) * omega_B

    def E(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        rho, u, v, p = prims
        e_internal = rho * (self.c_s(prims, X1, X2, t) ** 2)
        e_kinetic = 0.5 * rho * (u ** 2 + v ** 2)
        return e_internal + e_kinetic

    def c_s(self, prims: Primitives, X1: ArrayLike, X2: ArrayLike, t: float) -> Array:
        x1_1, x2_1, x1_2, x2_2 = self.get_positions(t)
        if self.coords == Coords.CARTESIAN:
            r, theta = cartesian_to_polar(X1, X2)
            r1, theta1 = cartesian_to_polar(x1_1, x2_1)
            r2, theta2 = cartesian_to_polar(x1_2, x2_2)
        else:
            r, theta = X1, X2
            r1, theta1 = x1_1, x2_1
            r2, theta2 = x1_2, x2_2

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

    def BH_gravity(self, U, r, theta, r_bh, theta_bh):
        delta_theta = theta - theta_bh
        dist = jnp.sqrt(r ** 2 + r_bh ** 2 - 2 * r *
                        r_bh * jnp.cos(delta_theta))
        g_acc = -self.G * (self.M / 2) / (dist ** 2 + self.eps ** 2)

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
        if self.coords == Coords.CARTESIAN:
            x1_1, x2_1 = (self.a / 2) * jnp.cos(self.omega_B *
                                                t), (self.a / 2) * jnp.cos(self.omega_B * t)
            x1_2, x2_2 = (self.a / 2) * jnp.cos(self.omega_B * t +
                                                delta), (self.a / 2) * jnp.cos(self.omega_B * t + delta)
        elif self.coords == Coords.POLAR:
            x1_1, x2_1 = (self.a / 2), (self.omega_B * t) % (2 * jnp.pi)
            x1_2, x2_2 = (self.a / 2), (self.omega_B *
                                        t + delta) % (2 * jnp.pi)

        return x1_1, x2_1, x1_2, x2_2

    def setup(self, X1, X2):
        t = 0
        r, theta = X1, X2

        def f(r):
            return 1
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

        return jnp.array([
            rho,
            rho * v_r,
            rho * v_theta,
            self.E((rho, v_r, v_theta, jnp.zeros_like(rho)), X1, X2, t)
        ]).transpose((1, 2, 0))

    # assumes U with ghost cells
    def check_U(self, lattice: Lattice, U: ArrayLike) -> Array:
        g = lattice.num_g
        if self.coords == Coords.POLAR:
            rho = U[:, :, 0]
            vr = U[:, :, 1] / rho
            vr = vr.at[:g, :].set(jnp.minimum(vr[:g, :], 0))
            U.at[:g, :, 1].set(rho[:g, :] * vr[:g, :])
        return U


""" DIAGNOSTICS """


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
    dA = r * dx1 * dx2  # dA = rdrdtheta

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
    Fg = hydro.G * (hydro.M / 2) * rho * dA / \
        (dist ** 2 + hydro.eps ** 2)
    Fg_theta = Fg * (r * jnp.sin(delta_theta)) / dist
    T = jnp.sum((hydro.a / 2) * Fg_theta)

    return T


def get_eccentricity(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    rho = U[..., 0]
    vr, vtheta = U[..., 1] / rho, U[..., 2] / rho
    r, theta = lattice.X1, lattice.X2
    R_interf, _ = jnp.meshgrid(lattice.x1_intf, lattice.x2, indexing="ij")
    x1_l, x1_r = R_interf[:-1, :], R_interf[1:, :]
    dx1 = x1_r - x1_l
    dx2 = lattice.x2[1] - lattice.x2[0]
    dA = r * dx1 * dx2
    e_x = (r * vr * vtheta / (hydro.G * hydro.M)) * jnp.sin(theta) + \
        (((r * vtheta ** 2) / (hydro.G * hydro.M)) - 1) * jnp.cos(theta)
    e_y = -(r * vr * vtheta / (hydro.G * hydro.M)) * jnp.cos(theta) + \
        (((r * vtheta ** 2) / (hydro.G * hydro.M)) - 1) * jnp.sin(theta)

    bounds = jnp.logical_and(r >= hydro.a, r <= 6 * hydro.a)
    ec_x = jnp.where(bounds, e_x * rho * dA, 0).sum() / \
        (35 * jnp.pi * hydro.Sigma_0 * (hydro.a ** 2))
    ec_y = jnp.where(bounds, e_y * rho * dA, 0).sum() / \
        (35 * jnp.pi * hydro.Sigma_0 * (hydro.a ** 2))
    return (ec_x, ec_y)


@partial(jit, static_argnames=["hydro", "lattice"])
def get_eccentricity_x(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    return get_eccentricity(hydro, lattice, U, flux, t)[0]


@partial(jit, static_argnames=["hydro", "lattice"])
def get_eccentricity_y(hydro: Hydro, lattice: Lattice, U: ArrayLike, flux: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], t: float):
    return get_eccentricity(hydro, lattice, U, flux, t)[1]


if __name__ == "__main__":
    # get smallest timestep from diagnostics
    diagnostics_high = read_csv("./500x3000/diagnostics.csv")
    min_dt = min(diagnostics_high["dt"])
    
    binary = Binary(coords=Coords.POLAR, mach=10, dt=min_dt)
    lattice = Lattice(
        coords=Coords.POLAR,
        bc_x1=(Boundary.OUTFLOW, Boundary.OUTFLOW),
        bc_x2=(Boundary.PERIODIC, Boundary.PERIODIC),
        nx1=500,
        nx2=3000,
        x1_range=(1, 30),
        x2_range=(0, 2 * jnp.pi),
        log_x1=True
    )

    OUT_PATH = "./500x3000_inflow"

    U = binary.setup(lattice.X1, lattice.X2)
    t = 0
    # U, t = load_U(f"{OUT_PATH}/checkpoints/out_1249.31.h5")

    diagnostics = []
    diagnostics.append(("m_dot", get_accr_rate))
    diagnostics.append(("L_dot", get_angular_mom_rate))
    diagnostics.append(("torque_1", get_torque1))
    diagnostics.append(("torque_2", get_torque2))
    diagnostics.append(("e_x", get_eccentricity_x))
    diagnostics.append(("e_y", get_eccentricity_y))

    run(binary, lattice, U, t=t, T=300 * 2 * jnp.pi,
        out=OUT_PATH, save_interval=(2 * jnp.pi / 24), diagnostics=diagnostics)
