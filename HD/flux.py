import jax.numpy as jnp
from functools import partial
from jax import vmap, jit, Array
from jax.typing import ArrayLike
from helpers import F_from_prim, G_from_prim, get_prims, add_ghost_cells, apply_bcs


def lambdas(v: ArrayLike, c_s: ArrayLike) -> tuple[Array, Array]:
    return v + c_s, v - c_s


def alphas(v_L: ArrayLike, v_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike) -> tuple[Array, Array]:
    lambda_L = lambdas(v_L, c_s_L)
    lambda_R = lambdas(v_R, c_s_R)

    alpha_p = jnp.maximum(0, jnp.maximum(lambda_L[0], lambda_R[0]))
    alpha_m = jnp.maximum(0, jnp.maximum(-lambda_L[1], -lambda_R[1]))

    return alpha_p, alpha_m


@vmap
@vmap
def hll_flux_x1(F_L: ArrayLike, F_R: ArrayLike, U_L: ArrayLike, U_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike) -> Array:
    rho_L, rho_R = U_L[0], U_R[0]
    v_L, v_R = U_L[1] / rho_L, U_R[1] / rho_R

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)


@vmap
@vmap
def hll_flux_x2(F_L: ArrayLike, F_R: ArrayLike, U_L: ArrayLike, U_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike) -> Array:
    rho_L, rho_R = U_L[0], U_R[0]
    v_L, v_R = U_L[2] / rho_L, U_R[2] / rho_R

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)


def finite_difference_x1(lattice, u: ArrayLike, x1_g: ArrayLike, x2_g: ArrayLike) -> Array:
    g = lattice.num_g
    if lattice.coords == "cartesian":
        dx = lattice.x1[1] - lattice.x1[0]
        du = (u[(g):-(g-1), g:-g] - u[(g-1):-(g), g:-g]) / (dx)
    elif lattice.coords == "polar":
        X1, _ = jnp.meshgrid(x1_g[1:-1], lattice.x2, indexing="ij")
        dR = jnp.diff(X1, axis=0)
        du = jnp.diff(u[(g-1):-(g-1), g:-g], axis=0) / dR
    return du


def finite_difference_x2(lattice, u: ArrayLike, x1_g: ArrayLike, x2_g: ArrayLike) -> Array:
    g = lattice.num_g

    if lattice.coords == "cartesian":
        dy = lattice.x2[1] - lattice.x2[0]
        du = (u[g:-g, (g):-(g-1)] - u[g:-g, (g-1):-(g)]) / (dy)
    elif lattice.coords == "polar":
        dtheta = lattice.x2[1] - lattice.x2[0]
        X1, _ = jnp.meshgrid(lattice.x1, lattice.x2_intf, indexing="ij")
        du = jnp.diff(u[g:-g, (g-1):-(g-1)], axis=1) / (X1 * dtheta)
    return du


def viscosity(hydro, lattice, U: ArrayLike, x1_g: ArrayLike, x2_g: ArrayLike) -> tuple[Array, Array, Array, Array]:
    g = lattice.num_g
    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    dudx = finite_difference_x1(lattice, u, x1_g, x2_g)
    dudy = finite_difference_x2(lattice, u, x1_g, x2_g)
    dvdx = finite_difference_x1(lattice, v, x1_g, x2_g)
    dvdy = finite_difference_x2(lattice, v, x1_g, x2_g)

    zero = jnp.zeros((lattice.nx1, lattice.nx2))

    rho_l = (rho[(g-1):-(g+1), g:-g] + rho[g:-g, g:-g]) / 2
    rho_r = (rho[g:-g, g:-g] + rho[(g+1):-(g-1), g:-g]) / 2
    Fv_l = -hydro.nu * jnp.array([
        zero,
        rho_l * dudx[:-1, :],
        rho_l * dvdx[:-1, :],
        zero
    ]).transpose((1, 2, 0))

    Fv_r = -hydro.nu * jnp.array([
        zero,
        rho_r * dudx[1:, :],
        rho_r * dvdx[1:, :],
        zero
    ]).transpose((1, 2, 0))

    rho_l = (rho[g:-g, (g-1):-(g+1)] + rho[g:-g, g:-g]) / 2
    rho_r = (rho[g:-g, g:-g] + rho[g:-g, (g+1):-(g-1)]) / 2
    Gv_l = -hydro.nu * jnp.array([
        zero,
        rho_l * dudy[:, :-1],
        rho_l * dvdy[:, :-1],
        zero
    ]).transpose((1, 2, 0))

    Gv_r = -hydro.nu * jnp.array([
        zero,
        rho_r * dudy[:, 1:],
        rho_r * dvdy[:, 1:],
        zero
    ]).transpose((1, 2, 0))

    return Fv_l, Fv_r, Gv_l, Gv_r


def interface_flux(hydro, lattice, U: ArrayLike, t: float) -> tuple[Array, Array, Array, Array]:
    g = lattice.num_g
    x1, x2 = lattice.x1, lattice.x2

    # add ghost regions to x1 and x2
    x1_g = jnp.concatenate([
        jnp.array([x1[0] - 2 * (x1[1] - x1[0]),
                   x1[0] - (x1[1] - x1[0])]),
        x1,
        jnp.array([x1[-1] + (x1[-1] - x1[-2]),
                   x1[-1] + 2 * (x1[-1] - x1[-2])])
    ])
    x2_g = jnp.concatenate([
        jnp.array([x2[0] - 2 * (x2[1] - x2[0]),
                   x2[0] - (x2[1] - x2[0])]),
        x2,
        jnp.array([x2[-1] + (x2[-1] - x2[-2]),
                   x2[-1] + 2 * (x2[-1] - x2[-2])])
    ])

    X1, X2 = jnp.meshgrid(x1_g, x2_g, indexing="ij")

    U = add_ghost_cells(U, g, axis=1)
    U = add_ghost_cells(U, g, axis=0)
    U = apply_bcs(lattice, U)
    U = hydro.check_U(lattice, U)

    prims = get_prims(hydro, U, X1, X2, t)
    F = F_from_prim(hydro, prims, X1, X2, t)
    G = G_from_prim(hydro, prims, X1, X2, t)

    F_L = F[(g-1):-(g+1), g:-g, :]
    F_C = F[g:-g, g:-g, :]
    F_R = F[(g+1):-(g-1), g:-g, :]
    G_L = G[g:-g, (g-1):-(g+1), :]
    G_C = G[g:-g, g:-g, :]
    G_R = G[g:-g, (g+1):-(g-1), :]
    X1_L = X1[(g-1):-(g+1), g:-g]
    X1_C = X1[g:-g, g:-g]
    X1_R = X1[(g+1):-(g-1), g:-g]
    X2_L = X2[g:-g, (g-1):-(g+1)]
    X2_C = X2[g:-g, g:-g]
    X2_R = X2[g:-g, (g+1):-(g-1)]

    U_L = U[(g-1):-(g+1), g:-g, :]
    U_C = U[g:-g, g:-g, :]
    U_R = U[(g+1):-(g-1), g:-g, :]
    prims_L = get_prims(hydro, U_L, X1_L, X2_C, t)
    prims_C = get_prims(hydro, U_C, X1_C, X2_C, t)
    prims_R = get_prims(hydro, U_R, X1_R, X2_C, t)
    c_s_L = hydro.c_s(prims_L, X1_L, X2_C, t)
    c_s_C = hydro.c_s(prims_C, X1_C, X2_C, t)
    c_s_R = hydro.c_s(prims_R, X1_R, X2_C, t)
    # F_(i-1/2)
    F_l = hll_flux_x1(F_L, F_C, U_L, U_C, c_s_L, c_s_C)
    # F_(i+1/2)
    F_r = hll_flux_x1(F_C, F_R, U_C, U_R, c_s_C, c_s_R)

    U_L = U[g:-g, (g-1):-(g+1), :]
    U_C = U[g:-g, g:-g, :]
    U_R = U[g:-g, (g+1):-(g-1)]
    prims_L = get_prims(hydro, U_L, X1_C, X2_L, t)
    prims_C = get_prims(hydro, U_C, X1_C, X2_C, t)
    prims_R = get_prims(hydro, U_R, X1_C, X2_R, t)
    c_s_L = hydro.c_s(prims_L, X1_C, X2_L, t)
    c_s_C = hydro.c_s(prims_C, X1_C, X2_C, t)
    c_s_R = hydro.c_s(prims_R, X1_C, X2_R, t)
    # G_(i-1/2)
    G_l = hll_flux_x2(G_L, G_C, U_L, U_C, c_s_L, c_s_C)
    # F_(i+1/2)
    G_r = hll_flux_x2(G_C, G_R, U_C, U_R, c_s_C, c_s_R)

    Fv_l, Fv_r, Gv_l, Gv_r = viscosity(hydro, lattice, U, x1_g, x2_g)
    F_l += Fv_l
    F_r += Fv_r
    G_l += Gv_l
    G_r += Gv_r

    return F_l, F_r, G_l, G_r
