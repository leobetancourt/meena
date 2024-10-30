import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from ..common.helpers import add_ghost_cells, apply_bcs, minmod, enthalpy, finite_difference_x1, finite_difference_x2

def get_prims(hydro, U, *args):
    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    e = U[..., 3]
    p = hydro.P((rho, rho * u, rho * v, e), *args)
    return rho, u, v, p


def U_from_prim(hydro, prims, *args):
    rho, u, v, _ = prims
    e = hydro.E(prims, *args)
    return jnp.array([
        rho,
        rho * u,
        rho * v,
        e
    ]).transpose((1, 2, 0))


def F_from_prim(hydro, prims, *args):
    rho, u, v, p = prims
    e = hydro.E(prims, *args)
    return jnp.array([
        rho * u,
        rho * (u ** 2) + p,
        rho * u * v,
        (e + p) * u
    ]).transpose((1, 2, 0))


def G_from_prim(hydro, prims, *args):
    rho, u, v, p = prims
    e = hydro.E(prims, *args)
    return jnp.array([
        rho * v,
        rho * u * v,
        rho * (v ** 2) + p,
        (e + p) * v
    ]).transpose((1, 2, 0))

def lambdas(v: ArrayLike, c_s: ArrayLike) -> tuple[Array, Array]:
    return v + c_s, v - c_s


def alphas(v_L: ArrayLike, v_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike) -> tuple[Array, Array]:
    lambda_L = lambdas(v_L, c_s_L)
    lambda_R = lambdas(v_R, c_s_R)

    alpha_p = jnp.maximum(0, jnp.maximum(lambda_L[0], lambda_R[0]))
    alpha_m = jnp.maximum(0, jnp.maximum(-lambda_L[1], -lambda_R[1]))

    return alpha_p, alpha_m


def hll_flux_x1(F_L: ArrayLike, F_R: ArrayLike,
                U_L: ArrayLike, U_R: ArrayLike,
                c_s_L: ArrayLike, c_s_R: ArrayLike) -> Array:
    rho_L, rho_R = U_L[..., 0], U_R[..., 0]
    v_L, v_R = U_L[..., 1] / rho_L, U_R[..., 1] / rho_R

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)
    a_p, a_m = a_p[..., None], a_m[..., None]

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)


def hll_flux_x2(F_L: ArrayLike, F_R: ArrayLike, U_L: ArrayLike, U_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike) -> Array:
    rho_L, rho_R = U_L[..., 0], U_R[..., 0]
    v_L, v_R = U_L[..., 2] / rho_L, U_R[..., 2] / rho_R

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)
    a_p, a_m = a_p[..., None], a_m[..., None]

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)


def F_star_x1(hydro, F_k, S_k, S_M, U_k, *args):
    rho_k, vx_k, vy_k, p_k = get_prims(hydro, U_k, *args)
    E_k = U_k[..., -1]
    v_k = vx_k

    rho_star = rho_k * (S_k - v_k) / (S_k - S_M)
    p_star = p_k + rho_k * (v_k - S_k) * (v_k - S_M)
    momx1_star = rho_k * ((S_k - v_k) / (S_k - S_M)) * S_M
    momx2_star = rho_k * vy_k * ((S_k - v_k) / (S_k - S_M))
    E_star = (E_k * (S_k - v_k) - p_k *
              v_k + p_star * S_M) / (S_k - S_M)

    U_star = jnp.array([rho_star, momx1_star, momx2_star, E_star])
    U_star = jnp.transpose(U_star, (1, 2, 0))
    S_k = S_k[..., None]
    return F_k + S_k * (U_star - U_k)


def F_star_x2(hydro, F_k, S_k, S_M, U_k, *args):
    rho_k, vx_k, vy_k, p_k = get_prims(hydro, U_k, *args)
    E_k = U_k[..., -1]
    v_k = vy_k

    rho_star = rho_k * (S_k - v_k) / (S_k - S_M)
    p_star = p_k + rho_k * (v_k - S_k) * (v_k - S_M)
    momx1_star = rho_k * vx_k * ((S_k - v_k) / (S_k - S_M))
    momx2_star = rho_k * ((S_k - v_k) / (S_k - S_M)) * S_M
    E_star = (E_k * (S_k - v_k) - p_k *
              v_k + p_star * S_M) / (S_k - S_M)

    U_star = jnp.array([rho_star, momx1_star, momx2_star, E_star])
    U_star = jnp.transpose(U_star, (1, 2, 0))
    S_k = S_k[..., None]
    return F_k + S_k * (U_star - U_k)


def hllc_flux_x1(hydro, F_L: ArrayLike, F_R: ArrayLike, U_L: ArrayLike, U_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike, *args) -> Array:
    """
            HLLC algorithm adapted from Robert Caddy
            https://robertcaddy.com/posts/HLLC-Algorithm/
    """
    rho_L, v_L, _, p_L = get_prims(hydro, U_L, *args)
    rho_R, v_R, _, p_R = get_prims(hydro, U_R, *args)
    e_L, e_R = U_L[..., -1], U_R[..., -1]

    R_rho = jnp.sqrt(rho_R / rho_L)
    H_L = enthalpy(rho_L, p_L, e_L)
    H_R = enthalpy(rho_R, p_R, e_R)
    H_t = (H_L + (H_R * R_rho)) / (1 + R_rho)  # H tilde
    v_t = (v_L + (v_R * R_rho)) / (1 + R_rho)
    c_t = jnp.sqrt((hydro.gamma() - 1) * (H_t - (0.5 * v_t ** 2)))

    S_L = jnp.minimum(v_L - c_s_L, v_t - c_t)
    S_R = jnp.maximum(v_R + c_s_R, v_t + c_t)
    S_M = (rho_R * v_R * (S_R - v_R) - rho_L * v_L * (S_L - v_L) + p_L - p_R) \
        / (rho_R * (S_R - v_R) - rho_L * (S_L - v_L))

    F = jnp.empty_like(F_L)
    case_1 = S_L > 0
    case_2 = (S_L <= 0) & (S_M > 0)
    case_3 = (S_M <= 0) & (S_R >= 0)
    case_4 = S_R < 0
    case_1 = case_1[..., None]
    case_2 = case_2[..., None]
    case_3 = case_3[..., None]
    case_4 = case_4[..., None]
    F = jnp.where(case_1, F_L, F)
    F = jnp.where(case_2, F_star_x1(
        hydro, F_L, S_L, S_M, U_L, *args), F)
    F = jnp.where(case_3, F_star_x1(
        hydro, F_R, S_R, S_M, U_R, *args), F)
    F = jnp.where(case_4, F_R, F)

    return F


def hllc_flux_x2(hydro, G_L: ArrayLike, G_R: ArrayLike, U_L: ArrayLike, U_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike, *args) -> Array:
    """
            HLLC algorithm adapted from Robert Caddy
            https://robertcaddy.com/posts/HLLC-Algorithm/
    """
    rho_L, _, v_L, p_L = get_prims(hydro, U_L, *args)
    rho_R, _, v_R, p_R = get_prims(hydro, U_R, *args)
    e_L, e_R = U_L[..., -1], U_R[..., -1]

    R_rho = jnp.sqrt(rho_R / rho_L)
    H_L = enthalpy(rho_L, p_L, e_L)
    H_R = enthalpy(rho_R, p_R, e_R)
    H_t = (H_L + (H_R * R_rho)) / (1 + R_rho)  # H tilde
    v_t = (v_L + (v_R * R_rho)) / (1 + R_rho)
    c_t = jnp.sqrt((hydro.gamma() - 1) * (H_t - (0.5 * v_t ** 2)))

    S_L = jnp.minimum(v_L - c_s_L, v_t - c_t)
    S_R = jnp.maximum(v_R + c_s_R, v_t + c_t)
    S_M = (rho_R * v_R * (S_R - v_R) - rho_L * v_L * (S_L - v_L) + p_L - p_R) \
        / (rho_R * (S_R - v_R) - rho_L * (S_L - v_L))

    G = jnp.empty_like(G_L)
    case_1 = S_L > 0
    case_2 = (S_L <= 0) & (S_M > 0)
    case_3 = (S_M <= 0) & (S_R >= 0)
    case_4 = S_R < 0
    case_1 = case_1[..., None]
    case_2 = case_2[..., None]
    case_3 = case_3[..., None]
    case_4 = case_4[..., None]
    G = jnp.where(case_1, G_L, G)
    G = jnp.where(case_2, F_star_x2(
        hydro, G_L, S_L, S_M, U_L, *args), G)
    G = jnp.where(case_3, F_star_x2(
        hydro, G_R, S_R, S_M, U_R, *args), G)
    G = jnp.where(case_4, G_R, G)

    return G


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
    Fv_l = -hydro.nu() * jnp.array([
        zero,
        rho_l * dudx[:-1, :],
        rho_l * dvdx[:-1, :],
        zero
    ]).transpose((1, 2, 0))

    Fv_r = -hydro.nu() * jnp.array([
        zero,
        rho_r * dudx[1:, :],
        rho_r * dvdx[1:, :],
        zero
    ]).transpose((1, 2, 0))

    rho_l = (rho[g:-g, (g-1):-(g+1)] + rho[g:-g, g:-g]) / 2
    rho_r = (rho[g:-g, g:-g] + rho[g:-g, (g+1):-(g-1)]) / 2
    Gv_l = -hydro.nu() * jnp.array([
        zero,
        rho_l * dudy[:, :-1],
        rho_l * dvdy[:, :-1],
        zero
    ]).transpose((1, 2, 0))

    Gv_r = -hydro.nu() * jnp.array([
        zero,
        rho_r * dudy[:, 1:],
        rho_r * dvdy[:, 1:],
        zero
    ]).transpose((1, 2, 0))

    return Fv_l, Fv_r, Gv_l, Gv_r


def interface_flux(hydro, lattice, U: ArrayLike, t: float) -> tuple[Array, Array, Array, Array]:
    g = lattice.num_g
    x1, x2 = lattice.x1, lattice.x2

    x1_left = x1[0] - (x1[1] - x1[0]) * jnp.arange(g, 0, -1)
    x1_right = x1[-1] + (x1[-1] - x1[-2]) * jnp.arange(1, g + 1)
    x1_g = jnp.concatenate([x1_left, x1, x1_right])

    x2_left = x2[0] - (x2[1] - x2[0]) * jnp.arange(g, 0, -1)
    x2_right = x2[-1] + (x2[-1] - x2[-2]) * jnp.arange(1, g + 1)
    x2_g = jnp.concatenate([x2_left, x2, x2_right])

    X1, X2 = jnp.meshgrid(x1_g, x2_g, indexing="ij")

    U = add_ghost_cells(U, g, axis=1)
    U = add_ghost_cells(U, g, axis=0)
    U = apply_bcs(lattice, U)
    U = hydro.check_U(lattice, U, t)

    X1_LL = X1[:-(g+2), g:-g]
    X1_L = X1[(g-1):-(g+1), g:-g]
    X1_C = X1[g:-g, g:-g]
    X1_R = X1[(g+1):-(g-1), g:-g]
    X1_RR = X1[(g+2):, g:-g]
    X2_LL = X2[g:-g, :-(g+2)]
    X2_L = X2[g:-g, (g-1):-(g+1)]
    X2_C = X2[g:-g, g:-g]
    X2_R = X2[g:-g, (g+1):-(g-1)]
    X2_RR = X2[g:-g, (g+2):]

    if hydro.PLM():
        theta = hydro.theta_PLM()

        prims_C = jnp.asarray(get_prims(hydro, U[g:-g, g:-g], X1_C, X2_C, t))
        prims_LL = jnp.asarray(
            get_prims(hydro, U[:-(g+2), g:-g], X1_LL, X2_C, t))
        prims_L = jnp.asarray(
            get_prims(hydro, U[(g-1):-(g+1), g:-g], X1_L, X2_C, t))
        prims_R = jnp.asarray(
            get_prims(hydro, U[(g+1):-(g-1), g:-g], X1_R, X2_C, t))
        prims_RR = jnp.asarray(
            get_prims(hydro, U[(g+2):, g:-g], X1_RR, X2_C, t))

        # left cell interface (i-1/2)
        # left-biased state
        prims_ll = prims_L - 0.5 * \
            minmod(theta * (prims_L - prims_LL), 0.5 *
                   (prims_C - prims_LL), theta * (prims_C - prims_L))
        # right-biased state
        prims_lr = prims_C + 0.5 * \
            minmod(theta * (prims_C - prims_L), 0.5 *
                   (prims_R - prims_L), theta * (prims_R - prims_C))

        # right cell interface (i+1/2)
        # left-biased state
        prims_rl = prims_C - 0.5 * \
            minmod(theta * (prims_C - prims_L), 0.5 *
                   (prims_R - prims_L), theta * (prims_R - prims_C))
        # right-biased state
        prims_rr = prims_R + 0.5 * \
            minmod(theta * (prims_R - prims_C), 0.5 *
                   (prims_RR - prims_C), theta * (prims_RR - prims_R))

        # maybe for rl and lr i need to use X1_C and X2_C
        F_ll, F_lr, F_rl, F_rr = F_from_prim(hydro, prims_ll, X1_L, X2_C, t), F_from_prim(
            hydro, prims_lr, X1_C, X2_C, t), F_from_prim(hydro, prims_rl, X1_C, X2_C, t), F_from_prim(hydro, prims_rr, X1_R, X2_C, t)
        U_ll, U_lr, U_rl, U_rr = U_from_prim(hydro, prims_ll, X1_L, X2_C, t), U_from_prim(
            hydro, prims_lr, X1_C, X2_C, t), U_from_prim(hydro, prims_rl, X1_C, X2_C, t), U_from_prim(hydro, prims_rr, X1_R, X2_C, t)
        c_s_ll, c_s_lr, c_s_rl, c_s_rr = hydro.c_s(prims_ll, X1_L, X2_C, t), hydro.c_s(
            prims_lr, X1_C, X2_C, t), hydro.c_s(prims_rl, X1_C, X2_C, t), hydro.c_s(prims_rr, X1_R, X2_C, t)

        if hydro.solver() == "hll":
            F_l, F_r = hll_flux_x1(F_ll, F_lr, U_ll, U_lr, c_s_ll, c_s_lr), hll_flux_x1(
                F_rl, F_rr, U_rl, U_rr, c_s_rl, c_s_rr)
        elif hydro.solver() == "hllc":
            F_l, F_r = hllc_flux_x1(hydro, F_ll, F_lr, U_ll, U_lr, c_s_ll, c_s_lr, X1_L, X1_C, X2_C, t), hllc_flux_x1(hydro,
                                                                                                                      F_rl, F_rr, U_rl, U_rr, c_s_rl, c_s_rr, X1_C, X1_R, X2_C, t)

        prims_LL = jnp.asarray(
            get_prims(hydro, U[g:-g, :-(g+2)], X1_C, X2_LL, t))
        prims_L = jnp.asarray(
            get_prims(hydro, U[g:-g, (g-1):-(g+1)], X1_C, X2_L, t))
        prims_R = jnp.asarray(
            get_prims(hydro, U[g:-g, (g+1):-(g-1)], X1_C, X2_R, t))
        prims_RR = jnp.asarray(
            get_prims(hydro, U[g:-g, (g+2):], X1_C, X2_RR, t))

        # left cell interface (i-1/2)
        # left-biased state
        prims_ll = prims_L - 0.5 * \
            minmod(theta * (prims_L - prims_LL), 0.5 *
                   (prims_C - prims_LL), theta * (prims_C - prims_L))
        # right-biased state
        prims_lr = prims_C + 0.5 * \
            minmod(theta * (prims_C - prims_L), 0.5 *
                   (prims_R - prims_L), theta * (prims_R - prims_C))

        # right cell interface (i+1/2)
        # left-biased state
        prims_rl = prims_C - 0.5 * \
            minmod(theta * (prims_C - prims_L), 0.5 *
                   (prims_R - prims_L), theta * (prims_R - prims_C))
        # right-biased state
        prims_rr = prims_R + 0.5 * \
            minmod(theta * (prims_R - prims_C), 0.5 *
                   (prims_RR - prims_C), theta * (prims_RR - prims_R))

        G_ll, G_lr, G_rl, G_rr = G_from_prim(hydro, prims_ll, X1_C, X2_L, t), G_from_prim(
            hydro, prims_lr, X1_C, X2_C, t), G_from_prim(hydro, prims_rl, X1_C, X2_C, t), G_from_prim(hydro, prims_rr, X1_C, X2_R, t)
        U_ll, U_lr, U_rl, U_rr = U_from_prim(hydro, prims_ll, X1_C, X2_L, t), U_from_prim(
            hydro, prims_lr, X1_C, X2_C, t), U_from_prim(hydro, prims_rl, X1_C, X2_C, t), U_from_prim(hydro, prims_rr, X1_C, X2_R, t)
        c_s_ll, c_s_lr, c_s_rl, c_s_rr = hydro.c_s(prims_ll, X1_C, X2_L, t), hydro.c_s(
            prims_lr, X1_C, X2_C, t), hydro.c_s(prims_rl, X1_C, X2_C, t), hydro.c_s(prims_rr, X1_C, X2_R, t)

        if hydro.solver() == "hll":
            G_l, G_r = hll_flux_x2(G_ll, G_lr, U_ll, U_lr, c_s_ll, c_s_lr), hll_flux_x2(
                G_rl, G_rr, U_rl, U_rr, c_s_rl, c_s_rr)
        elif hydro.solver() == "hllc":
            G_l, G_r = hllc_flux_x2(hydro, G_ll, G_lr, U_ll, U_lr, c_s_ll, c_s_lr, X1_C, X2_L, X2_C, t), hllc_flux_x2(
                hydro, G_rl, G_rr, U_rl, U_rr, c_s_rl, c_s_rr, X1_C, X2_C, X2_R, t)
    else:
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
        if hydro.solver() == "hll":
            # F_(i-1/2)
            F_l = hll_flux_x1(F_L, F_C, U_L, U_C, c_s_L, c_s_C)
            # F_(i+1/2)
            F_r = hll_flux_x1(F_C, F_R, U_C, U_R, c_s_C, c_s_R)
        elif hydro.solver() == "hllc":
            F_l = hllc_flux_x1(hydro, F_L, F_C, U_L, U_C,
                               c_s_L, c_s_C, X1_L, X1_C, X2_C, t)
            F_r = hllc_flux_x1(hydro, F_C, F_R, U_C, U_R,
                               c_s_C, c_s_R, X1_C, X1_R, X2_C, t)

        U_L = U[g:-g, (g-1):-(g+1), :]
        U_C = U[g:-g, g:-g, :]
        U_R = U[g:-g, (g+1):-(g-1)]
        prims_L = get_prims(hydro, U_L, X1_C, X2_L, t)
        prims_C = get_prims(hydro, U_C, X1_C, X2_C, t)
        prims_R = get_prims(hydro, U_R, X1_C, X2_R, t)
        c_s_L = hydro.c_s(prims_L, X1_C, X2_L, t)
        c_s_C = hydro.c_s(prims_C, X1_C, X2_C, t)
        c_s_R = hydro.c_s(prims_R, X1_C, X2_R, t)
        if hydro.solver() == "hll":
            # G_(i-1/2)
            G_l = hll_flux_x2(G_L, G_C, U_L, U_C, c_s_L, c_s_C)
            # F_(i+1/2)
            G_r = hll_flux_x2(G_C, G_R, U_C, U_R, c_s_C, c_s_R)
        elif hydro.solver() == "hllc":
            # G_(i-1/2)
            G_l = hllc_flux_x2(hydro, G_L, G_C, U_L, U_C,
                               c_s_L, c_s_C, X1_C, X2_L, X2_C, t)
            # F_(i+1/2)
            G_r = hllc_flux_x2(hydro, G_C, G_R, U_C, U_R,
                               c_s_C, c_s_R, X1_C, X2_C, X2_R, t)

    if hydro.nu():
        Fv_l, Fv_r, Gv_l, Gv_r = viscosity(hydro, lattice, U, x1_g, x2_g)
        F_l += Fv_l
        F_r += Fv_r
        G_l += Gv_l
        G_r += Gv_r

    return F_l, F_r, G_l, G_r