import jax.numpy as jnp
from jax import Array
from jax import debug
from jax.typing import ArrayLike
from ..common.helpers import add_ghost_cells, apply_bcs, minmod, enthalpy

def get_prims(hydro, U, *args):
    rho = U[..., 0]
    u, v = U[..., 1] / rho, U[..., 2] / rho
    p = hydro.P(U, *args)
    return jnp.array([
        rho,
        u,
        v,
        p
    ]).transpose((1, 2, 0))

def U_from_prim(hydro, prims, *args):
    rho, u, v = prims[..., 0], prims[..., 1], prims[..., 2]
    e = hydro.E(prims, *args)
    return jnp.array([
        rho,
        rho * u,
        rho * v,
        e
    ]).transpose((1, 2, 0))


def F_from_prim(hydro, prims, *args):
    rho, u, v, p = prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]
    e = hydro.E(prims, *args)
    return jnp.array([
        rho * u,
        rho * (u ** 2) + p,
        rho * u * v,
        (e + p) * u
    ]).transpose((1, 2, 0))


def G_from_prim(hydro, prims, *args):
    rho, u, v, p = prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]
    e = hydro.E(prims, *args)
    return jnp.array([
        rho * v,
        rho * u * v,
        rho * (v ** 2) + p,
        (e + p) * v
    ]).transpose((1, 2, 0))

def lambdas(v: ArrayLike, c_s: ArrayLike) -> tuple[Array, Array]:
    return v - c_s, v + c_s

def alphas(v_L: ArrayLike, v_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike) -> tuple[Array, Array]:
    lambda_L = lambdas(v_L, c_s_L)
    lambda_R = lambdas(v_R, c_s_R)

    alpha_m = jnp.minimum(0, jnp.minimum(lambda_L[0], lambda_R[0]))
    alpha_p = jnp.maximum(0, jnp.maximum(lambda_L[1], lambda_R[1]))

    return alpha_p, alpha_m


def hll_flux_x1(hydro, prims_L: ArrayLike, prims_R: ArrayLike, *args) -> Array:
    v_L, v_R = prims_L[..., 1], prims_R[..., 1]
    c_s_L, c_s_R = hydro.c_s(prims_L, *args), hydro.c_s(prims_R, *args)

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)
    a_p, a_m = a_p[..., jnp.newaxis], a_m[..., jnp.newaxis]
    
    F_L, F_R = F_from_prim(hydro, prims_L, *args), F_from_prim(hydro, prims_R, *args)
    U_L, U_R = U_from_prim(hydro, prims_L, *args), U_from_prim(hydro, prims_R, *args)

    return (a_p * F_L - a_m * F_R - a_p * a_m * (U_L - U_R)) / (a_p - a_m)


def hll_flux_x2(hydro, prims_L: ArrayLike, prims_R: ArrayLike, *args) -> Array:
    v_L, v_R = prims_L[..., 2], prims_R[..., 2]
    c_s_L, c_s_R = hydro.c_s(prims_L, *args), hydro.c_s(prims_R, *args)

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)
    a_p, a_m = a_p[..., jnp.newaxis], a_m[..., jnp.newaxis]
    
    G_L, G_R = G_from_prim(hydro, prims_L, *args), G_from_prim(hydro, prims_R, *args)
    U_L, U_R = U_from_prim(hydro, prims_L, *args), U_from_prim(hydro, prims_R, *args)

    return (a_p * G_L - a_m * G_R - a_p * a_m * (U_L - U_R)) / (a_p - a_m)


def F_star_x1(F_k, S_k, S_M, prims_k, U_k):
    rho_k, vx_k, vy_k, p_k = prims_k[..., 0], prims_k[..., 1], prims_k[..., 2], prims_k[..., 3]
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


def F_star_x2(F_k, S_k, S_M, prims_k, U_k):
    rho_k, vx_k, vy_k, p_k = prims_k[..., 0], prims_k[..., 1], prims_k[..., 2], prims_k[..., 3]
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


def hllc_flux_x1(hydro, prims_L: ArrayLike, prims_R: ArrayLike, *args) -> Array:
    """
            HLLC algorithm adapted from Robert Caddy
            https://robertcaddy.com/posts/HLLC-Algorithm/
    """
    rho_L, v_L, p_L = prims_L[..., 0], prims_L[..., 1], prims_L[..., 3]
    rho_R, v_R, p_R = prims_R[..., 0], prims_R[..., 1], prims_R[..., 3]
    c_s_L, c_s_R = hydro.c_s(prims_L, *args), hydro.c_s(prims_R, *args)
    
    U_L, U_R = U_from_prim(hydro, prims_L, *args), U_from_prim(hydro, prims_R, *args)
    e_L, e_R = U_L[..., 3], U_R[..., 3]
    
    F_L, F_R = F_from_prim(hydro, prims_L, *args), F_from_prim(hydro, prims_R, *args)

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
    F = jnp.where(case_2, F_star_x1(F_L, S_L, S_M, prims_L, U_L), F)
    F = jnp.where(case_3, F_star_x1(F_R, S_R, S_M, prims_R, U_R), F)
    F = jnp.where(case_4, F_R, F)

    return F


def hllc_flux_x2(hydro, prims_L: ArrayLike, prims_R: ArrayLike, *args) -> Array:
    """
            HLLC algorithm adapted from Robert Caddy
            https://robertcaddy.com/posts/HLLC-Algorithm/
    """
    rho_L, v_L, p_L = prims_L[..., 0], prims_L[..., 2], prims_L[..., 3]
    rho_R, v_R, p_R = prims_R[..., 0], prims_R[..., 2], prims_R[..., 3]
    c_s_L, c_s_R = hydro.c_s(prims_L, *args), hydro.c_s(prims_R, *args)
    
    U_L, U_R = U_from_prim(hydro, prims_L, *args), U_from_prim(hydro, prims_R, *args)
    e_L, e_R = U_L[..., 3], U_R[..., 3]
    
    G_L, G_R = G_from_prim(hydro, prims_L, *args), G_from_prim(hydro, prims_R, *args)

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
    G = jnp.where(case_2, F_star_x2(G_L, S_L, S_M, prims_L, U_L), G)
    G = jnp.where(case_3, F_star_x2(G_R, S_R, S_M, prims_R, U_R), G)
    G = jnp.where(case_4, G_R, G)

    return G

def plm_grad(yl, y0, yr, theta):
    return minmod(theta * (y0 - yl), 0.5 * (yr - yl), theta * (yr - y0))

def shear_strain(gx, gy, dx, dy):
    sxx = 4.0 / 3.0 * gx[..., 1] / dx - 2.0 / 3.0 * gy[..., 2] / dy
    syy =-2.0 / 3.0 * gx[..., 1] / dx + 4.0 / 3.0 * gy[..., 2] / dy
    sxy = 1.0 / 1.0 * gx[..., 2] / dx + 1.0 / 1.0 * gy[..., 1] / dy
    syx = sxy
    
    return jnp.array([sxx, sxy, syx, syy])


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
    
    prims = get_prims(hydro, U, lattice.X1, lattice.X2, t)
    prims = add_ghost_cells(prims, g, axis=1)
    prims = add_ghost_cells(prims, g, axis=0)
    prims = apply_bcs(lattice, prims)

    if hydro.PLM():
        X1_L = X1[(g-1):-(g+1), g:-g]
        X1_C = X1[g:-g, g:-g]
        X1_R = X1[(g+1):-(g-1), g:-g]
        X2_L = X2[g:-g, (g-1):-(g+1)]
        X2_C = X2[g:-g, g:-g]
        X2_R = X2[g:-g, (g+1):-(g-1)]
        
        theta = hydro.theta_PLM()

        prims_cc = prims[g:-g, g:-g]
        prims_ki = prims[:-(g+2), g:-g]
        prims_li = prims[(g-1):-(g+1), g:-g]
        prims_ri = prims[(g+1):-(g-1), g:-g]
        prims_ti = prims[(g+2):, g:-g]
        
        prims_kj = prims[g:-g, :-(g+2)]
        prims_lj = prims[g:-g, (g-1):-(g+1)]
        prims_rj = prims[g:-g, (g+1):-(g-1)]
        prims_tj = prims[g:-g, (g+2):]
        
        prims_ll = prims[(g-1):-(g+1), (g-1):-(g+1)]
        prims_lr = prims[(g-1):-(g+1), (g+1):-(g-1)]
        prims_rr = prims[(g+1):-(g-1), (g+1):-(g-1)]
        prims_rl = prims[(g+1):-(g-1), (g-1):-(g+1)]
        
        # left cell interface (i-1/2)
        # left-biased state
        gxli = plm_grad(prims_ki, prims_li, prims_cc, theta)
        prims_lim = prims_li + 0.5 * gxli
        # right-biased state
        gxcc = plm_grad(prims_li, prims_cc, prims_ri, theta)
        prims_lip = prims_cc - 0.5 * gxcc
        
        # right cell interface (i+1/2)
        # left-biased state
        prims_rim = prims_cc + 0.5 * gxcc
        
        # right-biased state
        gxri = plm_grad(prims_cc, prims_ri, prims_ti, theta)
        prims_rip = prims_ri - 0.5 * gxri

        if hydro.solver() == "hll":
            F_l, F_r = hll_flux_x1(hydro, prims_lim, prims_lip, X1_L, X2_C, t), hll_flux_x1(hydro, prims_rim, prims_rip, X1_R, X2_C, t)
        elif hydro.solver() == "hllc":
            F_l, F_r = hllc_flux_x1(hydro, prims_lim, prims_lip, X1_L, X2_C, t), hllc_flux_x1(hydro, prims_rim, prims_rip, X1_R, X2_C, t)

        # left cell interface (j-1/2)
        # left-biased state
        gylj = plm_grad(prims_kj, prims_lj, prims_cc, theta)
        prims_ljm = prims_lj + 0.5 * gylj
        # right-biased state
        gycc = plm_grad(prims_lj, prims_cc, prims_rj, theta)
        prims_ljp = prims_cc - 0.5 * gycc

        # right cell interface (j+1/2)
        # left-biased state
        prims_rjm = prims_cc + 0.5 * gycc
        # right-biased state
        gyrj = plm_grad(prims_cc, prims_rj, prims_tj, theta)
        prims_rjp = prims_rj - 0.5 * gyrj

        if hydro.solver() == "hll":
            G_l, G_r = hll_flux_x2(hydro, prims_ljm, prims_ljp, X1_C, X2_L, t), hll_flux_x2(hydro, prims_rjm, prims_rjp, X1_C, X2_R, t)
        elif hydro.solver() == "hllc":
            G_l, G_r = hllc_flux_x2(hydro, prims_ljm, prims_ljp, X1_C, X2_L, t), hllc_flux_x2(hydro, prims_rjm, prims_rjp, X1_C, X2_R, t)
        
        # compute additional PLM gradients
        gyli = plm_grad(prims_ll, prims_li, prims_lr, theta)
        gyri = plm_grad(prims_rl, prims_ri, prims_rr, theta)
        gxlj = plm_grad(prims_ll, prims_lj, prims_rl, theta)
        gxrj = plm_grad(prims_lr, prims_rj, prims_rr, theta)
    else:
        X1_L = X1[(g-1):-(g+1), g:-g]
        X1_C = X1[g:-g, g:-g]
        X1_R = X1[(g+1):, g:-g]
        X2_L = X2[g:-g, (g-1):-(g+1)]
        X2_C = X2[g:-g, g:-g]
        X2_R = X2[g:-g, (g+1):]
        
        prims_cc = prims[g:-g, g:-g]
        prims_li = prims[(g-1):-(g+1), g:-g]
        prims_ri = prims[(g+1):, g:-g]
        
        prims_lj = prims[g:-g, (g-1):-(g+1)]
        prims_rj = prims[g:-g, (g+1):]
        
        prims_ll = prims[(g-1):-(g+1), (g-1):-(g+1)]
        prims_lr = prims[(g-1):-(g+1), (g+1):]
        prims_rr = prims[(g+1):, (g+1):]
        prims_rl = prims[(g+1):, (g-1):-(g+1)]
        
        if hydro.solver() == "hll":
            F_l = hll_flux_x1(hydro, prims_li, prims_cc, X1_L, X2_C, t)
            F_r = hll_flux_x1(hydro, prims_cc, prims_ri, X1_R, X2_C, t)
            
            G_l = hll_flux_x2(hydro, prims_lj, prims_cc, X1_C, X2_L, t)
            G_r = hll_flux_x2(hydro, prims_cc, prims_rj, X1_C, X2_R, t)
        elif hydro.solver() == "hllc":
            F_l = hllc_flux_x1(hydro, prims_li, prims_cc, X1_L, X2_C, t)
            F_r = hllc_flux_x1(hydro, prims_cc, prims_ri, X1_R, X2_C, t)
            
            G_l = hllc_flux_x2(hydro, prims_lj, prims_cc, X1_C, X2_L, t)
            G_r = hllc_flux_x2(hydro, prims_cc, prims_rj, X1_C, X2_R, t)

        gxli = prims_cc - prims_li
        gxcc = (prims_ri - prims_li) / 2
        gxri = prims_ri - prims_cc
        
        gylj = prims_cc - prims_lj
        gycc = (prims_rj - prims_lj) / 2
        gyrj = prims_rj - prims_cc
        
        gyli = (prims_lr - prims_ll) / 2
        gyri = (prims_rr - prims_rl) / 2
        gxlj = (prims_rl - prims_ll) / 2
        gxrj = (prims_rr - prims_lr) / 2
            
    if hydro.nu():
        nu = hydro.nu()

        dx, dy = x1[1] - x1[0], x2[1] - x2[0]
        
        sli = shear_strain(gxli, gyli, dx, dy)
        sri = shear_strain(gxri, gyri, dx, dy)
        slj = shear_strain(gxlj, gylj, dy, dy)
        srj = shear_strain(gxrj, gyrj, dx, dy)
        scc = shear_strain(gxcc, gycc, dx, dy)
        
        F_l = F_l.at[..., 1].add(-0.5 * nu * (prims_li[..., 0] * sli[0] + prims_cc[..., 0] * scc[0]))
        F_l = F_l.at[..., 2].add(-0.5 * nu * (prims_li[..., 0] * sli[1] + prims_cc[..., 0] * scc[1]))
        F_r = F_r.at[..., 1].add(-0.5 * nu * (prims_cc[..., 0] * scc[0] + prims_ri[..., 0] * sri[0]))
        F_r = F_r.at[..., 2].add(-0.5 * nu * (prims_cc[..., 0] * scc[1] + prims_ri[..., 0] * sri[1]))
        G_l = G_l.at[..., 1].add(-0.5 * nu * (prims_lj[..., 0] * slj[2] + prims_cc[..., 0] * scc[2]))
        G_l = G_l.at[..., 2].add(-0.5 * nu * (prims_lj[..., 0] * slj[3] + prims_cc[..., 0] * scc[3]))
        G_r = G_r.at[..., 1].add(-0.5 * nu * (prims_cc[..., 0] * scc[2] + prims_rj[..., 0] * srj[2]))
        G_r = G_r.at[..., 2].add(-0.5 * nu * (prims_cc[..., 0] * scc[3] + prims_rj[..., 0] * srj[3]))
            

    return F_l, F_r, G_l, G_r