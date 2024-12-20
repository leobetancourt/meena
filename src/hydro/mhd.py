import jax.numpy as jnp
import jax
from jax import Array, jit
from jax.typing import ArrayLike

from meena import Hydro, Lattice, Prims
from ..common.helpers import add_ghost_cells, apply_bcs_mhd, apply_bcs_B, enthalpy

import os
# jax.config.update('jax_log_compiles', True)

# shifts axes of x to the right (last axis becomes the first, all other axes are incremented to the next axis)
def roll(x: ArrayLike):
    axis_order = tuple(range(-1, x.ndim - 1))
    return jnp.transpose(x, axes=axis_order)

def c_A(prims: Prims):
    """
        Returns:
            Alfven speed
    """
    rho = prims[0]
    Bx, By, Bz = prims[5], prims[6], prims[7]
    return jnp.sqrt((Bx ** 2 + By ** 2 + Bz ** 2) / rho)

def c_fm(hydro: Hydro, prims: Prims, dir=0, *args):
    """
        Returns:
            Fast magnetosonic speed
    """
    rho = prims[0]
    c_s2 = hydro.c_s(prims, *args) ** 2
    c_A2 = c_A(prims) ** 2
    c_Ax2 = (prims[dir + 5] ** 2) / rho
    
    return jnp.sqrt(0.5 * (c_s2 + c_A2) + 0.5 * jnp.sqrt((c_s2 + c_A2) ** 2 - 4 * c_s2 * c_Ax2))

def c_sm(hydro: Hydro, prims: Prims, dir=0, *args):
    """
        Returns:
            Slow magnetosonic speed
    """
    rho = prims[0]
    c_s2 = hydro.c_s(prims, *args) ** 2
    c_A2 = c_A(prims) ** 2
    c_Ax2 = (prims[dir + 5] ** 2) / rho 
       
    return jnp.sqrt(0.5 * (c_s2 + c_A2) - 0.5 * jnp.sqrt((c_s2 + c_A2) ** 2 - 4 * c_s2 * c_Ax2))
    
def get_prims(hydro, U, *args):
    rho = U[..., 0]
    u, v, w = U[..., 1] / rho, U[..., 2] / rho, U[..., 3] / rho
    bx, by, bz = U[..., 5], U[..., 6], U[..., 7]
    p = hydro.P(roll(U), *args)
    return rho, u, v, w, p, bx, by, bz

def get_prims_from_state(hydro, state, t):
    U, X = state[..., :8], state[..., 8:]
    return get_prims(hydro, U, *roll(X), t)

def U_from_prim(hydro, prims, *args):
    rho, u, v, w, _, Bx, By, Bz = prims
    e = hydro.E(prims, *args)
    return jnp.array([
        rho,
        rho * u,
        rho * v,
        rho * w,
        e,
        Bx,
        By,
        Bz
    ]).transpose((1, 2, 3, 0))

def state_from_prim(hydro, prims, *args):
    rho, u, v, w, _, Bx, By, Bz = prims
    e = hydro.E(prims, *args)
    X1, X2, X3, _ = args
    return jnp.array([
        rho,
        rho * u,
        rho * v,
        rho * w,
        e,
        Bx,
        By,
        Bz,
        X1,
        X2,
        X3
    ]).transpose(1, 2, 3, 0)

def F_from_prim(hydro, prims, dir, *args):
    rho, u, v, w, p, Bx, By, Bz = prims
    B2 = Bx**2 + By**2 + Bz**2
    p_star = p + B2/2
    Bv = Bx*u + By*v + Bz*w
    e = hydro.E(prims, *args)
    zero = jnp.zeros_like(rho)
    
    return [
        jnp.array([
            rho*u,
            rho*u**2 + p_star - Bx**2,
            rho*u*v - Bx*By,
            rho*u*w - Bx*Bz,
            (e + p_star)*u - (Bv)*Bx,
            zero,
            By*u - Bx*v,
            Bz*u - Bx*w 
        ]),
        jnp.array([
            rho*v,
            rho*v*u - By*Bx,
            rho*v**2 + p_star - By**2,
            rho*v*w - By*Bz,
            (e + p_star)*v - (Bv)*By,
            Bx*v - By*u,
            zero,
            Bz*v - By*w
        ]),
        jnp.array([
            rho*w,
            rho*w*u - Bz*Bx,
            rho*w*v - Bz*By,
            rho*w**2 + p_star - Bz**2,
            (e + p_star)*w - (Bv)*Bz,
            Bx*w - Bz*u,
            By*w - Bz*v,
            zero
        ])
    ][dir].transpose((1, 2, 3, 0))


def lambdas(v: ArrayLike, c_s: ArrayLike) -> tuple[Array, Array]:
    return v + c_s, v - c_s


def alphas(v_L: ArrayLike, v_R: ArrayLike, c_s_L: ArrayLike, c_s_R: ArrayLike) -> tuple[Array, Array]:
    lambda_L = lambdas(v_L, c_s_L)
    lambda_R = lambdas(v_R, c_s_R)

    alpha_p = jnp.maximum(0, jnp.maximum(lambda_L[0], lambda_R[0]))
    alpha_m = jnp.maximum(0, jnp.maximum(-lambda_L[1], -lambda_R[1]))

    return alpha_p, alpha_m


def hll_flux(hydro: Hydro, dir: int, state_L: ArrayLike, state_R: ArrayLike, t: float) -> Array:
    U_L, U_R = state_L[..., :8], state_R[..., :8]
    X_L, X_R = state_L[..., 8:].transpose((-1, 0, 1, 2)), state_R[..., 8:].transpose((-1, 0, 1, 2))
    
    prims_L = get_prims(hydro, U_L, *X_L, t) # unpack each coordinate component
    prims_R = get_prims(hydro, U_R, *X_R, t)
    F_L = F_from_prim(hydro, prims_L, dir, *X_L, t)
    F_R = F_from_prim(hydro, prims_R, dir, *X_R, t)
    c_L = c_fm(hydro, prims_L, dir, *X_L, t)
    c_R = c_fm(hydro, prims_R, dir, *X_R, t)
    
    v_L, v_R = prims_L[1 + dir], prims_R[1 + dir]
    a_p, a_m = alphas(v_L, v_R, c_L, c_R)
    a_p, a_m = a_p[..., None], a_m[..., None]

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

def F_star(hydro: Hydro, dir: int, F_k: ArrayLike, state_k: ArrayLike, S_k: ArrayLike, S_M: ArrayLike, U_hll: ArrayLike, b_star: ArrayLike, t: float):
    U_k = state_k[..., :8]
    X_k = state_k[..., 8:].transpose((-1, 0, 1, 2))
    
    prims_k = get_prims(hydro, U_k, *X_k, t)
    rho_k, u_k, v_k, w_k, p_k, bx_k, by_k, bz_k = prims_k
    e_k = U_k[..., 4]
    vd_k = prims_k[dir + 1]
    bd_k = prims_k[dir + 5]
    
    first_term = (S_k - vd_k - (b_star[dir] * bd_k) / (rho_k * (S_k - vd_k)))
    by_star = (first_term * by_k - (b_star[dir] - bd_k) * v_k) / (S_k - S_M - (b_star[dir]**2) / (rho_k * (S_k - vd_k)))
    bz_star = (first_term * bz_k - (b_star[dir] - bd_k) * w_k) / (S_k - S_M - (b_star[dir]**2) / (rho_k * (S_k - vd_k)))
    
    rho_star = rho_k * (S_k - vd_k) / (S_k - S_M)
    momx_star = rho_star * S_M
    momy_star = rho_k * v_k * ((S_k - vd_k) / (S_k - S_M)) - (b_star[0]*by_star - bx_k * by_k) / (S_k - S_M)
    momz_star = rho_k * w_k * ((S_k - vd_k) / (S_k - S_M)) - (b_star[0]*bz_star - bx_k * bz_k) / (S_k - S_M)
    p_star = p_k + rho_k * (S_k - vd_k) * (S_M - vd_k) - bd_k**2 + b_star[0]**2

    b_dot_u_star = (b_star[0] * U_hll[..., 1] + b_star[1] * U_hll[..., 2] + b_star[2] * U_hll[..., 3]) / U_hll[..., 0]
    b_dot_u = bx_k * u_k + by_k * v_k + bz_k * w_k
    b_term = b_star[dir] * b_dot_u_star - bd_k * b_dot_u
    e_star = (e_k * (S_k - vd_k) - p_k * vd_k + p_star * S_M - b_term) / (S_k - S_M)
    
    U_star = jnp.array([rho_star, momx_star, momy_star, momz_star, e_star, b_star[dir], by_star, bz_star]).transpose((1, 2, 3, 0))
    S_k = S_k[..., None]
    return F_k + S_k * (U_star - U_k)

def hllc_flux(hydro: Hydro, dir: int, state_L: ArrayLike, state_R: ArrayLike, t: float) -> Array:
    U_L, U_R = state_L[..., :8], state_R[..., :8]
    X_L, X_R = state_L[..., 8:].transpose((-1, 0, 1, 2)), state_R[..., 8:].transpose((-1, 0, 1, 2))
    
    prims_L = get_prims(hydro, U_L, *X_L, t)
    rho_L, u_L, v_L, w_L, p_L, bx_L, by_L, bz_L = prims_L
    e_L = U_L[..., 4]
    prims_R = get_prims(hydro, U_R, *X_R, t)
    rho_R, u_R, v_R, w_R, p_R, bx_R, by_R, bz_R = prims_R
    e_R = U_R[..., 4]
    
    # fluxes
    F_L = F_from_prim(hydro, prims_L, dir, *X_L, t)
    F_R = F_from_prim(hydro, prims_R, dir, *X_R, t)
    
    # velocity in this flux direction
    vd_L, vd_R = prims_L[dir + 1], prims_R[dir + 1]
    # magnetic field in this flux direction
    bd_L, bd_R = prims_L[dir + 5], prims_R[dir + 5]
    
    R_rho = jnp.sqrt(prims_R[0] / prims_L[0])
    H_L = enthalpy(rho_L, p_L, e_L)
    H_R = enthalpy(rho_R, p_R, e_R)
    H_t = (H_L + (H_R * R_rho)) / (1 + R_rho)  # H tilde
    v_t = (vd_L + (vd_R * R_rho)) / (1 + R_rho)
    c_t = jnp.sqrt((hydro.gamma() - 1) * (H_t - (0.5 * v_t ** 2)))
    
    c_L = c_fm(hydro, prims_L, dir, *X_L, t)
    c_R = c_fm(hydro, prims_R, dir, *X_R, t)
    
    S_L = jnp.minimum(vd_L, vd_R) - jnp.maximum(c_L, c_R)
    S_R = jnp.maximum(vd_L, vd_R) + jnp.minimum(c_L, c_R)
    S_M = (rho_R * vd_R * (S_R - vd_R) - rho_L * vd_L * (S_L - vd_L) + p_L - p_R - bd_L**2 + bd_R**2) \
        / (rho_R * (S_R - vd_R) - rho_L * (S_L - vd_L))
        
    bx_star = (S_R * bx_R - S_L * bx_L) / (S_R - S_L)
    by_star = (S_R * by_R - S_L * by_L) / (S_R - S_L)
    bz_star = (S_R * bz_R - S_L * bz_L) / (S_R - S_L)
    b_star = jnp.array([bx_star, by_star, bz_star])

    U_star = (S_R[..., None] * U_R - S_L[..., None] * U_L - (F_R - F_L)) / (S_R[..., None] - S_L[..., None])
    U_hll = jnp.empty_like(U_L)
    case_1 = S_L > 0
    case_2 = (S_L <= 0) & (S_R >= 0)
    case_3 = S_R < 0
    case_1 = case_1[..., None]
    case_2 = case_2[..., None]
    case_3 = case_3[..., None]
    U_hll = jnp.where(case_1, U_L, U_hll)
    U_hll = jnp.where(case_2, U_star, U_hll)
    U_hll = jnp.where(case_3, U_R, U_hll)
    
    F = jnp.empty_like(F_L)
    case_1 = S_L >= 0
    case_2 = S_L <= S_M
    case_3 = S_M <= S_R
    case_4 = S_R <= 0
    case_1 = case_1[..., None]
    case_2 = case_2[..., None]
    case_3 = case_3[..., None]
    case_4 = case_4[..., None]
    
    F = jnp.where(case_1, F_L, F)
    F = jnp.where(case_2, F_star(
        hydro, dir, F_L, state_L, S_L, S_M, U_hll, b_star, t), F)
    F = jnp.where(case_3, F_star(
        hydro, dir, F_R, state_R, S_R, S_M, U_hll, b_star, t), F)
    F = jnp.where(case_4, F_R, F)

    return F

def dexi_dxj(i: int, dir: int, e_ref: ArrayLike, Fj: ArrayLike, F_contact: ArrayLike, dxi: float, flux_sign: int):
    l = tuple(slice(None, -1) if (k != i and k != dir) else slice(None) for k in range(3))
    r = tuple(slice(1, None) if (k != i and k != dir) else slice(None) for k in range(3))
    ll = l[:dir] + (slice(None, -1),) + l[dir+1:]
    rl = l[:dir] + (slice(1, None),) + l[dir+1:]
    lr = r[:dir] + (slice(None, -1),) + r[dir+1:]
    rr = r[:dir] + (slice(1, None),) + r[dir+1:]
    
    contact_l = tuple(slice(None, -1) if k == dir else slice(None) for k in range(3))
    contact_r = tuple(slice(1, None) if k == dir else slice(None) for k in range(3))
    F_contact = F_contact[l]
    v = F_contact[..., 0] # use density flux for the upwind check
    
    Fj = flux_sign * Fj[..., 5 + int(flux_sign == 1)] # get electric field from flux
    # e_ref = e_ref[..., i] # get exi component of reference emf

    dl1 = 2 * (Fj[rl] - e_ref[ll]) / dxi
    dl2 = 2 * (Fj[rr] - e_ref[lr]) / dxi
    dl = jnp.select([v[contact_l] > 0, v[contact_l] < 0], 
                         [dl1, dl2], default=0.5 * (dl1 + dl2))
    
    dr1 = 2 * (Fj[rl] - e_ref[rl]) / dxi
    dr2 = 2 * (Fj[rr] - e_ref[rr]) / dxi
    dr = jnp.select([v[contact_r] > 0, v[contact_r] < 0], 
                         [dr1, dr2], default=0.5 * (dr1 + dr2))
    
    return dl, dr

def PLM(hydro: Hydro, dir: int, state: ArrayLike, t: float):
    L = (slice(None),) + tuple(slice(None, -2) if k == dir else slice(1, -1) for k in range(3))
    C = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
    R = (slice(None),) + tuple(slice(2, None) if k == dir else slice(1, -1) for k in range(3))
    
    prims = jnp.asarray(get_prims_from_state(hydro, state, t))
    
    prims_L = prims[L]
    prims_C = prims[C]
    prims_R = prims[R]
    
    dp_L = prims_C - prims_L
    dp_R = prims_R - prims_C
    dp_C = 0.5 * (prims_R - prims_L)
    
    dpm = jnp.sign(dp_C) * jnp.minimum(2 * jnp.absolute(dp_L), jnp.minimum(2 * jnp.absolute(dp_R), jnp.absolute(dp_C)))
    
    l = tuple(slice(None, -1) if k == dir else slice(None) for k in range(3))
    r = tuple(slice(1, None) if k == dir else slice(None) for k in range(3))
    prims_l = prims.at[C].set(prims_C + 0.5 * dpm)
    prims_r = prims.at[C].set(prims_C - 0.5 * dpm)

    return prims_l, prims_r

def timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float):
    prims = get_prims(hydro, U, lattice.X1, lattice.X2, lattice.X3, t)
    velocities = jnp.stack([prims[1], prims[2], prims[3]])
    dX = jnp.stack([lattice.dX1, lattice.dX2, lattice.dX3])
    C = jnp.stack([c_fm(hydro, prims, 0, lattice.X1, lattice.X2, lattice.X3, t),
                  c_fm(hydro, prims, 1, lattice.X1, lattice.X2, lattice.X3, t),
                  c_fm(hydro, prims, 2, lattice.X1, lattice.X2, lattice.X3, t)])
    dt = dX / (jnp.abs(velocities) + C)
    dt = jnp.where(dt == 0, jnp.inf, dt)  # Replace zeros with infinity
    return hydro.cfl() * jnp.min(dt)

def add_ghost(arr, num_g):
    arr = add_ghost_cells(arr, num_g, axis=2)
    arr = add_ghost_cells(arr, num_g, axis=1)
    arr = add_ghost_cells(arr, num_g, axis=0)
    return arr

def VL_CT(hydro: Hydro, lattice: Lattice, U: ArrayLike, B: ArrayLike, t: float) -> tuple[Array, Array]:
    g = hydro.num_g()
    dt = timestep(hydro, lattice, U, t)
    dx, dy, dz = lattice.x1_intf[1] - lattice.x1_intf[0], lattice.x2_intf[1] - lattice.x2_intf[0], lattice.x3_intf[1] - lattice.x3_intf[0]

    X1, X2, X3 = lattice.X1_g, lattice.X2_g, lattice.X3_g
    X1_INTF, X2_INTF, X3_INTF = lattice.X1_INTF_g, lattice.X2_INTF_g, lattice.X3_INTF_g

    # add ghost cells to vector of conserved variables
    U = add_ghost(U, g)
    U = apply_bcs_mhd(lattice, U)
 
    Bx, By, Bz = B # face-centered components of B
    pad_mode = {"outflow": "edge", "reflective": "reflect", "periodic": "wrap"}
    Bx = jnp.pad(Bx, pad_width=((g, g), (g, g), (g, g)), mode=pad_mode[lattice.bc_x1[0]])
    By = jnp.pad(By, pad_width=((g, g), (g, g), (g, g)), mode=pad_mode[lattice.bc_x2[0]])
    Bz = jnp.pad(Bz, pad_width=((g, g), (g, g), (g, g)), mode=pad_mode[lattice.bc_x3[0]])

    # create state vector [*U, *X] which stores both conservative variables and coordinates at each zone
    state = jnp.zeros((*X1.shape, U.shape[-1] +  3))
    state = state.at[..., :8].set(U)
    state = state.at[..., 8].set(X1)
    state = state.at[..., 9].set(X2)
    state = state.at[..., 10].set(X3)

    # left, center and right slicing
    l = slice(None, -1)
    c = slice(1, -1)
    r = slice(1, None)
    
    flux = hll_flux
    if hydro.solver() == "hllc":
        flux = hllc_flux
    
    # STEP 1: Construct first-order upwind fluxes
    # set longitudinal component of B field to face-centered value
    state_l = jnp.pad(state, pad_width=((1, 0), (0, 0), (0, 0), (0, 0)), mode="edge")[l]
    F_l = flux(hydro, 0, state_l, state, t)
    state_l = jnp.pad(state, pad_width=((0, 0), (1, 0), (0, 0), (0, 0)), mode="edge")[:, l]
    G_l = jnp.zeros_like(state_l[..., :8]) # flux(hydro, 1, state_l, state, t)
    state_l = jnp.pad(state, pad_width=((0, 0), (0, 0), (1, 0), (0, 0)), mode="edge")[:, :, l]
    H_l = jnp.zeros_like(state_l[..., :8]) # flux(hydro, 2, state_l, state, t)
    
    # STEP 2: Calculate CT electric fields at cell-corners
    # rho = U[..., 0]
    # u, v, w = U[..., 1] / rho, U[..., 2] / rho, U[..., 3] / rho
    # bx, by, bz = U[..., 5], U[..., 6], U[..., 7]
    
    # dez_dy_l, dez_dy_r = dexi_dxj(2, 1, v*bx - u*by, G_l, F_l, dy, 1)
    # dez_dx_l, dez_dx_r = dexi_dxj(2, 0, v*bx - u*by, F_l, G_l, dx, -1)
    
    # dey_dx_l, dey_dx_r = dexi_dxj(1, 0, u*bz - w*bx, F_l, H_l, dx, 1) # why the extra negative here
    # dey_dz_l, dey_dz_r = dexi_dxj(1, 2, u*bz - w*bx, H_l, F_l, dz, -1)
    
    # dex_dy_l, dex_dy_r = dexi_dxj(0, 1, w*by - v*bz, G_l, H_l, dy, -1)
    # dex_dz_l, dex_dz_r = dexi_dxj(0, 2, w*by - v*bz, H_l, G_l, dz, 1)
        
    # ez = 0.25 * (-F_l[r, r, :, 5] - F_l[r, l, :, 5] + G_l[r, r, :, 6] + G_l[l, r, :, 6]) \
    #     + (dy / 8) * (dez_dy_r - dez_dy_l) + (dx / 8) * (dez_dx_r - dez_dx_l)
    
    # ey = 0.25 * (F_l[r, :, r, 6] + F_l[r, :, l, 6] - H_l[r, :, r, 5] - H_l[l, :, r, 5]) \
    #     + (dx / 8) * (dey_dx_r - dey_dx_l) + (dz / 8) * (dey_dz_r - dey_dz_l)
    
    # ex = 0.25 * (-G_l[:, r, r, 5] - G_l[:, r, l, 5] + H_l[:, r, r, 6] + H_l[:, l, r, 6]) \
    #     + (dy / 8) * (dex_dy_r - dex_dy_l) + (dz / 8) * (dex_dz_r - dex_dz_l)
    
    # STEP 3: Update cell-centered hydro variables and face-centered magnetic fields for one-half time step
    U = U.at[c, c, c].add(1 * dt * (-(F_l[2:, c, c] - F_l[c, c, c]) / dx - (G_l[c, 2:, c] - G_l[c, c, c]) / dy - (H_l[c, c, 2:] - H_l[c, c, c]) / dz))
    # Bx_half = Bx.at[c, c, c].add(1 * dt * (-(1 / dy) * (ez[:, r, c] - ez[:, l, c]) + (1 / dz) * (ey[:, c, r] - ey[:, c, l])))
    # By_half = By.at[c, c, c].add(1 * dt * ((1 / dx) * (ez[r, :, c] - ez[l, :, c]) - (1 / dz) * (ex[c, :, r] - ex[c, :, l])))
    # Bz_half = Bz.at[c, c, c].add(1 * dt * (-(1 / dx) * (ey[r, c] - ey[l, c]) + (1 / dy) * (ex[c, r] - ex[c, l])))
    
    # # STEP 4: Compute cell-centered magnetic field at half time step
    # bx_half = 0.5 * (Bx_half[r] + Bx_half[l])
    # by_half = 0.5 * (By_half[:, r] + By_half[:, l])
    # bz_half = 0.5 * (Bz_half[:, :, r] + Bz_half[:, :, l])
    # U_half = U_half.at[c, c, c, 5].set(bx_half[c, c, c])
    # U_half = U_half.at[c, c, c, 6].set(by_half[c, c, c])
    # U_half = U_half.at[c, c, c, 7].set(bz_half[c, c, c])
    
    # # STEP 5: Compute left- and right-state primitives at half time step at cell interfaces using PLM reconstruction
    # state_half = state.at[..., :8].set(U_half)
    # prims_xl, prims_xr = PLM(hydro, 0, state_half, t)
    # prims_yl, prims_yr = PLM(hydro, 1, state_half, t)
    # prims_zl, prims_zr = PLM(hydro, 2, state_half, t)

    # # STEP 6: Construct 1D fluxes at interfaces in all three dimensions
    # state_xl, state_xr = state_from_prim(hydro, prims_xl, X1_INTF[r], X2, X3, t), state_from_prim(hydro, prims_xr, X1_INTF[l], X2, X3, t)
    # state_xl, state_xr = state_xl.at[..., 5].set(Bx_half[r]), state_xr.at[..., 5].set(Bx_half[l]) # set longitudinal component of B at interfaces
    # F_l = hll_flux_x(hydro, state_xl, state_xr, t)
    
    # state_yl, state_yr = state_from_prim(hydro, prims_yl, X1, X2_INTF[:, r], X3, t), state_from_prim(hydro, prims_yr, X1, X2_INTF[:, l], X3, t)
    # state_yl, state_yr = state_yl.at[..., 6].set(By_half[:, r]), state_yr.at[..., 6].set(By_half[:, l])
    # G_l = hll_flux_y(hydro, state_yl, state_yr, t)
    
    # state_zl, state_zr = state_from_prim(hydro, prims_zl, X1, X2, X3_INTF[:, :, r], t), state_from_prim(hydro, prims_zr, X1, X2, X3_INTF[:, :, l], t)
    # state_zl, state_zr = state_zl.at[..., 7].set(Bz_half[:, :, r]), state_zr.at[..., 7].set(Bz_half[:, :, l])
    # H_l = hll_flux_z(hydro, state_zl, state_zr, t)
    
    # # STEP 7: Calculate CT electric fields at cell corners using face-centered fluxes computed in step 6 and reference fields from the half step
    # rho = U_half[..., 0]
    # u, v, w = U_half[..., 1] / rho, U_half[..., 2] / rho, U_half[..., 3] / rho
    
    # dez_dy_l, dez_dy_r = dexi_dxj(2, 1, v*bx - u*by, G_l, F_l, dy, 1)
    # dez_dx_l, dez_dx_r = dexi_dxj(2, 0, v*bx - u*by, F_l, G_l, dx, -1)
    
    # dey_dx_l, dey_dx_r = dexi_dxj(1, 0, u*bz - w*bx, F_l, H_l, dx, 1) # why the extra negative here
    # dey_dz_l, dey_dz_r = dexi_dxj(1, 2, u*bz - w*bx, H_l, F_l, dz, -1)
    
    # dex_dy_l, dex_dy_r = dexi_dxj(0, 1, w*by - v*bz, G_l, H_l, dy, -1)
    # dex_dz_l, dex_dz_r = dexi_dxj(0, 2, w*by - v*bz, H_l, G_l, dz, 1)
        
    # ez = 0.25 * (-F_l[r, r, :, 5] - F_l[r, l, :, 5] + G_l[r, r, :, 6] + G_l[l, r, :, 6]) \
    #     + (dy / 8) * (dez_dy_r - dez_dy_l) + (dx / 8) * (dez_dx_r - dez_dx_l)
    
    # ey = 0.25 * (F_l[r, :, r, 6] + F_l[r, :, l, 6] - H_l[r, :, r, 5] - H_l[l, :, r, 5]) \
    #     + (dx / 8) * (dey_dx_r - dey_dx_l) + (dz / 8) * (dey_dz_r - dey_dz_l)
    
    # ex = 0.25 * (-G_l[:, r, r, 5] - G_l[:, r, l, 5] + H_l[:, r, r, 6] + H_l[:, l, r, 6]) \
    #     + (dy / 8) * (dex_dy_r - dex_dy_l) + (dz / 8) * (dex_dz_r - dex_dz_l)
    
    # # STEP 8: Update cell-centered hydro variables and face-centered magnetic fields for a full timestep
    # U = U.at[c, c, c].add(dt * (-(F_l[2:, c, c] - F_l[c, c, c]) / dx - (G_l[c, 2:, c] - G_l[c, c, c]) / dy - (H_l[c, c, 2:] - H_l[c, c, c]) / dz))
    # Bx = Bx.at[c, c, c].add(dt * (-(1 / dy) * (ez[:, r, c] - ez[:, l, c]) + (1 / dz) * (ey[:, c, r] - ey[:, c, l])))
    # By = By.at[c, c, c].add(dt * ((1 / dx) * (ez[r, :, c] - ez[l, :, c]) - (1 / dz) * (ex[c, :, r] - ex[c, :, l])))
    # Bz = Bz.at[c, c, c].add(dt * (-(1 / dx) * (ey[r, c] - ey[l, c]) + (1 / dy) * (ex[c, r] - ex[c, l])))

    # STEP 9: Compute cell-centered magnetic field
    bx = 0.5 * (Bx[r] + Bx[l])
    by = 0.5 * (By[:, r] + By[:, l])
    bz = 0.5 * (Bz[:, :, r] + Bz[:, :, l])
    
    no_ghost = (slice(g, -g), slice(g, -g), slice(g, -g))
    flux = F_l[no_ghost], G_l[no_ghost], H_l[no_ghost]
    # U = U.at[..., 5].set(bx)
    # U = U.at[..., 6].set(by)
    # U = U.at[..., 7].set(bz)
    B = (Bx[no_ghost], By[no_ghost], Bz[no_ghost])
    
    return U[no_ghost], B, flux, dt

