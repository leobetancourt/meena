import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from meena import Hydro, Lattice, Prims
from ..common.helpers import add_ghost_cells, apply_bcs_mhd, apply_bcs_B

import os
os.environ['JAX_LOG_COMPILES'] = '1'

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

def c_fm(hydro: Hydro, prims: Prims, *args):
    """
        Returns:
            Fast magnetosonic speed
    """
    Bx, By, Bz = prims[5], prims[6], prims[7]
    c_s2 = hydro.c_s(prims, *args) ** 2
    
    B2 = Bx**2 + By**2 + Bz**2
    
    return jnp.sqrt(0.5 * (c_s2 + B2 + jnp.sqrt((c_s2 + B2) ** 2 - 4 * c_s2 * Bx**2)))

def c_sm(hydro: Hydro, prims: Prims, *args):
    """
        Returns:
            Slow magnetosonic speed
    """
    Bx, By, Bz = prims[5], prims[6], prims[7]
    c_s2 = hydro.c_s(prims, *args) ** 2
    
    B2 = Bx**2 + By**2 + Bz**2
    
    return jnp.sqrt(0.5 * (c_s2 + B2 - jnp.sqrt((c_s2 + B2) ** 2 - 4 * c_s2 * Bx**2)))

    
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

    
def F_from_prim(hydro, prims, dir: int = 0, *args):
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


def hll_flux(hydro: Hydro, state_L: ArrayLike, state_R: ArrayLike, t: float, dir: int = 0) -> Array:
    U_L, U_R = state_L[..., :8], state_R[..., :8]
    X_L, X_R = roll(state_L[..., 8:]), roll(state_R[..., 8:])
    
    prims_L = get_prims(hydro, U_L, *X_L, t) # unpack each coordinate component
    prims_R = get_prims(hydro, U_R, *X_R, t)
    F_L = F_from_prim(hydro, prims_L, dir, *X_L, t)
    F_R = F_from_prim(hydro, prims_R, dir, *X_R, t)
    c_s_L = hydro.c_s(prims_L, *X_L, t)
    c_s_R = hydro.c_s(prims_R, *X_R, t)
    
    v_L, v_R = prims_L[dir+1], prims_R[dir+1]
    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)
    a_p, a_m = a_p[..., None], a_m[..., None]

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

def dexi_dxj(i: int, dir: int, e_ref: ArrayLike, Fj: ArrayLike, F_contact: ArrayLike, dxi: float):
    l = tuple(slice(None, -1) if (k != i and k != dir) else slice(None) for k in range(3))
    r = tuple(slice(1, None) if (k != i and k != dir) else slice(None) for k in range(3))
    ll = l[:dir] + (slice(None, -1),) + l[dir+1:]
    rl = l[:dir] + (slice(1, None),) + l[dir+1:]
    lr = r[:dir] + (slice(None, -1),) + r[dir+1:]
    rr = r[:dir] + (slice(1, None),) + r[dir+1:]
    
    contact_l = tuple(slice(None, -1) if k == dir else slice(None) for k in range(3))
    contact_r = tuple(slice(1, None) if k == dir else slice(None) for k in range(3))
    contact_idx = (set([0, 1, 2]) - {i, dir}).pop() # get appropriate contact direction (not i and not dir)
    v = F_contact[..., contact_idx+1]
    
    Fj = -Fj[..., 5+i] # get exi component of flux
    e_ref = e_ref[..., i] # get exi component of reference emf

    dl1 = 2 * (Fj[l] - e_ref[ll]) / dxi
    dl2 = 2 * (Fj[r] - e_ref[lr]) / dxi
    dl = jnp.select([v[contact_l] > 0, v[contact_l] < 0], 
                         [dl1, dl2], default=0.5 * (dl1 + dl2))
    
    dr1 = 2 * (e_ref[rl] - Fj[l]) / dxi
    dr2 = 2 * (e_ref[rr] - Fj[r]) / dxi
    dr = jnp.select([v[contact_r] > 0, v[contact_r] < 0], 
                         [dr1, dr2], default=0.5 * (dr1 + dr2))
    
    return dl, dr

def PLM(hydro: Hydro, dir: int, state: ArrayLike, t: float):
    L = tuple(slice(None, -2) if k == dir else slice(1, -1) for k in range(3))
    C = (slice(1, -1), slice(1, -1), slice(1, -1))
    R = tuple(slice(2, None) if k == dir else slice(1, -1) for k in range(3))
    
    prims_L = jnp.asarray(get_prims_from_state(hydro, state[L], t))
    prims_C = jnp.asarray(get_prims_from_state(hydro, state[C], t))
    prims_R = jnp.asarray(get_prims_from_state(hydro, state[R], t))
    
    dp_L = prims_C - prims_L
    dp_R = prims_R - prims_C
    dp_C = 0.5 * (prims_R - prims_L)
    
    dpm = jnp.sign(dp_C) * jnp.minimum(2 * jnp.absolute(dp_L), jnp.minimum(2 * jnp.absolute(dp_R), jnp.absolute(dp_C)))
    
    l = tuple(slice(None, -1) if k == dir else slice(None) for k in range(3))
    r = tuple(slice(1, None) if k == dir else slice(None) for k in range(3))
    prims_l = (prims_C + 0.5 * dpm)[(slice(None),) + l]
    prims_r = (prims_C - 0.5 * dpm)[(slice(None),) + r]

    return prims_l, prims_r

def timestep(hydro: Hydro, lattice: Lattice, U: ArrayLike, t: float):
    prims = get_prims(hydro, U, lattice.X1, lattice.X2, lattice.X3, t)
    u, v, w = prims[1], prims[2], prims[3]
    C = c_fm(hydro, prims, lattice.X1, lattice.X2, lattice.X3, t) # fast magnetosonic wave speed
    dt1 = jnp.min(lattice.dX1 / (jnp.abs(u) + C))
    dt2 = jnp.min(lattice.dX2 / (jnp.abs(v) + C))
    dt3 = jnp.min(lattice.dX3 / (jnp.abs(w) + C))
    return hydro.cfl() * jnp.minimum(dt1, jnp.minimum(dt2, dt3))

def add_ghost(arr, num_g):
    arr = add_ghost_cells(arr, num_g, axis=2)
    arr = add_ghost_cells(arr, num_g, axis=1)
    arr = add_ghost_cells(arr, num_g, axis=0)
    return arr

@jit
def f(x):
  return x

    
def VL_CT(hydro: Hydro, lattice: Lattice, U: ArrayLike, B: ArrayLike, t: float) -> tuple[Array, Array]:
    print(f._cache_size())
    g = hydro.num_g() # update hydro/lattice variable if regime is mhd
    dt = timestep(hydro, lattice, U, t)

    x1, x2, x3 = lattice.x1, lattice.x2, lattice.x3
    dx, dy, dz = x1[1] - x1[0], x2[1] - x2[0], x3[1] - x3[0]

    # add ghost cells to lattice
    # FIX: this assumes outflow BCS.
    x1_left = x1[0] - (x1[1] - x1[0]) * jnp.arange(g, 0, -1)
    x1_right = x1[-1] + (x1[-1] - x1[-2]) * jnp.arange(1, g + 1)
    x1_g = jnp.concatenate([x1_left, x1, x1_right])

    x2_left = x2[0] - (x2[1] - x2[0]) * jnp.arange(g, 0, -1)
    x2_right = x2[-1] + (x2[-1] - x2[-2]) * jnp.arange(1, g + 1)
    x2_g = jnp.concatenate([x2_left, x2, x2_right])
    
    x3_left = x3[0] - (x3[1] - x3[0]) * jnp.arange(g, 0, -1)
    x3_right = x3[-1] + (x3[-1] - x3[-2]) * jnp.arange(1, g + 1)
    x3_g = jnp.concatenate([x3_left, x3, x3_right])

    X1, X2, X3 = jnp.meshgrid(x1_g, x2_g, x3_g, indexing="ij")
    
    # add ghost cells to vector of conserved variables
    U = add_ghost(U, g)
    U = apply_bcs_mhd(lattice, U)
 
    Bx, By, Bz = B # face-centered components of B
    Bx, By, Bz = add_ghost(Bx, g), add_ghost(By, g), add_ghost(Bz, g)
    Bx, By, Bz = apply_bcs_B(lattice, g, Bx), apply_bcs_B(lattice, g, By), apply_bcs_B(lattice, g, Bz)

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
    
    # STEP 1: Construct first-order upwind fluxes
    # set longitudinal component of B field to face-centered value
    F_l = hll_flux(hydro, state[:-1].at[..., 5].set(Bx[c]), state[1:].at[..., 5].set(Bx[c]), t, dir=0)
    G_l  = hll_flux(hydro, state[:, :-1].at[..., 6].set(By[:, c]), state[:, 1:].at[..., 6].set(By[:, c]), t, dir=1)
    H_l = hll_flux(hydro, state[:, :, :-1].at[..., 7].set(Bz[:, :, c]), state[:, :, 1:].at[..., 7].set(Bz[:, :, c]), t, dir=2)
    
    # STEP 2: Calculate CT electric fields at cell-corners
    rho = U[..., 0]
    u, v, w = U[..., 1] / rho, U[..., 2] / rho, U[..., 3] / rho
    bx, by, bz = U[..., 5], U[..., 6], U[..., 7]
    V = jnp.stack([u, v, w], axis=-1)
    B = jnp.stack([bx, by, bz], axis=-1)
    e_ref = -jnp.cross(V, B, axisa=-1, axisb=-1)
    
    dez_dy_l, dez_dy_r = dexi_dxj(2, 1, e_ref, G_l, F_l, dy)
    dez_dx_l, dez_dx_r = dexi_dxj(2, 0, e_ref, F_l, G_l, dx)
    
    dey_dx_l, dey_dx_r = dexi_dxj(1, 0, e_ref, F_l, H_l, dx)
    dey_dz_l, dey_dz_r = dexi_dxj(1, 2, e_ref, H_l, F_l, dz)
    
    dex_dy_l, dex_dy_r = dexi_dxj(0, 1, e_ref, G_l, H_l, dy)
    dex_dz_l, dex_dz_r = dexi_dxj(0, 2, e_ref, H_l, G_l, dz)
    
    ez = 0.25 * (F_l[:, 1:, :, 7] + F_l[:, :-1, :, 7] + G_l[1:, :, :, 7] + G_l[:-1, :, :, 7]) \
        + (dy / 8) * (dez_dy_r - dez_dy_l) + (dx / 8) * (dez_dx_r - dez_dx_l)
    
    ey = 0.25 * (F_l[:, :, 1:, 6] + F_l[:, :, :-1, 6] + H_l[1:, :, :, 6] + H_l[:-1, :, :, 6]) \
        + (dx / 8) * (dey_dx_r - dey_dx_l) + (dz / 8) * (dey_dz_r - dey_dz_l)
    
    ex = 0.25 * (G_l[:, :, 1:, 5] + G_l[:, :, :-1, 5] + H_l[:, 1:, :, 5] + H_l[:, :-1, :, 5]) \
        + (dy / 8) * (dex_dy_r - dex_dy_l) + (dz / 8) * (dex_dz_r - dex_dz_l)
    
    # STEP 3: Update cell-centered hydro variables and face-centered magnetic fields for one-half time step
    U_half = U[c, c, c] + 0.5 * dt * (-(F_l[r, c, c] - F_l[l, c, c]) / dx - (G_l[c, r, c] - G_l[c, l, c]) / dy - (H_l[c, c, r] - H_l[c, c, l]) / dz)
    Bx_half = Bx[c, c, c] + 0.5 * dt * (-(1 / dy) * (ez[:, r, c] - ez[:, l, c]) + (1 / dz) * (ey[:, c, r] - ey[:, c, l]))
    By_half = By[c, c, c] + 0.5 * dt * ((1 / dx) * (ez[r, :, c] - ez[l, :, c]) - (1 / dz) * (ex[c, :, r] - ex[c, :, l]))
    Bz_half = Bz[c, c, c] + 0.5 * dt * (-(1 / dx) * (ey[r, c] - ey[l, c]) + (1 / dy) * (ex[c, r] - ex[c, l]))

    # STEP 4: Compute cell-centered magnetic field at half time step
    bx_half = 0.5 * (Bx_half[r] - Bx_half[l])
    by_half = 0.5 * (By_half[:, r] - By_half[:, l])
    bz_half = 0.5 * (Bz_half[:, :, r] - Bz_half[:, :, l])
    U_half = U_half.at[..., 5].set(bx_half)
    U_half = U_half.at[..., 6].set(by_half)
    U_half = U_half.at[..., 7].set(bz_half)
        
    # STEP 5: Compute left- and right-state primitives at half time step at cell interfaces using PLM reconstruction
    state = jnp.zeros((*X1[c, c, c].shape, (U.shape[-1]+X1.shape[-1])))
    state = state.at[..., :8].set(U_half)
    state = state.at[..., 8].set(X1[c, c, c])
    state = state.at[..., 9].set(X2[c, c, c])
    state = state.at[..., 10].set(X3[c, c, c])
    
    prims_xl, prims_xr = PLM(hydro, 0, state, t)
    prims_yl, prims_yr = PLM(hydro, 1, state, t)
    prims_zl, prims_zr = PLM(hydro, 2, state, t)

    # STEP 6: Construct 1D fluxes at interfaces in all three dimensions
    X1_INTF, X2, X3 = jnp.meshgrid(lattice.x1_intf, x2_g[(g-1):-(g-1)], x3_g[(g-1):-(g-1)], indexing="ij")
    state_xl, state_xr = state_from_prim(hydro, prims_xl, X1_INTF, X2, X3, t), state_from_prim(hydro, prims_xr, X1_INTF, X2, X3, t)
    state_xl, state_xr = state_xl.at[..., 5].set(Bx_half[2:-2, c, c]), state_xr.at[..., 5].set(Bx_half[2:-2, c, c]) # set longitudinal component of B at interfaces
    F_l = hll_flux(hydro, state_xl, state_xr, t, dir=0)
    X1, X2_INTF, X3 = jnp.meshgrid(x1_g[(g-1):-(g-1)], lattice.x2_intf, x3_g[(g-1):-(g-1)], indexing="ij")
    state_yl, state_yr = state_from_prim(hydro, prims_yl, X1, X2_INTF, X3, t), state_from_prim(hydro, prims_yr, X1, X2_INTF, X3, t)
    state_yl, state_yr = state_yl.at[..., 6].set(By_half[c, 2:-2, c]), state_yr.at[..., 6].set(By_half[c, 2:-2, c])
    G_l = hll_flux(hydro, state_yl, state_yr, t, dir=1)
    
    X1, X2, X3_INTF = jnp.meshgrid(x1_g[(g-1):-(g-1)], x2_g[(g-1):-(g-1)], lattice.x3_intf, indexing="ij")
    state_zl, state_zr = state_from_prim(hydro, prims_zl, X1, X2, X3_INTF, t), state_from_prim(hydro, prims_zr, X1, X2, X3_INTF, t)
    state_zl, state_zr = state_zl.at[..., 7].set(Bz_half[c, c, 2:-2]), state_zr.at[..., 7].set(Bz_half[c, c, 2:-2])
    H_l = hll_flux(hydro, state_zl, state_zr, t, dir=2)
        
    # STEP 7: Calculate CT electric fields at cell corners using face-centered fluxes computed in step 6 and reference fields from the half step
    rho = U_half[..., 0]
    u, v, w = U_half[..., 1] / rho, U_half[..., 2] / rho, U_half[..., 3] / rho
    V = jnp.stack([u, v, w], axis=-1)
    B = jnp.stack([bx_half, by_half, bz_half], axis=-1)
    e_ref = -jnp.cross(V[c, c, c], B[c, c, c], axisa=-1, axisb=-1)
    
    dez_dy_l, dez_dy_r = dexi_dxj(2, 1, e_ref, G_l, F_l, dy)
    dez_dx_l, dez_dx_r = dexi_dxj(2, 0, e_ref, F_l, G_l, dx)
    
    dey_dx_l, dey_dx_r = dexi_dxj(1, 0, e_ref, F_l, H_l, dx)
    dey_dz_l, dey_dz_r = dexi_dxj(1, 2, e_ref, H_l, F_l, dz)
    
    dex_dy_l, dex_dy_r = dexi_dxj(0, 1, e_ref, G_l, H_l, dy)
    dex_dz_l, dex_dz_r = dexi_dxj(0, 2, e_ref, H_l, G_l, dz)
    
    ez = 0.25 * (F_l[:, 1:, :, 7] + F_l[:, :-1, :, 7] + G_l[1:, :, :, 7] + G_l[:-1, :, :, 7]) \
        + (dy / 8) * (dez_dy_r - dez_dy_l) + (dx / 8) * (dez_dx_r - dez_dx_l)
    
    ey = 0.25 * (F_l[:, :, 1:, 6] + F_l[:, :, :-1, 6] + H_l[1:, :, :, 6] + H_l[:-1, :, :, 6]) \
        + (dx / 8) * (dey_dx_r - dey_dx_l) + (dz / 8) * (dey_dz_r - dey_dz_l)
    
    ex = 0.25 * (G_l[:, :, 1:, 5] + G_l[:, :, :-1, 5] + H_l[:, 1:, :, 5] + H_l[:, :-1, :, 5]) \
        + (dy / 8) * (dex_dy_r - dex_dy_l) + (dz / 8) * (dex_dz_r - dex_dz_l)
    
    # STEP 8: Update cell-centered hydro variables and face-centered magnetic fields for a full timestep
    U = U[g:-g, g:-g, g:-g] + dt * (-(F_l[r, c, c] - F_l[l, c, c]) / dx - (G_l[c, r, c] - G_l[c, l, c]) / dy - (H_l[c, c, r] - H_l[c, c, l]) / dz)
    Bx = Bx[g:-g, g:-g, g:-g] + dt * (-(1 / dy) * (ez[:, r, c] - ez[:, l, c]) + (1 / dz) * (ey[:, c, r] - ey[:, c, l]))
    By = By[g:-g, g:-g, g:-g] + dt * ((1 / dx) * (ez[r, :, c] - ez[l, :, c]) - (1 / dz) * (ex[c, :, r] - ex[c, :, l]))
    Bz = Bz[g:-g, g:-g, g:-g] + dt * (-(1 / dx) * (ey[r, c] - ey[l, c]) + (1 / dy) * (ex[c, r] - ex[c, l]))
    
    # STEP 9: Compute cell-centered magnetic field
    bx = 0.5 * (Bx[r] - Bx[l])
    by = 0.5 * (By[:, r] - By[:, l])
    bz = 0.5 * (Bz[:, :, r] - Bz[:, :, l])
    
    flux = F_l, G_l, H_l
    U = U.at[..., 5].set(bx)
    U = U.at[..., 6].set(by)
    U = U.at[..., 7].set(bz)
    B = (Bx, By, Bz)
    
    return U, B, flux, dt

