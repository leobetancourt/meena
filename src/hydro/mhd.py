import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from ..common.helpers import add_ghost_cells, apply_bcs, minmod

def get_prims(hydro, U, *args):
    rho = U[..., 0]
    # can do something like U[..., 1:(hydro.dims+1)]
    v = U[..., 1:4] / rho
    B = U[..., 4:7]
    e = U[..., 7]
    cons = (rho, *(rho * v), e, *B)
    p = hydro.P(cons, *args)
    return rho, *v, p, *B


def U_from_prim(hydro, prims, *args):
    rho, u, v, w, p, Bx, By, Bz = prims
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


def F_from_prim(hydro, prims, *args):
    rho, u, v, w, p, Bx, By, Bz = prims
    B2 = Bx**2 + By**2 + Bz**2
    p_star = p + B2/2
    Bv = Bx*u + By*v + Bz*w
    e = hydro.E(prims, *args)
    return jnp.array([
        rho*u,
        rho*u**2 + p + B2/2 - Bx**2,
        rho*u*v - Bx*By,
        rho*u*w - Bx*Bz,
        (e + p_star)*u - (Bv)*Bx,
        0,
        By*u - Bx*v,
        Bz*u - Bx*w 
    ]).transpose((1, 2, 3, 0))


def G_from_prim(hydro, prims, *args):
    rho, u, v, w, p, Bx, By, Bz = prims
    B2 = Bx**2 + By**2 + Bz**2
    p_star = p + B2/2
    Bv = Bx*u + By*v + Bz*w
    e = hydro.E(prims, *args)
    return jnp.array([
        rho*v,
        rho*v*u - By*Bx,
        rho*v**2 + p_star - By**2,
        rho*v*w - By*Bz,
        (e + p_star)*v - (Bv)*By,
        Bx*v - By*u,
        0,
        Bz*v - By*w
    ]).transpose((1, 2, 3, 0))
    
def H_from_prim(hydro, prims, *args):
    rho, u, v, w, p, Bx, By, Bz = prims
    B2 = Bx**2 + By**2 + Bz**2
    p_star = p + B2/2
    Bv = Bx*u + By*v + Bz*w
    e = hydro.E(prims, *args)
    return jnp.array([
        rho*w,
        rho*w*u - Bz*Bx,
        rho*w*v - Bz*By,
        rho*w**2 + p_star - Bz**2,
        (e + p_star)*w - (Bv)*Bz,
        Bx*w - Bz*u,
        By*w - Bz*v,
        0
    ]).transpose((1, 2, 3, 0))

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


def hll_flux_x2(F_L: ArrayLike, F_R: ArrayLike, 
                U_L: ArrayLike, U_R: ArrayLike, 
                c_s_L: ArrayLike, c_s_R: ArrayLike) -> Array:
    rho_L, rho_R = U_L[..., 0], U_R[..., 0]
    v_L, v_R = U_L[..., 2] / rho_L, U_R[..., 2] / rho_R

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)
    a_p, a_m = a_p[..., None], a_m[..., None]

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

def hll_flux_x3(F_L: ArrayLike, F_R: ArrayLike, 
                U_L: ArrayLike, U_R: ArrayLike, 
                c_s_L: ArrayLike, c_s_R: ArrayLike) -> Array:
    rho_L, rho_R = U_L[..., 0], U_R[..., 0]
    v_L, v_R = U_L[..., 3] / rho_L, U_R[..., 3] / rho_R

    a_p, a_m = alphas(v_L, v_R, c_s_L, c_s_R)
    a_p, a_m = a_p[..., None], a_m[..., None]

    return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

def interface_flux(hydro, lattice, U: ArrayLike, t: float) -> tuple[Array, Array, Array, Array]:
    g = lattice.num_g
    x1, x2, x3 = lattice.x1, lattice.x2, lattice.x3
    
    return
    
    