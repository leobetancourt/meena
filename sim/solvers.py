from sim.helpers import E, F_from_prim, U_from_prim, enthalpy, c_s, get_prims, minmod

from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    def __init__(self, gamma, res, num_g, x, y, dx, dy, high_order=False):
        self.gamma = gamma
        self.num_g = num_g
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.res_x, self.res_y = res
        self.high_order = high_order

    @abstractmethod
    def flux(self, F_L, F_R, U_L, U_R, x=True):
        pass

    def L(self, F_l, F_r, G_l, G_r):
        return - ((F_r - F_l) / self.dx) - ((G_r - G_l) / self.dy)

    def compute_flux(self, U):
        rho, u, v, p = get_prims(self.gamma, U)
        e = E(self.gamma, rho, p, u, v)

        F = np.array([
            rho * u,
            rho * (u ** 2) + p,
            rho * u * v,
            (e + p) * u
        ]).transpose((1, 2, 0))  # transpose to match original shape (self.res_x, self.res_y, 4)

        G = np.array([
            rho * v,
            rho * u * v,
            rho * (v ** 2) + p,
            (e + p) * v
        ]).transpose((1, 2, 0))

        return F, G

    # returns U_rl, U_rr, U_ll, U_lr, F_rl, F_rr, F_ll, F_lr
    def PLM_states(self, U, x=True):
        g = self.num_g
        theta = 1.5

        prims_C = np.asarray(get_prims(self.gamma, U[g:-g, g:-g, :]))
        if x:
            prims_L = np.asarray(
                get_prims(self.gamma, U[(g-1):-(g+1), g:-g, :]))
            prims_LL = np.asarray(get_prims(
                self.gamma, U[(g-2):-(g+2), g:-g, :]))
            prims_R = np.asarray(
                get_prims(self.gamma, U[(g+1):-(g-1), g:-g, :]))
            prims_RR = np.asarray(get_prims(
                self.gamma, U[(g+2):, g:-g, :]))
        else:
            prims_L = np.asarray(
                get_prims(self.gamma, U[g:-g, (g-1):-(g+1), :]))
            prims_LL = np.asarray(get_prims(
                self.gamma, U[g:-g, (g-2):-(g+2), :]))
            prims_R = np.asarray(
                get_prims(self.gamma, U[g:-g, (g+1):-(g-1), :]))
            prims_RR = np.asarray(get_prims(
                self.gamma, U[g:-g, (g+2):, :]))

        # left cell interface (i-1/2)
        # left-biased state
        prims_ll = prims_L + 0.5 * \
            minmod(theta * (prims_L - prims_LL), 0.5 *
                   (prims_C - prims_LL), theta * (prims_C - prims_L))

        # right-biased state
        prims_lr = prims_C - 0.5 * \
            minmod(theta * (prims_C - prims_L), 0.5 *
                   (prims_R - prims_L), theta * (prims_R - prims_C))

        # right cell interface (i+1/2)
        # left-biased state
        prims_rl = prims_C + 0.5 * \
            minmod(theta * (prims_C - prims_L), 0.5 *
                   (prims_R - prims_L), theta * (prims_R - prims_C))

        # right-biased state
        prims_rr = prims_R - 0.5 * \
            minmod(theta * (prims_R - prims_C), 0.5 *
                   (prims_RR - prims_C), theta * (prims_RR - prims_R))

        # construct F_rl, F_rr, F_ll, F_lr and U_rl, U_rr, U_ll, U_lr
        F_ll, F_lr, F_rl, F_rr = F_from_prim(self.gamma, prims_ll, x), F_from_prim(self.gamma, prims_lr, x), F_from_prim(self.gamma, prims_rl, x), F_from_prim(
            self.gamma, prims_rr, x)
        U_ll, U_lr, U_rl, U_rr = U_from_prim(self.gamma, prims_ll), U_from_prim(self.gamma, prims_lr), U_from_prim(self.gamma, prims_rl), U_from_prim(
            self.gamma, prims_rr)

        return U_ll, U_lr, U_rl, U_rr, F_ll, F_lr, F_rl, F_rr

    def interface_flux(self, U):
        g = self.num_g

        if self.high_order:
            U_ll, U_lr, U_rl, U_rr, F_ll, F_lr, F_rl, F_rr = self.PLM_states(
                U, x=True)
            # F_(i-1/2)
            F_l = self.flux(F_ll, F_lr, U_ll, U_lr)
            # F_(i+1/2)
            F_r = self.flux(F_rl, F_rr, U_rl, U_rr)

            U_ll, U_lr, U_rl, U_rr, G_ll, G_lr, G_rl, G_rr  = self.PLM_states(
                U, x=False)
            # G_(i-1/2)
            G_l = self.flux(G_ll, G_lr, U_ll, U_lr, x=False)
            # G_(i+1/2)
            G_r = self.flux(G_rl, G_rr, U_rl, U_rr, x=False)
        else:
            F, G = self.compute_flux(U)
            F_L = F[(g-1):-(g+1), g:-g, :]
            F_R = F[(g+1):-(g-1), g:-g, :]
            G_L = G[g:-g, (g-1):-(g+1), :]
            G_R = G[g:-g, (g+1):-(g-1), :]

            F_C = F[g:-g, g:-g, :]
            G_C = G[g:-g, g:-g, :]

            U_L = U[(g-1):-(g+1), g:-g, :]
            U_R = U[(g+1):-(g-1), g:-g, :]
            U_C = U[g:-g, g:-g, :]
            # F_(i-1/2)
            F_l = self.flux(F_L, F_C, U_L, U_C)
            # F_(i+1/2)
            F_r = self.flux(F_C, F_R, U_C, U_R)

            U_L = U[g:-g, (g-1):-(g+1), :]
            U_R = U[g:-g, (g+1):-(g-1), :]
            # G_(i-1/2)
            G_l = self.flux(G_L, G_C, U_L, U_C, x=False)
            # G_(i+1/2)
            G_r = self.flux(G_C, G_R, U_C, U_R, x=False)

        return F_l, F_r, G_l, G_r

    def solve(self, U):
        F_l, F_r, G_l, G_r = self.interface_flux(U)
        return self.L(F_l, F_r, G_l, G_r)


class HLLC(Solver):

    def F_star(self, F_k, S_k, S_M, U_k, x=True):
        rho_k, vx_k, vy_k, p_k = get_prims(self.gamma, U_k)
        E_k = E(self.gamma, rho_k, p_k, vx_k, vy_k)
        v_k = vx_k if x else vy_k

        rho_star = rho_k * (S_k - v_k) / (S_k - S_M)
        p_star = p_k + rho_k * (v_k - S_k) * (v_k - S_M)
        rhov_star = (rho_k * v_k * (S_k - v_k) +
                     p_star - p_k) / (S_k - S_M)
        E_star = (E_k * (S_k - v_k) - p_k *
                  v_k + p_star * S_M) / (S_k - S_M)

        U_star = np.array([rho_star, rhov_star, rho_star * vy_k, E_star]) if x else np.array(
            [rho_star, rho_star * vx_k, rhov_star, E_star])
        U_star = np.transpose(U_star, (1, 2, 0))
        S_k = np.expand_dims(S_k, axis=-1)
        return F_k + S_k * (U_star - U_k)

    def flux(self, F_L, F_R, U_L, U_R, x=True):
        """
            HLLC algorithm adapted from Robert Caddy
            https://robertcaddy.com/posts/HLLC-Algorithm/
        """
        rho_L, vx_L, vy_L, p_L = get_prims(self.gamma, U_L)
        rho_R, vx_R, vy_R, p_R = get_prims(self.gamma, U_R)
        E_L, E_R = E(self.gamma, rho_L, p_L, vx_L, vy_L), E(
            self.gamma, rho_R, p_R, vx_R, vy_R)
        v_L = vx_L if x else vy_L
        v_R = vx_R if x else vy_R

        R_rho = np.sqrt(rho_R / rho_L)
        H_L = enthalpy(rho_L, p_L, E_L)
        H_R = enthalpy(rho_R, p_R, E_R)
        H_t = (H_L + (H_R * R_rho)) / (1 + R_rho)  # H tilde
        v_t = (v_L + (v_R * R_rho)) / (1 + R_rho)
        c_t = np.sqrt((self.gamma - 1) * (H_t + (0.5 * v_t ** 2)))
        c_L, c_R = c_s(self.gamma, p_L, rho_L), c_s(self.gamma, p_R, rho_R)

        S_L = np.minimum(v_L - c_L, v_t - c_t)
        S_R = np.maximum(v_R + c_R, v_t + c_t)
        S_M = (rho_R * v_R * (S_R - v_R) - rho_L * v_L * (S_L - v_L) + p_L - p_R) \
            / (rho_R * (S_R - v_R) - rho_L * (S_L - v_L))

        F = np.empty_like(F_L)

        case_1 = S_L > 0
        case_2 = (S_L <= 0) & (S_M >= 0)
        case_3 = (S_M <= 0) & (S_R >= 0)
        case_4 = S_R < 0
        F[case_1] = F_L[case_1]
        F[case_2] = self.F_star(F_L, S_L, S_M, U_L, x)[case_2]
        F[case_3] = self.F_star(F_R, S_R, S_M, U_R, x)[case_3]
        F[case_4] = F_R[case_4]

        return F


class HLL(Solver):

    # returns (lambda_plus, lambda_minus)
    def lambdas(self, U, x=True):
        rho, u, v, p = get_prims(self.gamma, U)
        cs = c_s(self.gamma, p, rho)

        v = u if x else v
        return v + cs, v - cs

    # returns (alpha_p, alpha_m)
    def alphas(self, U_L, U_R, x=True):
        lambda_L = self.lambdas(U_L, x=x)
        lambda_R = self.lambdas(U_R, x=x)
        # element-wise max()
        alpha_p = np.maximum(0, np.maximum(lambda_L[0], lambda_R[0]))
        alpha_m = np.maximum(0, np.maximum(-lambda_L[1], -lambda_R[1]))

        return alpha_p, alpha_m

    def flux(self, F_L, F_R, U_L, U_R, x=True):
        a_p, a_m = self.alphas(U_L, U_R, x=x)
        # add dimension to match F, U arrays
        a_p = np.expand_dims(a_p, axis=-1)
        a_m = np.expand_dims(a_m, axis=-1)
        return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)
