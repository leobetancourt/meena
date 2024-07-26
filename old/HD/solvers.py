from HD.helpers import enthalpy, minmod, add_ghost_cells

from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    def __init__(self, hd, gamma, nu, num_g, coords, res, x1, x2, x1_interf, x2_interf, high_order=False):
        self.hd = hd
        self.gamma = gamma
        self.nu = nu
        self.num_g = num_g
        self.coords = coords
        self.x1, self.x2 = x1, x2
        self.x1_interf, self.x2_interf = x1_interf, x2_interf
        self.x1_g = np.concatenate([
            [self.x1[0] - 2 * (self.x1_interf[1] - self.x1_interf[0]),
             self.x1[0] - (self.x1_interf[1] - self.x1_interf[0])],
            self.x1,
            [self.x1[-1] + (self.x1_interf[-1] - self.x1_interf[-2]),
                self.x1[-1] + 2 * (self.x1_interf[-1] - self.x1_interf[-2])]
        ])
        self.x2_g = np.concatenate([
            [self.x2[0] - 2 * (self.x2_interf[1] - self.x2_interf[0]),
             self.x2[0] - (self.x2_interf[1] - self.x2_interf[0])],
            self.x2,
            [self.x2[-1] + (self.x2_interf[-1] - self.x2_interf[-2]),
                self.x2[-1] + 2 * (self.x2_interf[-1] - self.x2_interf[-2])]
        ])
        self.res_x1, self.res_x2 = res
        self.high_order = high_order
        self.F_l, self.F_r, self.G_l, self.G_r = None, None, None, None

    @abstractmethod
    def flux(self, F_L, F_R, U_L, U_R, x1=True):
        pass

    def L(self, U, F_l, F_r, G_l, G_r):
        rho, u, v, p = self.hd.get_prims()
        self.F_l, self.F_r, self.G_l, self.G_r = F_l, F_r, G_l, G_r
        
        if self.coords == "cartesian":
            dx1, dx2 = self.x1[1] - self.x1[0], self.x2[1] - self.x2[0]
            return - ((F_r - F_l) / dx1) - ((G_r - G_l) / dx2)
        elif self.coords == "polar":
            R_interf, _ = np.meshgrid(self.x1_interf, self.x2, indexing="ij")
            x1_l, x1_r = R_interf[:-1, :], R_interf[1:, :]
            dx1 = x1_r - x1_l
            dx2 = self.x2[1] - self.x2[0]
            R, _ = np.meshgrid(self.x1, self.x2, indexing="ij")
            R = R[:, :, np.newaxis]
            x1_l, x1_r = x1_l[:, :, np.newaxis], x1_r[:, :, np.newaxis]
            dx1 = dx1[:, :, np.newaxis]
            S = np.array([
                np.zeros_like(rho),
                (p / R[:, :, 0]) + (rho * v ** 2) / R[:, :, 0],
                - rho * u * v / R[:, :, 0],
                np.zeros_like(rho)
            ]).transpose(1, 2, 0)

            return - ((x1_r * F_r - x1_l * F_l) / (R * dx1)) - ((G_r - G_l) / (R * dx2)) + S

    # add polar coordinate support
    def PLM_states(self, U, x1=True):
        """Compute and return the reconstructed states and fluxes using the piecewise linear method.

            Returns: U_rl, U_rr, U_ll, U_lr, F_rl, F_rr, F_ll, F_lr
        """
        g = self.num_g
        theta = 1.5

        prims_C = np.asarray(self.hd.get_prims())
        if x1:
            prims_L = np.asarray(
                self.hd.get_prims(U[(g-1):-(g+1), g:-g, :]))
            prims_LL = np.asarray(self.hd.get_prims(U[(g-2):-(g+2), g:-g, :]))
            prims_R = np.asarray(
                self.hd.get_prims(U[(g+1):-(g-1), g:-g, :]))
            prims_RR = np.asarray(self.hd.get_prims(U[(g+2):, g:-g, :]))
        else:
            prims_L = np.asarray(
                self.hd.get_prims(U[g:-g, (g-1):-(g+1), :]))
            prims_LL = np.asarray(self.hd.get_prims(U[g:-g, (g-2):-(g+2), :]))
            prims_R = np.asarray(
                self.hd.get_prims(U[g:-g, (g+1):-(g-1), :]))
            prims_RR = np.asarray(self.hd.get_prims(U[g:-g, (g+2):, :]))

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
        F_ll, F_lr, F_rl, F_rr = self.hd.F_from_prim(prims_ll, x1), self.hd.F_from_prim(prims_lr, x1), self.hd.F_from_prim(prims_rl, x1), self.hd.F_from_prim(
            prims_rr, x1)
        U_ll, U_lr, U_rl, U_rr = self.hd.U_from_prim(prims_ll), self.hd.U_from_prim(
            prims_lr), self.hd.U_from_prim(prims_rl), self.hd.U_from_prim(prims_rr)

        return U_ll, U_lr, U_rl, U_rr, F_ll, F_lr, F_rl, F_rr

    def finite_difference(self, u, x1=True):
        g = self.num_g

        if self.coords == "cartesian":
            dx = self.x1[1] - self.x1[0]
            dy = self.x2[1] - self.x2[0]
            if x1:
                du = (u[(g):-(g-1), g:-g] - u[(g-1):-(g), g:-g]) / (dx)
            else:
                du = (u[g:-g, (g):-(g-1)] - u[g:-g, (g-1):-(g)]) / (dy)
        elif self.coords == "polar":
            dtheta = self.x2[1] - self.x2[0]
            if x1:
                X1, _ = np.meshgrid(self.x1_g[1:-1], self.x2, indexing="ij")
                dR = np.diff(X1, axis=0)
                du = np.diff(u[(g-1):-(g-1), g:-g], axis=0) / dR
            else:
                X1, _ = np.meshgrid(self.x1, self.x2_interf, indexing="ij")
                du = np.diff(u[g:-g, (g-1):-(g-1)], axis=1) / (X1 * dtheta)
        return du

    def viscosity(self, rho, u, v):
        g = self.num_g
        # compute viscous flux and add to F and G
        dudx = self.finite_difference(u, x1=True)
        dudy = self.finite_difference(u, x1=False)
        dvdx = self.finite_difference(v, x1=True)
        dvdy = self.finite_difference(v, x1=False)

        zero = np.zeros((self.res_x1, self.res_x2))

        rho_l = (rho[(g-1):-(g+1), g:-g] + rho[g:-g, g:-g]) / 2
        rho_r = (rho[g:-g, g:-g] + rho[(g+1):-(g-1), g:-g]) / 2
        Fv_l = -self.nu * np.array([
            zero,
            rho_l * dudx[:-1, :],
            rho_l * dvdx[:-1, :],
            zero
        ]).transpose((1, 2, 0))

        Fv_r = -self.nu * np.array([
            zero,
            rho_r * dudx[1:, :],
            rho_r * dvdx[1:, :],
            zero
        ]).transpose((1, 2, 0))

        rho_l = (rho[g:-g, (g-1):-(g+1)] + rho[g:-g, g:-g]) / 2
        rho_r = (rho[g:-g, g:-g] + rho[g:-g, (g+1):-(g-1)]) / 2
        Gv_l = -self.nu * np.array([
            zero,
            rho_l * dudy[:, :-1],
            rho_l * dvdy[:, :-1],
            zero
        ]).transpose((1, 2, 0))

        Gv_r = -self.nu * np.array([
            zero,
            rho_r * dudy[:, 1:],
            rho_r * dvdy[:, 1:],
            zero
        ]).transpose((1, 2, 0))

        return Fv_l, Fv_r, Gv_l, Gv_r

    def interface_flux(self, U):
        g = self.num_g

        X1, X2 = np.meshgrid(self.x1_g, self.x2_g, indexing="ij")

        if self.high_order:
            U_ll, U_lr, U_rl, U_rr, F_ll, F_lr, F_rl, F_rr = self.PLM_states(
                U, x1=True)
            # F_(i-1/2)
            F_l = self.flux(F_ll, F_lr, U_ll, U_lr)
            # F_(i+1/2)
            F_r = self.flux(F_rl, F_rr, U_rl, U_rr)

            U_ll, U_lr, U_rl, U_rr, G_ll, G_lr, G_rl, G_rr = self.PLM_states(
                U, x1=False)
            # G_(i-1/2)
            G_l = self.flux(G_ll, G_lr, U_ll, U_lr, x1=False)
            # G_(i+1/2)
            G_r = self.flux(G_rl, G_rr, U_rl, U_rr, x1=False)
        else:
            if self.hd.eos == "ideal":
                rho, u, v, p = self.hd.get_prims(U)
            elif self.hd.eos == "isothermal":
                rho, u, v, p = self.hd.get_prims(U, X1, X2)
            F = self.hd.F_from_prim((rho, u, v, p), X1, X2, x1=True)
            G = self.hd.F_from_prim((rho, u, v, p), X1, X2, x1=False)

            F_L = F[(g-1):-(g+1), g:-g, :]
            F_R = F[(g+1):-(g-1), g:-g, :]
            G_L = G[g:-g, (g-1):-(g+1), :]
            G_R = G[g:-g, (g+1):-(g-1), :]

            F_C = F[g:-g, g:-g, :]
            G_C = G[g:-g, g:-g, :]

            U_L = U[(g-1):-(g+1), g:-g, :]
            U_R = U[(g+1):-(g-1), g:-g, :]
            U_C = U[g:-g, g:-g, :]
            X1_L = X1[(g-1):-(g+1), g:-g]
            X1_R = X1[(g+1):-(g-1), g:-g]
            X1_C = X1[g:-g, g:-g]
            X2_L = X2[g:-g, (g-1):-(g+1)]
            X2_R = X2[g:-g, (g+1):-(g-1)]
            X2_C = X2[g:-g, g:-g]
            # F_(i-1/2)
            F_l = self.flux(F_L, F_C, U_L, U_C, X1_L, X1_C, X2_C, X2_C)
            # F_(i+1/2)
            F_r = self.flux(F_C, F_R, U_C, U_R, X1_C, X1_R, X2_C, X2_C)

            U_L = U[g:-g, (g-1):-(g+1), :]
            U_R = U[g:-g, (g+1):-(g-1), :]
            # G_(i-1/2)
            G_l = self.flux(G_L, G_C, U_L, U_C, X1_C,
                            X1_C, X2_L, X2_C, x1=False)
            # G_(i+1/2)
            G_r = self.flux(G_C, G_R, U_C, U_R, X1_C,
                            X1_C, X2_C, X2_R, x1=False)

            # add viscous flux to interface flux
            if self.nu:
                Fv_l, Fv_r, Gv_l, Gv_r = self.viscosity(rho, u, v)
                F_l += Fv_l
                F_r += Fv_r
                G_l += Gv_l
                G_r += Gv_r

        return F_l, F_r, G_l, G_r

    def solve(self, U):
        F_l, F_r, G_l, G_r = self.interface_flux(U)
        return self.L(U, F_l, F_r, G_l, G_r)


class HLLC(Solver):

    def F_star(self, F_k, S_k, S_M, U_k, x1=True):
        rho_k, vx_k, vy_k, p_k = self.hd.get_prims(U_k)
        E_k = self.hd.E(rho_k, p_k, vx_k, vy_k)
        v_k = vx_k if x1 else vy_k

        rho_star = rho_k * (S_k - v_k) / (S_k - S_M)
        p_star = p_k + rho_k * (v_k - S_k) * (v_k - S_M)
        rhov_star = (rho_k * v_k * (S_k - v_k) +
                     p_star - p_k) / (S_k - S_M)
        E_star = (E_k * (S_k - v_k) - p_k *
                  v_k + p_star * S_M) / (S_k - S_M)

        U_star = np.array([rho_star, rhov_star, rho_star * vy_k, E_star]) if x1 else np.array(
            [rho_star, rho_star * vx_k, rhov_star, E_star])
        U_star = np.transpose(U_star, (1, 2, 0))
        S_k = np.expand_dims(S_k, axis=-1)
        return F_k + S_k * (U_star - U_k)

    def flux(self, F_L, F_R, U_L, U_R, x1=True):
        """
            HLLC algorithm adapted from Robert Caddy
            https://robertcaddy.com/posts/HLLC-Algorithm/
        """
        rho_L, vx_L, vy_L, p_L = self.hd.get_prims(U_L)
        rho_R, vx_R, vy_R, p_R = self.hd.get_prims(U_R)
        E_L, E_R = self.hd.E(rho_L, p_L, vx_L, vy_L), self.hd.E(
            rho_R, p_R, vx_R, vy_R)
        v_L = vx_L if x1 else vy_L
        v_R = vx_R if x1 else vy_R

        R_rho = np.sqrt(rho_R / rho_L)
        H_L = enthalpy(rho_L, p_L, E_L)
        H_R = enthalpy(rho_R, p_R, E_R)
        H_t = (H_L + (H_R * R_rho)) / (1 + R_rho)  # H tilde
        v_t = (v_L + (v_R * R_rho)) / (1 + R_rho)
        c_t = np.sqrt((self.gamma - 1) * (H_t + (0.5 * v_t ** 2)))
        c_L, c_R = self.hd.c_s(p_L, rho_L), self.hd.c_s(p_R, rho_R)

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
        F[case_2] = self.F_star(F_L, S_L, S_M, U_L, x1)[case_2]
        F[case_3] = self.F_star(F_R, S_R, S_M, U_R, x1)[case_3]
        F[case_4] = F_R[case_4]

        return F


class HLL(Solver):

    # returns (lambda_plus, lambda_minus)
    def lambdas(self, U, X1, X2, x1=True):
        rho, u, v, p = self.hd.get_prims(U, X1, X2)
        if self.hd.eos == "ideal":
            cs = self.hd.c_s(p, rho)
        elif self.hd.eos == "isothermal":
            cs = self.hd.c_s(X1, X2)

        v = u if x1 else v
        return v + cs, v - cs

    # returns (alpha_p, alpha_m)
    def alphas(self, U_L, U_R, X1_L, X1_R, X2_L, X2_R, x1=True):
        lambda_L = self.lambdas(U_L, X1_L, X2_L, x1=x1)
        lambda_R = self.lambdas(U_R, X1_R, X2_R, x1=x1)
        # element-wise max()
        alpha_p = np.maximum(0, np.maximum(lambda_L[0], lambda_R[0]))
        alpha_m = np.maximum(0, np.maximum(-lambda_L[1], -lambda_R[1]))

        return alpha_p, alpha_m

    def flux(self, F_L, F_R, U_L, U_R, X1_L, X1_R, X2_L, X2_R, x1=True):
        a_p, a_m = self.alphas(U_L, U_R, X1_L, X1_R, X2_L, X2_R, x1=x1)
        # add dimension to match F, U arrays
        a_p = np.expand_dims(a_p, axis=-1)
        a_m = np.expand_dims(a_m, axis=-1)
        return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)
