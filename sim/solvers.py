from sim.helpers import E, P, enthalpy, cartesian_to_polar, polar_to_cartesian, c_s

from abc import ABC, abstractmethod
import numpy as np

# abstract base class


class Solver(ABC):
    def __init__(self, gamma, res, x, y, dx, dy):
        self.gamma = gamma
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.res_x, self.res_y = res

    def get_vars(self, U):
        rho = U[0]
        u, v, E = U[1] / rho, U[2] / rho, U[3]
        p = P(self.gamma, rho, u, v, E)
        return rho, u, v, p, E

    @abstractmethod
    def flux(self, F_L, F_R, U_L, U_R, x=True):
        pass

    def L(self, F_L, F_R, G_L, G_R):
        return - ((F_R - F_L) / self.dx) - ((G_R - G_L) / self.dy)

    def solve(self, U, F, G):
        L_ = np.zeros_like(U)

        for i in range(1, len(U) - 1):
            for j in range(1, len(U[i]) - 1):
                F_L = self.flux(F[i - 1][j], F[i][j],
                                  U[i - 1][j], U[i][j])
                F_R = self.flux(F[i][j], F[i + 1][j],
                                  U[i][j], U[i + 1][j])

                G_L = self.flux(G[i][j - 1], G[i][j],
                                  U[i][j - 1], U[i][j], x=False)
                G_R = self.flux(G[i][j], G[i][j + 1],
                                  U[i][j], U[i][j + 1], x=False)

                # compute semi discrete L (including source term)
                L_[i][j] = self.L(F_L, F_R, G_L, G_R)

        return L_


class HLLC(Solver):

    def F_star(self, F_k, S_k, S_M, U_k, x=True):
        rho_k, vx_k, vy_k, p_k, E_k = self.get_vars(U_k)
        v_k = vx_k if x else vy_k

        rho_star = rho_k * (S_k - v_k) / (S_k - S_M)
        p_star = p_k + rho_k * (v_k - S_k) * (v_k - S_M)
        rhov_star = (rho_k * v_k * (S_k - v_k) +
                     p_star - p_k) / (S_k - S_M)
        E_star = (E_k * (S_k - v_k) - p_k *
                  v_k + p_star * S_M) / (S_k - S_M)

        U_star = np.array([rho_star, rhov_star, rho_star * vy_k, E_star]) if x else np.array(
            [rho_star, rho_star * vx_k, rhov_star, E_star])

        return F_k + np.multiply(S_k, U_star - U_k)

    # HLLC algorithm adapted from Robert Caddy
    # https://robertcaddy.com/posts/HLLC-Algorithm/
    def flux(self, F_L, F_R, U_L, U_R, x=True):
        rho_L, vx_L, vy_L, p_L, E_L = self.get_vars(U_L)
        rho_R, vx_R, vy_R, p_R, E_R = self.get_vars(U_R)
        v_L = vx_L if x else vy_L
        v_R = vx_R if x else vy_R

        R_rho = np.sqrt(rho_R / rho_L)
        H_L = enthalpy(rho_L, p_L, E_L)
        H_R = enthalpy(rho_R, p_R, E_R)
        H_t = (H_L + (H_R * R_rho)) / (1 + R_rho)  # H tilde
        v_t = (v_L + (v_R * R_rho)) / (1 + R_rho)
        c_t = np.sqrt((self.gamma - 1) * (H_t + (0.5 * v_t ** 2)))
        c_L, c_R = c_s(self.gamma, p_L, rho_L), c_s(self.gamma, p_R, rho_R)

        S_L = min(v_L - c_L, v_t - c_t)
        S_R = max(v_R + c_R, v_t + c_t)
        S_M = (rho_R * v_R * (S_R - v_R) - rho_L * v_L * (S_L - v_L) + p_L - p_R) \
            / (rho_R * (S_R - v_R) - rho_L * (S_L - v_L))

        if S_L > 0:
            return F_L
        elif S_L <= 0 and S_M >= 0:
            return self.F_star(F_L, S_L, S_M, U_L, x)
        if S_M <= 0 and S_R >= 0:
            return self.F_star(F_R, S_R, S_M, U_R, x)
        else:
            return F_R


class HLL(Solver):

    # returns (lambda_plus, lambda_minus)
    def lambdas(self, U, x=True):
        v_x, v_y = U[1] / U[0], U[2] / U[0]
        rho, E = U[0], U[-1]
        cs = c_s(self.gamma, P(self.gamma, rho, v_x, v_y, E), rho)

        v = v_x if x else v_y
        return (v + cs, v - cs)

    # returns (alpha_p, alpha_m)
    def alphas(self, U_L, U_R, x=True):
        lambda_L = self.lambdas(U_L, x=x)
        lambda_R = self.lambdas(U_R, x=x)
        alpha_p = max(0, lambda_L[0], lambda_R[0])
        alpha_m = max(0, -lambda_L[1], -lambda_R[1])

        return (alpha_p, alpha_m)

    # HLL flux
    def flux(self, F_L, F_R, U_L, U_R, x=True):
        a_p, a_m = self.alphas(U_L, U_R, x=x)

        return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)
