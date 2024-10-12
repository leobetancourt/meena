from MHD.helpers import c_s, c_fm, E, P, get_prims, F_from_prim

from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    def __init__(self, gamma, res, num_g, x, y, z, dx, dy, dz):
        self.gamma = gamma
        self.num_g = num_g
        self.x, self.y, self.z = x, y, z
        self.dx, self.dy, self.dz = dx, dy, dz
        self.res_x, self.res_y, self.res_z = res

    @abstractmethod
    def flux(self, F_L, F_R, U_L, U_R, dir="x"):
        pass

    def L(self, F_l, F_r, G_l, G_r, H_l, H_r):
        return - ((F_r - F_l) / self.dx) - ((G_r - G_l) / self.dy) - ((H_r - H_l) / self.dz)

    def interface_flux(self, U):
        g = self.num_g

        rho, u, v, w, p, Bx, By, Bz = get_prims(self.gamma, U)
        F = F_from_prim(self.gamma, (rho, u, v, w, p, Bx, By, Bz), dir="x")
        G = F_from_prim(self.gamma, (rho, u, v, w, p, Bx, By, Bz), dir="y")
        H = F_from_prim(self.gamma, (rho, u, v, w, p, Bx, By, Bz), dir="z")

        F_L = F[(g-1):-(g+1), g:-g, g:-g, :]
        F_R = F[(g+1):-(g-1), g:-g, g:-g, :]
        G_L = G[g:-g, (g-1):-(g+1), g:-g, :]
        G_R = G[g:-g, (g+1):-(g-1), g:-g, :]
        H_L = H[g:-g, g:-g, (g-1):-(g+1), :]
        H_R = H[g:-g, g:-g, (g+1):-(g-1), :]

        F_C = F[g:-g, g:-g, g:-g, :]
        G_C = G[g:-g, g:-g, g:-g, :]
        H_C = H[g:-g, g:-g, g:-g, :]

        U_L = U[(g-1):-(g+1), g:-g, g:-g, :]
        U_R = U[(g+1):-(g-1), g:-g, g:-g, :]
        U_C = U[g:-g, g:-g, g:-g, :]
        # F_(i-1/2)
        F_l = self.flux(F_L, F_C, U_L, U_C, dir="x")
        # F_(i+1/2)
        F_r = self.flux(F_C, F_R, U_C, U_R, dir="x")

        U_L = U[g:-g, (g-1):-(g+1), g:-g, :]
        U_R = U[g:-g, (g+1):-(g-1), g:-g, :]
        # G_(i-1/2)
        G_l = self.flux(G_L, G_C, U_L, U_C, dir="y")
        # G_(i+1/2)
        G_r = self.flux(G_C, G_R, U_C, U_R, dir="y")

        U_L = U[g:-g, g:-g, (g-1):-(g+1), :]
        U_R = U[g:-g, g:-g, (g+1):-(g-1), :]
        # G_(i-1/2)
        H_l = self.flux(H_L, H_C, U_L, U_C, dir="z")
        # G_(i+1/2)
        H_r = self.flux(H_C, H_R, U_C, U_R, dir="z")
        
        return F_l, F_r, G_l, G_r, H_l, H_r

    def solve(self, U):
        F_l, F_r, G_l, G_r, H_l, H_r = self.interface_flux(U)
        return self.L(F_l, F_r, G_l, G_r, H_l, H_r)


class HLL(Solver):

    # returns (lambda_plus, lambda_minus)
    def lambdas(self, U, dir="x"):
        rho, u, v, w, p, Bx, By, Bz = get_prims(self.gamma, U)
        cfm = c_fm(self.gamma, p, rho, Bx, By, Bz, dir)
        if dir == "x": vel = u
        elif dir == "y": vel = v
        elif dir == "z": vel = w
        return vel + cfm, vel - cfm

    # returns (alpha_p, alpha_m)
    def alphas(self, U_L, U_R, dir="x"):
        lambda_L = self.lambdas(U_L, dir)
        lambda_R = self.lambdas(U_R, dir)
        # element-wise max()
        alpha_p = np.maximum(0, np.maximum(lambda_L[0], lambda_R[0]))
        alpha_m = np.maximum(0, np.maximum(-lambda_L[1], -lambda_R[1]))

        return alpha_p, alpha_m

    def flux(self, F_L, F_R, U_L, U_R, dir="x"):
        a_p, a_m = self.alphas(U_L, U_R, dir)
        # add dimension to match F, U arrays
        a_p = np.expand_dims(a_p, axis=-1)
        a_m = np.expand_dims(a_m, axis=-1)
        return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)