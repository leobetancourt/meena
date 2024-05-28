from contextlib import nullcontext
import os

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


def E(gamma, rho, p, u, v):
    return (p / (gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))


def P(gamma, rho, u, v, E):
    return (gamma - 1) * (E - (0.5 * rho * (u ** 2 + v ** 2)))


def enthalpy(rho, p, E):
    return (E + p) / rho


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return (r, theta)


def polar_to_cartesian(r, theta):
    return (r * np.cos(theta), r * np.sin(theta))


def c_s(gamma, P, rho):
    return np.sqrt(gamma * P / rho)


class HLL_Solver:
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
    def F_HLL(self, F_L, F_R, U_L, U_R, x=True):
        a_p, a_m = self.alphas(U_L, U_R, x=x)

        return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

    # HLLC algorithm adapted from Robert Caddy
    # https://robertcaddy.com/posts/HLLC-Algorithm/
    def F_HLLC(self, F_L, F_R, U_L, U_R, x=True):
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

        def F_star(F_k, S_k, U_k):
            rho_k, vx_k, vy_k, p_k, E_k = self.get_vars(U_k)
            v_k = vx_k if x else vy_k

            rho_star = rho_k * (S_k - v_k) / (S_k - S_M)
            p_star = p_L + rho_L * (v_L - S_L) * (v_L - S_M)
            rhov_star = (rho_k * v_k * (S_k - v_k) +
                         p_star - p_k) / (S_k - S_M)
            E_star = (E_k * (S_k - v_k) - p_k *
                      v_k + p_star * S_M) / (S_k - S_M)

            U_star = np.array([rho_star, rhov_star, rho_star * vy_k, E_star]) if x else np.array(
                [rho_star, rho_star * vx_k, rhov_star, E_star])

            return F_k + np.multiply(S_k, U_star - U_k)

        if S_L > 0:
            return F_L
        elif S_L <= 0 and S_M >= 0:
            return F_star(F_L, S_L, U_L)
        if S_M <= 0 and S_R >= 0:
            return F_star(F_R, S_R, U_R)
        else:
            return F_R

    def L(self, F_L, F_R, G_L, G_R):
        return - ((F_R - F_L) / self.dx) - ((G_R - G_L) / self.dy)

    def solve(self, U, F, G):
        L_ = np.zeros_like(U)

        # compute HLL flux at each interface
        for i in range(1, len(U) - 1):
            for j in range(1, len(U[i]) - 1):
                F_L = self.F_HLLC(F[i - 1][j], F[i][j],
                                  U[i - 1][j], U[i][j])
                F_R = self.F_HLLC(F[i][j], F[i + 1][j],
                                  U[i][j], U[i + 1][j])

                G_L = self.F_HLLC(G[i][j - 1], G[i][j],
                                  U[i][j - 1], U[i][j], x=False)
                G_R = self.F_HLLC(G[i][j], G[i][j + 1],
                                  U[i][j], U[i][j + 1], x=False)

                # compute semi discrete L (including source term)
                L_[i][j] = self.L(F_L, F_R, G_L, G_R)

        return L_


class Boundary:
    OUTFLOW = "outflow"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"


class Sim_2D:
    def __init__(self, gamma=1.4, resolution=(100, 100),
                 xrange=(-1, 1), yrange=(-1, 1), order="first"):
        self.gamma = gamma

        self.res_x, self.res_y = resolution
        self.xmin, self.xmax = xrange
        self.ymin, self.ymax = yrange
        self.x = np.linspace(self.xmin, self.xmax,
                             num=self.res_x, endpoint=False)
        self.y = np.linspace(self.ymin, self.ymax,
                             num=self.res_y, endpoint=False)
        self.grid = np.array(
            [np.array([[_x, _y] for _y in self.y]) for _x in self.x])
        self.dx, self.dy = (self.xmax - self.xmin) / \
            self.res_x, (self.ymax - self.ymin) / self.res_y

        self.set_bcs()

        # conservative variable
        self.U = np.zeros((self.res_x, self.res_y, 4))
        # flux in x: F = (rho * u, rho * u^2 + P, rho * u * v, (E + P) * u)
        self.F = np.zeros((self.res_x, self.res_y, 4))
        # flux in y: G = (rho * v, rho * u * v, rho * v^2 + P, (E + P) * v)
        self.G = np.zeros((self.res_x, self.res_y, 4))

        # source term (function of U)
        self.S = None

        if order != "first" and order != "high":
            print("Invalid order provided: Must be 'first' or 'high'")

        self.order = order
        self.dt = self.dx
        self.cfl = 0.4
        self.solver = HLL_Solver(
            gamma, resolution, self.x, self.y, self.dx, self.dy)

    def get_vars(self):
        rho = self.U[:, :, 0]
        u, v, E = self.U[:, :, 1] / rho, self.U[:, :, 2] / rho, self.U[:, :, 3]
        p = P(self.gamma, rho, u, v, E)
        return rho, u, v, p, E

    def add_ghost_cells(self):
        # add ghost cells to the top and bottom boundaries
        self.U = np.hstack((self.U[:, 0:1, :], self.U, self.U[:, -1:, :]))

        # add ghost cells to the left and right boundaries
        self.U = np.vstack((self.U[0:1, :, :], self.U, self.U[-1:, :, :]))

    # bc_x = (left, right), bc_y = (bottom, top)
    def set_bcs(self, bc_x=(Boundary.OUTFLOW, Boundary.OUTFLOW), bc_y=(Boundary.OUTFLOW, Boundary.OUTFLOW)):
        self.bc_x = bc_x
        self.bc_y = bc_y

    def apply_bcs(self):
        # left
        if self.bc_x[0] == Boundary.OUTFLOW:
            self.U[0, :, :] = self.U[1, :, :]
        elif self.bc_x[0] == Boundary.REFLECTIVE:
            self.U[0, :, :] = self.U[1, :, :]
            self.U[0, :, 1] = -self.U[1, :, 1]  # invert x momentum
        elif self.bc_x[0] == Boundary.PERIODIC:
            self.U[0, :, :] = self.U[-2, :, :]

        # right
        if self.bc_x[1] == Boundary.OUTFLOW:
            self.U[-1, :, :] = self.U[-2, :, :]
        elif self.bc_x[1] == Boundary.REFLECTIVE:
            self.U[-1, :, :] = self.U[-2, :, :]
            self.U[-1, :, 1] = -self.U[-2, :, 1]  # invert x momentum
        elif self.bc_x[1] == Boundary.PERIODIC:
            self.U[-1, :, :] = self.U[1, :, :]

        # bottom
        if self.bc_y[0] == Boundary.OUTFLOW:
            self.U[:, 0, :] = self.U[:, 1, :]
        elif self.bc_y[0] == Boundary.REFLECTIVE:
            self.U[:, 0, :] = self.U[:, 1, :]
            self.U[:, 0, 2] = -self.U[:, 1, 2]  # invert y momentum
        elif self.bc_y[0] == Boundary.PERIODIC:
            self.U[:, 0, :] = self.U[:, -2, :]

        # top
        if self.bc_y[1] == Boundary.OUTFLOW:
            self.U[:, -1, :] = self.U[:, -2, :]
        elif self.bc_y[1] == Boundary.REFLECTIVE:
            self.U[:, -1, :] = self.U[:, -2, :]
            self.U[:, -1, 2] = -self.U[:, -2, 2]  # invert y momentum
        elif self.bc_y[1] == Boundary.PERIODIC:
            self.U[:, -1, :] = self.U[:, 1, :]

    def compute_flux(self):
        rho, u, v, p, E = self.get_vars()

        self.F = np.array([
            rho * u,
            rho * (u ** 2) + p,
            rho * u * v,
            (E + p) * u
        ]).transpose((1, 2, 0))  # transpose to match original shape (self.res_x, self.res_y, 4)

        self.G = np.array([
            rho * v,
            rho * u * v,
            rho * (v ** 2) + p,
            (E + p) * v
        ]).transpose((1, 2, 0))

    def compute_timestep(self):
        rho, u, v, p, E = self.get_vars()
        return self.cfl * \
            np.min(self.dx / (np.sqrt(self.gamma * p / rho) + np.sqrt(u**2 + v**2)))

    def first_order_step(self):
        self.dt = self.compute_timestep()

        if self.S:
            self.U = np.add(self.U, self.S(self.U) * (self.dt / 2))
        L = self.solver.solve(self.U, self.F, self.G)
        self.U = np.add(self.U, L * self.dt)
        if self.S:
            self.U = np.add(self.U, self.S(self.U) * (self.dt / 2))

    def high_order_step(self):
        """
        Third-order Runge-Kutta method
        """
        self.dt = self.compute_timestep()
        L = self.solver.solve(self.U, self.F, self.G)
        U_1 = np.add(self.U, L * self.dt)

        L_1 = self.solver.solve(U_1, self.F, self.G)
        U_2 = np.add(np.add((3/4) * self.U, (1/4) * U_1),
                     (1/4) * self.dt * L_1)

        L_2 = self.solver.solve(U_2, self.F, self.G)
        self.U = np.add(np.add((1/3) * self.U, (2/3) * U_2),
                        (2/3) * self.dt * L_2)

    def run(self, T, var="density", filename=None):
        t = 0
        fig = plt.figure()

        self.add_ghost_cells()

        if filename:
            # output video writer
            clear_frames = True
            fps = 24
            FFMpegWriter = animation.writers['ffmpeg']
            metadata = dict(title=filename, comment='')
            writer = FFMpegWriter(fps=fps, metadata=metadata)
            PATH = f"./videos/{filename}"
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            cm = writer.saving(fig, f"{PATH}/{var}.mp4", 100)
        else:
            cm = nullcontext()

        with cm:
            while t < T:
                self.apply_bcs()
                self.compute_flux()

                if self.order == "first":
                    self.first_order_step()
                elif self.order == "high":
                    self.high_order_step()

                if filename:
                    if clear_frames:
                        fig.clear()
                    self.plot(var)
                    writer.grab_frame()

                t = t + self.dt if (t + self.dt <= T) else T
                self.print_progress_bar(
                    t, T, suffix="complete", length=50)

        fig.clear()
        self.plot(var)
        plt.show()

    def initialize(self, U):
        self.U = U

    def add_source(self, source):
        self.S = source

    def sedov_blast(self, radius=0.1):
        r, _ = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        self.U[r < radius] = np.array([1, 0, 0, 10])
        self.U[r >= radius] = np.array([1, 0, 0, E(self.gamma, 1, 1e-4, 0, 0)])

    def implosion(self):
        r, _ = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        self.U[r < 0.2] = np.array([0.125, 0, 0, E(0.125, 0.14, 0, 0)])
        self.U[r >= 0.2] = np.array([1, 0, 0, E(1, 1, 0, 0)])

    # case 3 in Liska Wendroff
    def quadrants(self):
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        mid = (self.xmax - self.xmin) / 2
        rho_1, u_1, v_1, p_1 = 0.5323, 1.206, 0, 0.3
        rho_2, u_2, v_2, p_2 = 1.5, 0, 0, 1.5
        rho_3, u_3, v_3, p_3 = 0.138, 1.206, 1.206, 0.029
        rho_4, u_4, v_4, p_4 = 0.5323, 0, 1.206, 0.3
        self.U[(x < mid) & (y >= mid)] = np.array(
            [rho_1, rho_1 * u_1, rho_1 * v_1, E(self.gamma, rho_1, p_1, u_1, v_1)])
        self.U[(x >= mid) & (y >= mid)] = np.array(
            [rho_2, rho_2 * u_2, rho_2 * v_2, E(self.gamma, rho_2, p_2, u_2, v_2)])
        self.U[(x < mid) & (y < mid)] = np.array(
            [rho_3, rho_3 * u_3, rho_3 * v_3, E(self.gamma, rho_3, p_3, u_3, v_3)])
        self.U[(x >= mid) & (y < mid)] = np.array(
            [rho_4, rho_4 * u_4, rho_4 * v_4, E(self.gamma, rho_4, p_4, u_4, v_4)])

    # initial conditions from Deng, Boivin, Xiao
    # https://hal.science/hal-02100764/document
    def rayleigh_taylor(self):
        self.U = np.zeros((self.res_x, self.res_y, 4))
        g = -0.1

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                x = self.grid[i][j][0]
                y = self.grid[i][j][1]
                v = 0.0025 * (1-np.cos(4 * np.pi * x)) * \
                    (1-np.cos(4 * np.pi * y / 3))

                if y >= 0.75:
                    p = 2.5 + g * 2 * (y - 0.75)
                    self.U[i][j] = np.array(
                        [2, 0, 2 * v, E(self.gamma, 2, p, 0, v)])
                else:
                    p = 2.5 + g * 1 * (y - 0.75)
                    self.U[i][j] = np.array(
                        [1, 0, 1 * v, E(self.gamma, 1, p, 0, v)])

    def kelvin_helmholtz(self):
        self.U = np.zeros((self.res_x, self.res_y, 4))
        w0 = 0.1
        sigma = 0.05 / np.sqrt(2)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                x = self.grid[i][j][0]
                y = self.grid[i][j][1]

                if y > 0.75 or y < 0.25:
                    rho, u = 1, -0.5
                    v = w0*np.sin(4*np.pi*x) * (np.exp(-(y-0.25)**2 /
                                                       (2 * sigma**2)) + np.exp(-(y-0.75)**2/(2*sigma**2)))
                    p = 2.5
                    self.U[i][j] = np.array(
                        [1, -0.5, 0, E(self.gamma, rho, p, u, v)])
                else:
                    rho, u = 2, 0.5
                    v = w0*np.sin(4*np.pi*x) * (np.exp(-(y-0.25)**2 /
                                                       (2 * sigma**2)) + np.exp(-(y-0.75)**2/(2*sigma**2)))
                    p = 2.5
                    self.U[i][j] = np.array(
                        [rho, u, v, E(self.gamma, rho, p, u, v)])

    # call in a loop to print dynamic progress bar
    def print_progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # print new line on complete
        if iteration == total:
            print()

    def plot(self, var="density"):
        rho, u, v, p, E = self.get_vars()

        plt.cla()
        if var == "density":
            # plot density matrix (excluding ghost cells)
            c = plt.imshow(np.transpose(rho[1:-1, 1:-1]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        elif var == "pressure":
            c = plt.imshow(np.transpose(p[1:-1, 1:-1]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        elif var == "energy":
            c = plt.imshow(np.transpose(E[1:-1, 1:-1]), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        plt.colorbar(c)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(var)
        plt.pause(0.001)
