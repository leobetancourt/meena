from contextlib import nullcontext
import os

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


def E(gamma, rho, p, u, v):
    return (p / (gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))


def P(gamma, rho, u, v, E):
    return (gamma - 1) * (E - (0.5 * rho * (u ** 2 + v ** 2)))


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
        self.dt = dx  # timestep according to CFL condition

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

    def update_CFL(self, condition):
        if self.dt > condition:
            self.dt = condition

    # HLL flux
    def F_HLL(self, F_L, F_R, U_L, U_R):
        a_p, a_m = self.alphas(U_L, U_R, x=True)
        if max(a_p, a_m) > 0:
            self.update_CFL(self.dx / max(a_p, a_m))

        return (a_p * F_L + a_m * F_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

    def G_HLL(self, G_L, G_R, U_L, U_R):
        a_p, a_m = self.alphas(U_L, U_R, x=False)
        if max(a_p, a_m) > 0:
            self.update_CFL(self.dy / max(a_p, a_m))

        return (a_p * G_L + a_m * G_R - (a_p * a_m * (U_R - U_L))) / (a_p + a_m)

    def L(self, F_L, F_R, G_L, G_R):
        return - ((F_R - F_L) / self.dx) - ((G_R - G_L) / self.dy)

    def solve(self, U, F, G, S=None):
        self.dt = self.dx
        L_ = np.zeros((self.res_x, self.res_y, 4))

        # compute HLL flux at each interface
        for i in range(len(U)):
            for j in range(len(U[i])):
                F_L = self.F_HLL(F[i - 1 if i > 0 else 0][j], F[i][j],
                                 U[i - 1 if i > 0 else 0][j], U[i][j])
                F_R = self.F_HLL(F[i][j], F[i + 1 if i < len(U) - 1 else len(U) - 1][j],
                                 U[i][j], U[i + 1 if i < len(U) - 1 else len(U) - 1][j])

                G_L = self.G_HLL(G[i][j - 1 if j > 0 else 0], G[i][j],
                                 U[i][j - 1 if j > 0 else 0], U[i][j])
                G_R = self.G_HLL(G[i][j], G[i][j + 1 if j < len(U[i]) - 1 else len(U[i]) - 1],
                                 U[i][j], U[i][j + 1 if j < len(U[i]) - 1 else len(U[i]) - 1])

                # compute semi discrete L (including source term)
                L_[i][j] = self.L(F_L, F_R, G_L, G_R)

        if S is not None:
            return np.add(L_, S(U))
        else:
            return L_


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
            
        self.dt = self.dx
        self.solver = HLL_Solver(
            gamma, resolution, self.x, self.y, self.dx, self.dy)

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

        if var == "density":
            c = plt.imshow(np.transpose(rho), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        elif var == "pressure":
            c = plt.imshow(np.transpose(p), cmap="plasma", interpolation='nearest',
                           origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax])

        plt.colorbar(c)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(var)

    # returns rho, u, v, p, E
    def get_vars(self):
        rho = self.U[:, :, 0]
        u, v, E = self.U[:, :, 1] / rho, self.U[:, :, 2] / rho, self.U[:, :, 3]
        p = P(self.gamma, rho, u, v, E)
        return rho, u, v, p, E

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

    def first_order_step(self):
        L = self.solver.solve(self.U, self.F, self.G, self.S)
        self.dt = self.solver.dt / 2  # needs fixing (CFL at boundary)
        self.U = np.add(self.U, L * self.dt)

    def high_order_step(self):
        """
        Third-order Runge-Kutta method
        """
        L = self.solver.solve(self.U, self.F, self.G)
        self.dt = self.solver.dt / 2
        U_1 = np.add(self.U, L * self.dt)

        L_1 = self.solver.solve(U_1, self.F)
        U_2 = np.add(np.add((3/4) * self.U, (1/4) * U_1),
                     (1/4) * self.dt * L_1)

        L_2 = self.solver.solve(U_2, self.F)
        self.U = np.add(np.add((1/3) * self.U, (2/3) * U_2),
                        (2/3) * self.dt * L_2)

    def initialize(self, U):
        self.U = U
        
    def add_source(self, source):
        self.S = source

    def run_simulation(self, T, var="density", filename=None):
        t = 0
        self.plot(var)
        fig = plt.figure()

        dur = 8  # duration of video
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
                    t, T, prefix="Progress:", suffix="Complete", length=50)

        fig.clear()
        self.plot(var)
        plt.show()

    def initialize_U(self, U):
        self.U = U

    def sedov_blast(self):
        r, _ = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        self.U[r < 0.1] = np.array([1, 0, 0, 10])
        self.U[r >= 0.1] = np.array([1, 0, 0, E(self.gamma, 1, 1e-4, 0, 0)])

    def implosion(self):
        r, _ = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        self.U[r < 0.2] = np.array([0.125, 0, 0, E(0.125, 0.14, 0, 0)])
        self.U[r >= 0.2] = np.array([1, 0, 0, E(1, 1, 0, 0)])

    # case 3 in Liska Wendroff
    def quadrants(self):
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        rho_1, u_1, v_1, p_1 = 0.5323, 1.206, 0, 0.3
        rho_2, u_2, v_2, p_2 = 1.5, 0, 0, 1.5
        rho_3, u_3, v_3, p_3 = 0.138, 1.206, 1.206, 0.029
        rho_4, u_4, v_4, p_4 = 0.5323, 0, 1.206, 0.3
        self.U[(x < 0) & (y >= 0)] = np.array(
            [rho_1, rho_1 * u_1, rho_1 * v_1, E(rho_1, p_1, u_1, v_1)])
        self.U[(x >= 0) & (y >= 0)] = np.array(
            [rho_2, rho_2 * u_2, rho_2 * v_2, E(rho_2, p_2, u_2, v_2)])
        self.U[(x < 0) & (y < 0)] = np.array(
            [rho_3, rho_3 * u_3, rho_3 * v_3, E(rho_3, p_3, u_3, v_3)])
        self.U[(x >= 0) & (y < 0)] = np.array(
            [rho_4, rho_4 * u_4, rho_4 * v_4, E(rho_4, p_4, u_4, v_4)])

    def rayleigh_taylor(self):
        self.U = np.zeros((self.res_x, self.res_y, 4))
        g = -1
        def y_pert(x):
            return (0.5 - 0.02 * np.cos(4 * np.pi * x))
        
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                x = self.grid[i][j][0]
                y = self.grid[i][j][1]

                if y >= y_pert(x):
                    p = 2.5 + g * 2 * (y - 0.5)
                    self.U[i][j] = np.array([2, 0, 0, E(self.gamma, 2, p, 0, 0)])
                else:
                    p = 2.5 + g * 1 * (y - 0.5)
                    self.U[i][j] = np.array([1, 0, 0, E(self.gamma, 1, p, 0, 0)])