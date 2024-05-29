from sim.helpers import E, P, enthalpy, cartesian_to_polar, polar_to_cartesian, c_s
from sim.solvers import HLL, HLLC
from contextlib import nullcontext
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


class Boundary:
    OUTFLOW = "outflow"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"


class Sim_2D:
    def __init__(self, gamma=1.4, resolution=(100, 100),
                 xrange=(-1, 1), yrange=(-1, 1), solver="hll", order="first"):
        self.gamma = gamma

        # grid initialization
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
            raise Exception("Invalid order: must be 'first' or 'high'")
        self.order = order
        self.dt = self.dx
        self.cfl = 0.4
        if solver == "hll":
            self.solver = HLL(
                gamma, resolution, self.x, self.y, self.dx, self.dy)
        elif solver == "hllc":
            self.solver = HLLC(
                gamma, resolution, self.x, self.y, self.dx, self.dy)
        else:
            raise Exception("Invlaid solver: must be 'hll' or 'hllc'")

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
