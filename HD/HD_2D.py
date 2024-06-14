from HD.helpers import c_s, E, P, get_prims, cartesian_to_polar, plot_grid
from HD.solvers import HLL, HLLC

import numpy as np
import matplotlib.pyplot as plt
import h5py


class Boundary:
    OUTFLOW = "outflow"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"


class HD_2D:
    def __init__(self, gamma=1.4, resolution=(100, 100),
                 xrange=(-1, 1), yrange=(-1, 1), solver="hll", high_time=False, high_space=False):
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

        self.num_g = 2  # number of ghost cells
        self.set_bcs()

        # conservative variable
        self.U = np.zeros((self.res_x, self.res_y, 4))
        # source terms (each a function of U)
        self.S = []

        self.high_time = high_time
        self.high_space = high_space
        self.dt = self.dx
        self.cfl = 0.4
        if solver == "hll":
            self.solver = HLL(
                gamma, resolution, self.num_g, self.x, self.y, self.dx, self.dy, self.high_space)
        elif solver == "hllc":
            self.solver = HLLC(
                gamma, resolution, self.num_g, self.x, self.y, self.dx, self.dy, self.high_space)
        else:
            raise Exception("Invalid solver: must be 'hll' or 'hllc'")

    def add_ghost_cells(self):
        # add ghost cells to the top and bottom boundaries
        self.U = np.hstack((np.repeat(self.U[:, :1, :], self.num_g, axis=1), self.U, np.repeat(
            self.U[:, :1, :], self.num_g, axis=1)))

        # add ghost cells to the left and right boundaries
        self.U = np.vstack((np.repeat(self.U[:1, :, :], self.num_g, axis=0), self.U, np.repeat(
            self.U[:1, :, :], self.num_g, axis=0)))

    # bc_x = (left, right), bc_y = (bottom, top)
    def set_bcs(self, bc_x=(Boundary.OUTFLOW, Boundary.OUTFLOW), bc_y=(Boundary.OUTFLOW, Boundary.OUTFLOW)):
        self.bc_x = bc_x
        self.bc_y = bc_y

    def apply_bcs(self):
        g = self.num_g
        # left
        if self.bc_x[0] == Boundary.OUTFLOW:
            self.U[:g, :, :] = self.U[g:(g+1), :, :]
        elif self.bc_x[0] == Boundary.REFLECTIVE:
            self.U[:g, :, :] = np.flip(self.U[g:(2*g), :, :], axis=0)
            # invert x momentum
            self.U[:g, :, 1] = -np.flip(self.U[g:(2*g), :, 1], axis=0)
        elif self.bc_x[0] == Boundary.PERIODIC:
            self.U[:g, :, :] = self.U[(-2*g):(-g), :, :]

        # right
        if self.bc_x[1] == Boundary.OUTFLOW:
            self.U[-g:, :, :] = self.U[-(g+1):(-g), :, :]
        elif self.bc_x[1] == Boundary.REFLECTIVE:
            self.U[-g:, :, :] = np.flip(self.U[(-2*g):(-g), :, :], axis=0)
            # invert x momentum
            self.U[-g:, :, 1] = -np.flip(self.U[(-2*g):(-g), :, 1], axis=0)
        elif self.bc_x[1] == Boundary.PERIODIC:
            self.U[-g:, :, :] = self.U[g:(2*g), :, :]

        # bottom
        if self.bc_y[0] == Boundary.OUTFLOW:
            self.U[:, :g, :] = self.U[:, g:(g+1), :]
        elif self.bc_y[0] == Boundary.REFLECTIVE:
            self.U[:, :g, :] = np.flip(self.U[:, g:(2*g), :], axis=1)
            # invert y momentum
            self.U[:, :g, 2] = -np.flip(self.U[:, g:(2*g), 2], axis=1)
        elif self.bc_y[0] == Boundary.PERIODIC:
            self.U[:, :g, :] = self.U[:, (-2*g):(-g), :]

        # top
        if self.bc_y[1] == Boundary.OUTFLOW:
            self.U[:, -g:, :] = self.U[:, -(g+1):(-g), :]
        elif self.bc_y[1] == Boundary.REFLECTIVE:
            self.U[:, -g:, :] = np.flip(self.U[:, (-2*g):(-g), :], axis=1)
            # invert y momentum
            self.U[:, -g:, 2] = -np.flip(self.U[:, (-2*g):(-g), 2], axis=1)
        elif self.bc_y[1] == Boundary.PERIODIC:
            self.U[:, -g:, :] = self.U[:, g:(2*g), :]

    def add_source(self, source):
        self.S.append(source)

    def compute_timestep(self):
        rho, u, v, p = get_prims(self.gamma, self.U)
        t = self.cfl * \
            np.min(self.dx / (np.sqrt(self.gamma * p / rho) + np.sqrt(u**2 + v**2)))
        return t
    
    def check_physical(self):
        # g = self.num_g
        # u = self.U[g:-g, g:-g]
        # for i in range(len(u)):
        #     for j in range(len(u[i])):
        #         cons = u[i][j]
        #         rho = cons[0]
        #         vx, vy = cons[1] / rho, cons[2] / rho
        #         E = cons[3]
        #         p = P(self.gamma, rho, vx, vy, E)
        #         if np.isnan(rho) or rho <= 0:
        #             print(f"rho={rho} at ({self.x[i]}, {self.y[j]})")
        #         if np.isnan(p) or p <= 0:
        #             print(f"p={p} at ({self.x[i]}, {self.y[j]})")
        #         if np.isnan(E) or E <= 0:
        #             print(f"E={cons[3]} at ({self.x[i]}, {self.y[j]})")

        rho = self.U[:, :, 0]
        momx = self.U[:, :, 1]
        momy = self.U[:, :, 2]
        En = self.U[:, :, 3]
        p = P(self.gamma, rho, momx/rho, momy/rho, En)
        invalid = (rho <= 0) | (p <= 0) | (En <= 0)
        momx[invalid] = 1e-6
        momy[invalid] = 1e-6
        En[invalid] = E(self.gamma, 1e-6, 1e-6, 1e-6, 1e-6)
        rho[invalid] = 1e-6

    def first_order_step(self):
        g = self.num_g
        self.dt = self.compute_timestep()

        L = self.solver.solve(self.U)
        u = self.U[g:-g, g:-g, :]
        u += L * self.dt
        # add source terms
        for s in self.S:
            u += s(u) * self.dt
        self.U[g:-g, g:-g, :] = u

    def high_order_step(self):
        """
        Third-order Runge-Kutta method
        """
        self.dt = self.compute_timestep()
        L = self.solver.solve(self.U)
        U_1 = np.add(self.U, L * self.dt)

        L_1 = self.solver.solve(U_1)
        U_2 = np.add(np.add((3/4) * self.U, (1/4) * U_1),
                     (1/4) * self.dt * L_1)

        L_2 = self.solver.solve(U_2)
        self.U = np.add(np.add((1/3) * self.U, (2/3) * U_2),
                        (2/3) * self.dt * L_2)

    def run(self, T, plot=None, filename="out", save_interval=0.1):
        t = 0
        fig = plt.figure()
        PATH = f"./output/{filename}.hdf"
        self.add_ghost_cells()

        next_checkpoint = 0
        g = self.num_g

        # open HDF5 file to save U state at each checkpoint
        with h5py.File(PATH, "w") as outfile:
            # Create an extendable dataset
            # None allows unlimited growth in the first dimension
            max_shape = (None, self.res_x, self.res_y, 4)
            dataset = outfile.create_dataset("data", shape=(
                0, self.res_x, self.res_y, 4), maxshape=max_shape, chunks=True)

            # metadata
            dataset.attrs["gamma"] = self.gamma
            dataset.attrs["xrange"] = (self.xmin, self.xmax)
            dataset.attrs["yrange"] = (self.ymin, self.ymax)
            dataset.attrs["t"] = []

            while t < T:
                self.check_physical()
                self.apply_bcs()

                if self.high_time:
                    self.high_order_step()
                else:
                    self.first_order_step()

                # at each checkpoint, save the current state excluding the ghost cells
                if t >= next_checkpoint:
                    dataset.resize(dataset.shape[0] + 1, axis=0)
                    dataset[-1] = self.U[g:-g, g:-g]
                    next_checkpoint += save_interval
                    dataset.attrs["t"] = np.append(dataset.attrs["t"], t)

                t = t + self.dt if (t + self.dt <= T) else T
                self.print_progress_bar(t, T, suffix="complete", length=50)

                if plot:
                    fig.clear()
                    plot_grid(self.gamma, self.U[g:-g, g:-g], t=t, plot=plot,
                              extent=[self.xmin, self.xmax, self.ymin, self.ymax])
                    plt.pause(0.001)

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

        p_m = 2.5
        cs = c_s(self.gamma, p_m, 2)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                x = self.grid[i][j][0]
                y = self.grid[i][j][1]
                v = (cs * 0.01) * (1-np.cos(4 * np.pi * x)) * \
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

        def gravity(U):
            S = np.zeros_like(U)
            g = -0.1
            rho = U[:, :, 0]
            u = U[:, :, 1] / rho
            v = U[:, :, 2] / rho
            S[:, :, 2] = g * rho
            S[:, :, 3] = g * np.multiply(rho, v)
            return S

        self.add_source(gravity)

    def kepler(self):
        r, theta = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        eps = 0.01
        mach = 10
        G = 1
        M = 1
        g = - G * M / ((r + eps) ** 2)
        rho = np.ones_like(r) * 1
        v_k = np.sqrt(G * M / (r + eps))  # keplerian velocity
        cs = v_k / mach
        p = rho * (cs ** 2) / self.gamma

        u = - v_k * np.sin(theta)
        v = v_k * np.cos(theta)
        self.U = np.array([
            rho,
            rho * u,
            rho * v,
            E(self.gamma, rho, p, u, v)
        ]).transpose((1, 2, 0))
        
        def gravity(U):
            S = np.zeros_like(U)
            rho, u, v, p = get_prims(self.gamma, U)
            g_x, g_y = g * np.cos(theta), g * np.sin(theta)
            S[:, :, 1] = rho * g_x
            S[:, :, 2] = rho * g_y
            S[:, :, 3] = rho * (u * g_x + v * g_y)
            return S

        self.add_source(gravity)

        def kernel(U):
            a = 2.5
            Delta = (self.xmax - self.xmin) / 30
            gaussian = -a * np.exp(-(r ** 2) / (2 * Delta ** 2))
            S = np.zeros_like(U)
            S[:, :, 0] = gaussian
            S[:, :, 1] = 0 # gaussian * np.sign(U[:, :, 1])
            S[:, :, 2] = 0 # gaussian # * np.sign(U[:, :, 2])
            S[:, :, 3] = 0 # gaussian
            return S

        self.add_source(kernel)

    def binary(self):
        r, theta = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        eps = 0.01
        mach = 10
        G = 1
        M = 1
        g = - G * M / ((r + eps) ** 2)
        rho = np.ones_like(r) * 1
        v_k = np.sqrt(G * M / (r + eps))  # keplerian velocity
        cs = v_k / mach
        p = rho * (cs ** 2) / self.gamma

        u = - v_k * np.sin(theta)
        v = v_k * np.cos(theta)
        self.U = np.array([
            rho,
            rho * u,
            rho * v,
            E(self.gamma, rho, p, u, v)
        ]).transpose((1, 2, 0))