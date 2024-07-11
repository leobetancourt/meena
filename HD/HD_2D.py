from HD.helpers import add_ghost_cells, cartesian_to_polar, plot_grid, print_progress_bar
from HD.solvers import HLL, HLLC

import numpy as np
import matplotlib.pyplot as plt
import h5py


class Boundary:
    OUTFLOW = "outflow"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"


def linspace_cells(min, max, num):
    interfaces = np.linspace(min, max, num + 1)
    centers = (interfaces[:-1] + interfaces[1:]) / 2

    return centers, interfaces


def logspace_cells(min, max, num):
    interfaces = np.logspace(np.log10(min), np.log10(max), num + 1)
    centers = (interfaces[:-1] + interfaces[1:]) / 2

    return centers, interfaces


class HD_2D:
    def __init__(self, gamma=1.4, nu=None, coords="cartesian", resolution=(100, 100),
                 x1_range=(-1, 1), x2_range=(-1, 1), logspace=False, solver="hll", high_time=False, high_space=False):
        self.gamma = gamma
        self.nu = nu  # viscosity
        self.eos = "ideal"

        self.coords = coords
        # grid initialization
        self.res_x1, self.res_x2 = resolution
        self.x1_min, self.x1_max = x1_range
        self.x2_min, self.x2_max = x2_range
        self.x1, self.x1_interf = linspace_cells(
            self.x1_min, self.x1_max, num=self.res_x1)
        self.x2, self.x2_interf = linspace_cells(
            self.x2_min, self.x2_max, num=self.res_x2)
        if logspace:
            self.x1, self.x1_interf = logspace_cells(
                self.x1_min, self.x1_max, num=self.res_x1)
        self.grid = np.array(
            [np.array([[_x1, _x2] for _x2 in self.x2]) for _x1 in self.x1])

        self.dx1, self.dx2 = (self.x1_max - self.x1_min) / \
            self.res_x1, (self.x2_max - self.x2_min) / self.res_x2

        self.U = np.zeros((self.res_x1, self.res_x2, 4))

        self.num_g = 2  # number of ghost cells
        self.set_bcs()

        # source terms (each a function of U)
        self.S = []

        self.high_time = high_time
        self.high_space = high_space
        self.dt = self.dx1
        self.cfl = 0.5
        if solver == "hll":
            self.solver = HLL(self, gamma, self.nu, self.num_g, coords, resolution, self.x1,
                              self.x2, self.x1_interf, self.x2_interf, high_order=self.high_space)
        elif solver == "hllc":
            self.solver = HLLC(self, gamma, self.nu, self.num_g, coords, resolution, self.x1,
                               self.x2, self.x1_interf, self.x2_interf, high_order=self.high_space)
        else:
            raise Exception("Invalid solver: must be 'hll' or 'hllc'")

        # list of diagnostics, each a tuple (name : string, get : method)
        self.diagnostics = []

    def E(self, rho, p, u, v):
        return (p / (self.gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))

    def P(self, rho, u, v, E):
        p = (self.gamma - 1) * (E - (0.5 * rho * (u ** 2 + v ** 2)))
        return p

    def get_prims(self, U=None):
        g = self.num_g
        if U == None:
            U = self.U[g:-g, g:-g]
        u = np.copy(U)
        rho = u[:, :, 0]
        u, v = u[:, :, 1] / rho, u[:, :, 2] / rho
        E = u[:, :, 3]
        p = self.P(rho, u, v, E)
        return rho, u, v, p

    def U_from_prim(self, prims):
        rho, u, v, p = prims
        e = self.E(rho, p, u, v)
        U = np.array([rho, rho * u, rho * v, e]).transpose((1, 2, 0))
        return U

    def F_from_prim(self, prims, x1=True):
        rho, u, v, p = prims
        e = self.E(rho, p, u, v)
        if x1:
            F = np.array([
                rho * u,
                rho * (u ** 2) + p,
                rho * u * v,
                (e + p) * u
            ]).transpose((1, 2, 0))
        else:
            F = np.array([
                rho * v,
                rho * u * v,
                rho * (v ** 2) + p,
                (e + p) * v
            ]).transpose((1, 2, 0))

        return F

    def c_s(self, p=None, rho=None):
        if p is None:
            rho, _, _, p = self.get_prims()
        return np.sqrt(self.gamma * p / rho)

    def set_bcs(self, bc_x1=(Boundary.OUTFLOW, Boundary.OUTFLOW), bc_x2=(Boundary.OUTFLOW, Boundary.OUTFLOW)):
        # bc_i is a tuple of the form defining the inner and outer boundary conditions for coordinate i
        self.bc_x1 = bc_x1
        self.bc_x2 = bc_x2

    def apply_bcs(self):
        g = self.num_g
        # left
        if self.bc_x1[0] == Boundary.OUTFLOW:
            self.U[:g, :, :] = self.U[g:(g+1), :, :]
        elif self.bc_x1[0] == Boundary.REFLECTIVE:
            self.U[:g, :, :] = np.flip(self.U[g:(2*g), :, :], axis=0)
            # invert x momentum
            self.U[:g, :, 1] = -np.flip(self.U[g:(2*g), :, 1], axis=0)
        elif self.bc_x1[0] == Boundary.PERIODIC:
            self.U[:g, :, :] = self.U[(-2*g):(-g), :, :]

        # right
        if self.bc_x1[1] == Boundary.OUTFLOW:
            self.U[-g:, :, :] = self.U[-(g+1):(-g), :, :]
        elif self.bc_x1[1] == Boundary.REFLECTIVE:
            self.U[-g:, :, :] = np.flip(self.U[(-2*g):(-g), :, :], axis=0)
            # invert x momentum
            self.U[-g:, :, 1] = -np.flip(self.U[(-2*g):(-g), :, 1], axis=0)
        elif self.bc_x1[1] == Boundary.PERIODIC:
            self.U[-g:, :, :] = self.U[g:(2*g), :, :]

        # bottom
        if self.bc_x2[0] == Boundary.OUTFLOW:
            self.U[:, :g, :] = self.U[:, g:(g+1), :]
        elif self.bc_x2[0] == Boundary.REFLECTIVE:
            self.U[:, :g, :] = np.flip(self.U[:, g:(2*g), :], axis=1)
            # invert y momentum
            self.U[:, :g, 2] = -np.flip(self.U[:, g:(2*g), 2], axis=1)
        elif self.bc_x2[0] == Boundary.PERIODIC:
            self.U[:, :g, :] = self.U[:, (-2*g):(-g), :]

        # top
        if self.bc_x2[1] == Boundary.OUTFLOW:
            self.U[:, -g:, :] = self.U[:, -(g+1):(-g), :]
        elif self.bc_x2[1] == Boundary.REFLECTIVE:
            self.U[:, -g:, :] = np.flip(self.U[:, (-2*g):(-g), :], axis=1)
            # invert y momentum
            self.U[:, -g:, 2] = -np.flip(self.U[:, (-2*g):(-g), 2], axis=1)
        elif self.bc_x2[1] == Boundary.PERIODIC:
            self.U[:, -g:, :] = self.U[:, g:(2*g), :]

    def add_source(self, source):
        self.S.append(source)

    def compute_timestep(self):
        rho, u, v, p = self.get_prims()
        cs = self.c_s()
        if self.coords == "cartesian":
            dt = self.cfl * \
                min(np.min(self.dx1 / (np.abs(u) + cs)),
                    np.min(self.dx2 / (np.abs(v) + cs)))
        elif self.coords == "polar":
            R_interf, _ = np.meshgrid(self.x1_interf, self.x2, indexing="ij")
            x1_l, x1_r = R_interf[:-1, :], R_interf[1:, :]
            delta_r = x1_r - x1_l
            R, _ = np.meshgrid(self.x1, self.x2, indexing="ij")
            dt = self.cfl * \
                min(np.min(delta_r / (np.abs(u) + cs)),
                    np.min(R * self.dx2 / (np.abs(v) + cs)))
        return dt

    def check_physical(self):
        g = self.num_g
        u = self.U[g:-g, g:-g]
        rho = u[:, :, 0]
        momx = u[:, :, 1]
        momy = u[:, :, 2]
        En = u[:, :, 3]
        _, _, _, p = self.get_prims(u)
        invalid = (rho <= 0) | (p <= 0) | (En <= 0)
        momx[invalid] = 1e-12
        momy[invalid] = 1e-12
        En[invalid] = 1.5e-6
        rho[invalid] = 1e-6

    def first_order_step(self, t):
        g = self.num_g
        self.dt = self.compute_timestep()

        L = self.solver.solve(self.U)
        u = self.U[g:-g, g:-g, :]
        u += L * self.dt
        # add source terms
        for s in self.S:
            u += s(u) * self.dt
        self.U[g:-g, g:-g, :] = u

    def high_order_step(self, t):
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

    # resizes h5py dataset and saves d
    def save_to_dset(self, dset, d):
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1] = d

    def add_diagnostic(self, name, get_func):
        self.diagnostics.append((name, get_func))

    def run(self, T, plot=None, filename="out", save_interval=0.1):
        t = 0
        PATH = f"./output/{filename}.hdf"
        self.U = add_ghost_cells(self.U, self.num_g)
        labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$",
                  "v": r"$v$", "pressure": r"$P$", "energy": r"$E$", }

        next_checkpoint = 0
        g = self.num_g

        # open HDF5 file to save U state at each checkpoint
        with h5py.File(PATH, "w") as f:
            # metadata
            f.attrs["coords"] = self.coords
            f.attrs["gamma"] = self.gamma
            f.attrs["x1"] = self.x1
            f.attrs["x2"] = self.x2

            # create h5 datasets for time, conserved variables and diagnostics
            max_shape = (None, self.res_x1, self.res_x2)
            dset_t = f.create_dataset("t", (0,), maxshape=(
                None,), chunks=True, dtype="float64")  # simulation times
            dset_tc = f.create_dataset("tc", (0,), maxshape=(
                None,), chunks=True, dtype="float64")  # checkpoint times
            dset_rho = f.create_dataset(
                "rho", (0, self.res_x1, self.res_x2), maxshape=max_shape, chunks=True, dtype="float64")
            dset_momx1 = f.create_dataset(
                "momx1", (0, self.res_x1, self.res_x2), maxshape=max_shape, chunks=True, dtype="float64")
            dset_momx2 = f.create_dataset(
                "momx2", (0, self.res_x1, self.res_x2), maxshape=max_shape, chunks=True, dtype="float64")
            dset_E = f.create_dataset(
                "E", (0, self.res_x1, self.res_x2), maxshape=max_shape, chunks=True, dtype="float64")
            for i, tup in enumerate(self.diagnostics):
                name, get_func = tup
                dset = f.create_dataset(name, (0,), maxshape=(
                    None,), chunks=True, dtype="float64")
                self.diagnostics[i] = (name, get_func, dset)

            rho, u, v, p = self.get_prims()
            En = self.U[g:-g, g:-g, 3]
            vmin, vmax = None, None
            if plot == "density":
                matrix = rho
            elif plot == "log density":
                matrix = np.log10(rho)
                vmin, vmax = -3, 0.5
            elif plot == "u":
                matrix = u
            elif plot == "v":
                matrix = v
            elif plot == "pressure":
                matrix = p
            elif plot == "energy":
                matrix = En

            if plot:
                fig, ax, c, cb = plot_grid(
                    matrix, labels[plot], coords=self.coords, x1=self.x1, x2=self.x2, vmin=vmin, vmax=vmax)
                ax.set_title(f"t = {t:.2f}")

            while t < T:
                # at each timestep, save diagnostics
                self.save_to_dset(dset_t, t)
                for tup in self.diagnostics:
                    name, get_val, dset = tup
                    self.save_to_dset(dset, get_val())

                # at each checkpoint, save the current state excluding the ghost cells
                if t >= next_checkpoint:
                    self.save_to_dset(dset_tc, t)
                    self.save_to_dset(dset_rho, self.U[g:-g, g:-g, 0])
                    self.save_to_dset(dset_momx1, self.U[g:-g, g:-g, 1])
                    self.save_to_dset(dset_momx2, self.U[g:-g, g:-g, 2])
                    self.save_to_dset(dset_E, self.U[g:-g, g:-g, 3])

                    next_checkpoint += save_interval

                self.apply_bcs()

                if self.high_time:
                    self.high_order_step(t)
                else:
                    self.first_order_step(t)

                self.check_physical()

                if plot:
                    if self.coords == "cartesian":
                        c.set_data(matrix)
                    elif self.coords == "polar":
                        c.set_array(matrix.ravel())
                    # c.set_clim(vmin=np.min(matrix), vmax=np.max(matrix))
                    cb.update_normal(c)
                    ax.set_title(f"t = {t:.2f}")
                    fig.canvas.draw()
                    plt.pause(0.001)

                t = t + self.dt if (t + self.dt <= T) else T
                print_progress_bar(t, T, suffix="complete", length=50)

                rho, u, v, p = self.get_prims()
                En = self.U[g:-g, g:-g, 3]
                if plot == "density":
                    matrix = rho
                elif plot == "log density":
                    matrix = np.log10(rho)
                elif plot == "u":
                    matrix = u
                elif plot == "v":
                    matrix = v
                elif plot == "pressure":
                    matrix = p
                elif plot == "energy":
                    matrix = En

            for tup in self.diagnostics:
                name, get_val, dset = tup
                self.save_to_dset(dset, get_val())
            self.save_to_dset(dset_t, t)
            self.save_to_dset(dset_tc, t)
            self.save_to_dset(dset_rho, self.U[g:-g, g:-g, 0])
            self.save_to_dset(dset_momx1, self.U[g:-g, g:-g, 1])
            self.save_to_dset(dset_momx2, self.U[g:-g, g:-g, 2])
            self.save_to_dset(dset_E, self.U[g:-g, g:-g, 3])

        plt.show()

    def sedov_blast(self, radius=0.1):
        if self.coords == "cartesian":
            r, _ = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        elif self.coords == "polar":
            r, _ = self.grid[:, :, 0], self.grid[:, :, 1]
        self.U[r < radius] = np.array([1, 0, 0, 10])
        self.U[r >= radius] = np.array([1, 0, 0, self.E(1, 1e-4, 0, 0)])

    def implosion(self):
        if self.coords == "cartesian":
            r, _ = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])
        elif self.coords == "polar":
            r, _ = self.grid[:, :, 0], self.grid[:, :, 1]
        self.U[r < 0.2] = np.array([0.125, 0, 0, self.E(0.125, 0.14, 0, 0)])
        self.U[r >= 0.2] = np.array([1, 0, 0, self.E(1, 1, 0, 0)])

    def sheer(self):
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        mid = (self.x1_max + self.x1_min) / 2
        rho_L, u_L, v_L, p_L = 1, 0, 1, 1
        self.U[x <= -0.5] = np.array(
            [rho_L, rho_L * u_L, rho_L * v_L, self.E(rho_L, p_L, u_L, v_L)])
        rho_R, u_R, v_R, p_R = 1, 0, -1, 1
        self.U[x > -0.5] = np.array(
            [rho_R, rho_R * u_R, rho_R * v_R, self.E(rho_R, p_R, u_R, v_R)])

    # case 3 in Liska Wendroff
    def quadrants(self):
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        mid = (self.x1_max - self.x1_min) / 2
        rho_1, u_1, v_1, p_1 = 0.5323, 1.206, 0, 0.3
        rho_2, u_2, v_2, p_2 = 1.5, 0, 0, 1.5
        rho_3, u_3, v_3, p_3 = 0.138, 1.206, 1.206, 0.029
        rho_4, u_4, v_4, p_4 = 0.5323, 0, 1.206, 0.3
        self.U[(x < mid) & (y >= mid)] = np.array(
            [rho_1, rho_1 * u_1, rho_1 * v_1, self.E(rho_1, p_1, u_1, v_1)])
        self.U[(x >= mid) & (y >= mid)] = np.array(
            [rho_2, rho_2 * u_2, rho_2 * v_2, self.E(rho_2, p_2, u_2, v_2)])
        self.U[(x < mid) & (y < mid)] = np.array(
            [rho_3, rho_3 * u_3, rho_3 * v_3, self.E(rho_3, p_3, u_3, v_3)])
        self.U[(x >= mid) & (y < mid)] = np.array(
            [rho_4, rho_4 * u_4, rho_4 * v_4, self.E(rho_4, p_4, u_4, v_4)])

    # initial conditions from Deng, Boivin, Xiao
    # https://hal.science/hal-02100764/document
    def rayleigh_taylor(self):
        self.U = np.zeros((self.res_x1, self.res_x2, 4))
        g = -0.1

        p_m = 2.5
        cs = self.c_s(self.gamma, p_m, 2)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                x = self.grid[i][j][0]
                y = self.grid[i][j][1]
                v = (cs * 0.01) * (1-np.cos(4 * np.pi * x)) * \
                    (1-np.cos(4 * np.pi * y / 3))

                if y >= 0.75:
                    p = 2.5 + g * 2 * (y - 0.75)
                    self.U[i][j] = np.array(
                        [2, 0, 2 * v, self.E(2, p, 0, v)])
                else:
                    p = 2.5 + g * 1 * (y - 0.75)
                    self.U[i][j] = np.array(
                        [1, 0, 1 * v, self.E(1, p, 0, v)])

    def kelvin_helmholtz(self):
        self.U = np.zeros((self.res_x1, self.res_x2, 4))
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
                        [1, -0.5, 0, self.E(rho, p, u, v)])
                else:
                    rho, u = 2, 0.5
                    v = w0*np.sin(4*np.pi*x) * (np.exp(-(y-0.25)**2 /
                                                       (2 * sigma**2)) + np.exp(-(y-0.75)**2/(2*sigma**2)))
                    p = 2.5
                    self.U[i][j] = np.array(
                        [rho, u, v, self.E(rho, p, u, v)])

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
