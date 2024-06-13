from MHD.helpers import c_s, c_fm, E, P, get_prims, plot_grid
from MHD.solvers import HLL

import numpy as np
import matplotlib.pyplot as plt
import h5py


class Boundary:
    OUTFLOW = "outflow"
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"


class MHD:
    def __init__(self, gamma=1.4, resolution=(100, 100, 100),
                 xrange=(-1, 1), yrange=(-1, 1), zrange=(-1, 1), solver="hll", high_time=False, high_space=False):
        self.gamma = gamma

        # grid initialization
        self.res_x, self.res_y, self.res_z = resolution
        self.xmin, self.xmax = xrange
        self.ymin, self.ymax = yrange
        self.zmin, self.zmax = zrange
        self.x = np.linspace(self.xmin, self.xmax,
                             num=self.res_x, endpoint=False)
        self.y = np.linspace(self.ymin, self.ymax,
                             num=self.res_y, endpoint=False)
        self.z = np.linspace(self.zmin, self.zmax,
                             num=self.res_z, endpoint=False)
        self.grid = self.create_grid()
        self.dx, self.dy, self.dz = (self.xmax - self.xmin) / \
            self.res_x, (self.ymax - self.ymin) / \
            self.res_y, (self.zmax - self.zmin) / self.res_z

        self.num_g = 2  # number of ghost cells
        self.set_bcs()

        # conservative variable
        self.U = np.zeros((self.res_x, self.res_y, self.res_y, 8))
        # source terms (each a function of U)
        self.S = []

        self.high_time = high_time
        self.dt = self.dx
        self.cfl = 1
        self.solver = HLL(gamma, resolution, self.num_g, self.x,
                          self.y, self.z, self.dx, self.dy, self.dz)

    def create_grid(self):
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        grid = np.stack((X, Y, Z), axis=-1)
        return grid

    def add_ghost_cells(self):
        # add ghost cells to the z boundaries
        self.U = np.dstack((np.repeat(self.U[:, :, :1, :], self.num_g, axis=2), self.U, np.repeat(
            self.U[:, :, :1, :], self.num_g, axis=2)))

        # add ghost cells to the top and bottom boundaries
        self.U = np.hstack((np.repeat(self.U[:, :1, :, :], self.num_g, axis=1), self.U, np.repeat(
            self.U[:, :1, :, :], self.num_g, axis=1)))

        # add ghost cells to the left and right boundaries
        self.U = np.vstack((np.repeat(self.U[:1, :, :, :], self.num_g, axis=0), self.U, np.repeat(
            self.U[:1, :, :, :], self.num_g, axis=0)))

    # bc_x = (left, right), bc_y = (bottom, top)

    def set_bcs(self, bc_x=(Boundary.OUTFLOW, Boundary.OUTFLOW), bc_y=(Boundary.OUTFLOW, Boundary.OUTFLOW), bc_z=(Boundary.OUTFLOW, Boundary.OUTFLOW)):
        self.bc_x = bc_x
        self.bc_y = bc_y
        self.bc_z = bc_z

    def apply_bcs(self):
        """
            finish fixing for 3D MHD
        """
        g = self.num_g
        # x left
        if self.bc_x[0] == Boundary.OUTFLOW:
            self.U[:g, ...] = self.U[g:(g+1), ...]
        elif self.bc_x[0] == Boundary.REFLECTIVE:
            self.U[:g, ...] = np.flip(self.U[g:(2*g), ...], axis=0)
            # invert x momentum
            self.U[:g, :, :, 1] = -np.flip(self.U[g:(2*g), :, :, 1], axis=0)
        elif self.bc_x[0] == Boundary.PERIODIC:
            self.U[:g, ...] = self.U[(-2*g):(-g), ...]

        # x right
        if self.bc_x[1] == Boundary.OUTFLOW:
            self.U[-g:, ...] = self.U[-(g+1):(-g), ...]
        elif self.bc_x[1] == Boundary.REFLECTIVE:
            self.U[-g:, ...] = np.flip(self.U[(-2*g):(-g), ...], axis=0)
            # invert x momentum
            self.U[-g:, :, :, 1] = - \
                np.flip(self.U[(-2*g):(-g), :, :, 1], axis=0)
        elif self.bc_x[1] == Boundary.PERIODIC:
            self.U[-g:, ...] = self.U[g:(2*g), ...]

        # y left
        if self.bc_y[0] == Boundary.OUTFLOW:
            self.U[:, :g, :, :] = self.U[:, g:(g+1), :, :]
        elif self.bc_y[0] == Boundary.REFLECTIVE:
            self.U[:, :g, :, :] = np.flip(self.U[:, g:(2*g), :, :], axis=1)
            # invert y momentum
            self.U[:, :g, :, 2] = -np.flip(self.U[:, g:(2*g), :, 2], axis=1)
        elif self.bc_y[0] == Boundary.PERIODIC:
            self.U[:, :g, :, :] = self.U[:, (-2*g):(-g), :, :]

        # y right
        if self.bc_y[1] == Boundary.OUTFLOW:
            self.U[:, -g:, :, :] = self.U[:, -(g+1):(-g), :, :]
        elif self.bc_y[1] == Boundary.REFLECTIVE:
            self.U[:, -g:, :,
                :] = np.flip(self.U[:, (-2*g):(-g), :, :], axis=1)
            # invert y momentum
            self.U[:, -g:, :, 2] = - \
                np.flip(self.U[:, (-2*g):(-g), :, 2], axis=1)
        elif self.bc_y[1] == Boundary.PERIODIC:
            self.U[:, -g:, :, :] = self.U[:, g:(2*g), :, :]

        # z left
        if self.bc_z[0] == Boundary.OUTFLOW:
            self.U[:, :g, :, :] = self.U[:, g:(g+1), :, :]
        elif self.bc_z[0] == Boundary.REFLECTIVE:
            self.U[:, :g, :, :] = np.flip(self.U[:, g:(2*g), :, :], axis=1)
            # invert y momentum
            self.U[:, :g, :, 2] = -np.flip(self.U[:, g:(2*g), :, 2], axis=1)
        elif self.bc_z[0] == Boundary.PERIODIC:
            self.U[:, :g, :, :] = self.U[:, (-2*g):(-g), :, :]

        # z right
        if self.bc_z[1] == Boundary.OUTFLOW:
            self.U[:, :, -g:, :] = self.U[:, :, -(g+1):(-g), :]
        elif self.bc_z[1] == Boundary.REFLECTIVE:
            self.U[:, :, -g:,
                :] = np.flip(self.U[:, :, (-2*g):(-g), :], axis=2)
            # invert y momentum
            self.U[:, :, -g:, 3] = - \
                np.flip(self.U[:, :, (-2*g):(-g), 3], axis=2)
        elif self.bc_z[1] == Boundary.PERIODIC:
            self.U[:, :, -g:, :] = self.U[:, :, g:(2*g), :]

    def add_source(self, source):
        self.S.append(source)

    def compute_timestep(self):
        rho, u, v, w, p, Bx, By, Bz = get_prims(self.gamma, self.U)
        cfm = c_fm(self.gamma, p, rho, Bx, By, Bz)
        return self.cfl * np.min(self.dx / (np.sqrt(u ** 2 + v ** 2 + w ** 2) + cfm))

    def first_order_step(self):
        g = self.num_g
        self.dt = self.compute_timestep()

        L = self.solver.solve(self.U)
        u = self.U[g:-g, g:-g, g:-g, :]
        u += L * self.dt
        # add source terms
        for s in self.S:
            u += s(u) * self.dt
        self.U[g:-g, g:-g, g:-g, :] = u

    def high_order_step(self):
        # implement this
        g = self.num_g
        self.dt = self.compute_timestep()

        L = self.solver.solve(self.U)
        u = self.U[g:-g, g:-g, g:-g, :]
        u += L * self.dt
        # add source terms
        for s in self.S:
            u += s(u) * self.dt
        self.U[g:-g, g:-g, g:-g, :] = u

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
            max_shape = (None, self.res_x, self.res_y, self.res_z, 8)
            dataset = outfile.create_dataset("data", shape=(
                0, self.res_x, self.res_y, self.res_z, 8), maxshape=max_shape, chunks=True)

            # metadata
            dataset.attrs["gamma"] = self.gamma
            dataset.attrs["xrange"] = (self.xmin, self.xmax)
            dataset.attrs["yrange"] = (self.ymin, self.ymax)
            dataset.attrs["zrange"] = (self.ymin, self.ymax)
            dataset.attrs["t"] = []

            while t < T:
                self.apply_bcs()

                if self.high_time:
                    self.high_order_step()
                else:
                    self.first_order_step()

                # at each checkpoint, save the current state excluding the ghost cells
                if t >= next_checkpoint:
                    dataset.resize(dataset.shape[0] + 1, axis=0)
                    dataset[-1] = self.U[g:-g, g:-g, g:-g]
                    next_checkpoint += save_interval
                    dataset.attrs["t"] = np.append(dataset.attrs["t"], t)

                t = t + self.dt if (t + self.dt <= T) else T
                self.print_progress_bar(t, T, suffix="complete", length=50)

                if plot:
                    fig.clear()
                    plot_grid(self.gamma, self.U[g:-g, g:-g, g:-g], t=t, plot=plot, extent=[self.xmin, self.xmax, self.ymin, self.ymax])
                    plt.pause(0.001)

            plt.show()

    # call in a loop to print dynamic progress bar

    def print_progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        percent= ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength= int(length * iteration // total)
        bar= fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # print new line on complete
        if iteration == total:
            print()

    # Brio and Wu shock tube
    def shock_tube(self):
        Bx= 0.75
        rho_L, u_L, v_L, w_L, Bx_L, By_L, Bz_L, p_L= 1, 0, 0, 0, Bx, 1, 0, 1
        rho_R, u_R, v_R, w_R, Bx_R, By_R, Bz_R, p_R= 0.125, 0, 0, 0, Bx, -1, 0, 0.1
        E_L= E(self.gamma, rho_L, u_L, v_L, w_L, p_L, Bx_L, By_L, Bz_L)
        U_L= np.array([rho_L, rho_L * u_L, rho_L * v_L, rho_L * w_L, Bx_L, By_L, Bz_L, E_L])
        E_R= E(self.gamma, rho_R, u_R, v_R, w_R, p_R, Bx_R, By_R, Bz_R)
        U_R= np.array([rho_R, rho_R * u_R, rho_R * v_R, rho_R * w_R, Bx_R, By_R, Bz_R, E_R])
        self.U= np.array([[[U_L if x <= 0 else U_R]] for x in self.x])
