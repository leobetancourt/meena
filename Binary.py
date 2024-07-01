import numpy as np
from HD.HD_2D import HD_2D, Boundary
from HD.helpers import E, P, cartesian_to_polar, get_prims


class Binary(HD_2D):
    def __init__(self, gamma=1.4, resolution=(100, 100),
                 xrange=(-5, 5), yrange=(-5, 5), solver="hll", high_space=False):
        self.G, self.M = 1, 1
        self.mach = 10
        self.a = 1  # binary separation
        self.R_cav = 2.5 * self.a
        self.R_out = 10
        self.delta_0 = 1e-5
        self.Sigma_0 = 1
        self.eps = 0.05 * self.a  # gravitational softening
        self.period = 1
        self.omega = 2 * np.pi / self.period
        self.x1, self.y1 = self.a / 2, 0
        self.x2, self.y2 = -self.a / 2, 0
        self.ts = []
        self.accr_1 = []
        self.accr_2 = []

        super().__init__(gamma, nu=1e-3 * self.a ** 2 * self.omega, resolution=resolution, xrange=xrange,
                         yrange=yrange, solver=solver, high_space=high_space)
        self.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
                     (Boundary.OUTFLOW, Boundary.OUTFLOW))

        self.setup()
        # self.kepler()

    def buffer(self):
        g = self.num_g
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        r, theta = cartesian_to_polar(x, y)

        buff = r >= (0.95 * self.xmax)
        v = np.sqrt(self.G * self.M / r[buff])
        cs = v / self.mach
        u_k, v_k = - v * np.sin(theta[buff]), v * np.cos(theta[buff])
        rho = self.U[g:-g, g:-g, 0][buff]
        p = rho * (cs ** 2) / self.gamma

        self.U[g:-g, g:-g, 1][buff] = rho * u_k
        self.U[g:-g, g:-g, 2][buff] = rho * v_k
        self.U[g:-g, g:-g, 3][buff] = E(self.gamma, rho, p, u_k, v_k)

    def first_order_step(self, t):
        # update black hole positions
        delta = np.pi
        self.x1 = (self.a / 2) * np.cos(self.omega * t)
        self.y1 = (self.a / 2) * np.sin(self.omega * t)
        self.x2 = (self.a / 2) * np.cos(self.omega * t + delta)
        self.y2 = (self.a / 2) * np.sin(self.omega * t + delta)

        # set buffer velocities
        self.buffer()

        self.ts.append(t)

        super().first_order_step(t)

    def kepler(self):
        r, theta = cartesian_to_polar(self.grid[:, :, 0], self.grid[:, :, 1])

        def f(r):
            return 1 - (1 / (1 + np.exp(-2 * (r - self.R_out) / self.a)))
        rho = self.Sigma_0 * \
            ((1 - self.delta_0) * np.exp(-(self.R_cav /
             (r + self.eps)) ** 12) + self.delta_0) * f(r)
        v_k = np.sqrt(self.G * self.M / (r + self.eps))  # keplerian velocity
        cs = v_k / self.mach
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
            g = - self.G * self.M / ((r + self.eps) ** 2)
            S = np.zeros_like(U)
            rho, u, v, p = get_prims(self.gamma, U)
            g_x, g_y = g * np.cos(theta), g * np.sin(theta)
            S[:, :, 1] = rho * g_x
            S[:, :, 2] = rho * g_y
            S[:, :, 3] = rho * (u * g_x + v * g_y)
            return S

        self.add_source(gravity)

        def kernel(U):
            A = 10
            Delta = self.a / 30
            gaussian = -A * np.exp(-(r ** 2) / (2 * Delta ** 2))
            S = np.zeros_like(U)
            S[:, :, 0] = gaussian
            S[:, :, 1] = 0  # gaussian * np.sign(U[:, :, 1])
            S[:, :, 2] = 0  # gaussian # * np.sign(U[:, :, 2])
            S[:, :, 3] = 0  # gaussian
            return S

        self.add_source(kernel)

    def get_accr1_rate(self):
        g = self.num_g
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.x1, y - self.y1
        r = np.sqrt(dx ** 2 + dy ** 2)
        rho = self.U[:, :, 0]
        rate = np.sum(np.minimum(rho[g:-g, g:-g], self.gaussian_kernel(r)) * self.dx * self.dy)
        return rate

    def get_accr2_rate(self):
        g = self.num_g
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.x2, y - self.y2
        r = np.sqrt(dx ** 2 + dy ** 2)
        rho = self.U[:, :, 0]
        rate = np.sum(np.minimum(rho[g:-g, g:-g], self.gaussian_kernel(r)) * self.dx * self.dy)
        return rate

    def get_torque1(self):
        g = self.num_g
        rho = self.U[:, :, 0][g:-g, g:-g]
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.x1, y - self.y1
        r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
        dA = self.dx * self.dy
        Fg = (self.G * (self.M / 2) / (r ** 2 + self.eps ** 2)) * rho * dA
        Fg_x, Fg_y = (dx / (r + self.eps)) * Fg, (dy / (r + self.eps)) * Fg

        theta = np.arctan2(dy, dx)
        # theta-component of the gravitational force
        Fg_theta = -Fg_x * np.sin(theta) + Fg_y * np.cos(theta)

        T = np.sum((self.a / 2) * Fg_theta)

        return T

    def get_torque2(self):
        g = self.num_g
        rho = self.U[:, :, 0][g:-g, g:-g]
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.x2, y - self.y2
        r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
        dA = self.dx * self.dy
        Fg = (self.G * (self.M / 2) / (r ** 2 + self.eps ** 2)) * rho * dA
        Fg_x, Fg_y = (dx / (r + self.eps)) * Fg, (dy / (r + self.eps)) * Fg

        theta = np.arctan2(dy, dx)
        # theta-component of the gravitational force
        Fg_theta = -Fg_x * np.sin(theta) + Fg_y * np.cos(theta)

        T = np.sum((self.a / 2) * Fg_theta)

        return T

    def gaussian_kernel(self, r):
        A, Delta = 10, self.a / 50
        return A * np.exp(-(r ** 2) / (2 * Delta ** 2))

    def setup(self):
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        r, theta = cartesian_to_polar(x, y)
        # initial density profile from Santa Barbara paper

        def f(r):
            return 1
        rho = self.Sigma_0 * \
            ((1 - self.delta_0) * np.exp(-(self.R_cav /
             (r + self.eps)) ** 12) + self.delta_0) * f(r)
        v_k = np.sqrt(self.G * self.M / (r + self.eps))  # keplerian velocity
        cs = v_k / self.mach
        p = rho * (cs ** 2) / self.gamma

        u = - v_k * np.sin(theta)
        v = v_k * np.cos(theta)
        self.U = np.array([
            rho,
            rho * u,
            rho * v,
            E(self.gamma, rho, p, u, v)
        ]).transpose((1, 2, 0))

        def BH_gravity(U, bh_x, bh_y):
            dx, dy = x - bh_x, y - bh_y
            r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
            g = - self.G * (self.M / 2) / (r ** 2 + self.eps ** 2)

            g_x, g_y = (dx / (r + self.eps)) * g, (dy / (r + self.eps)) * g
            S = np.zeros_like(U)
            rho, u, v, p = get_prims(self.gamma, U)
            S[:, :, 1] = rho * g_x
            S[:, :, 2] = rho * g_y
            S[:, :, 3] = rho * (u * g_x + v * g_y)
            return S

        def gravity(U):
            return BH_gravity(U, self.x1, self.y1) + BH_gravity(U, self.x2, self.y2)

        self.add_source(gravity)

        def BH_kernel(U, bh_x, bh_y):
            dx, dy = x - bh_x, y - bh_y
            r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
            gaussian = self.gaussian_kernel(r)
            S = np.zeros_like(U)
            S[:, :, 0] = -gaussian
            S[:, :, 1] = 0  # -gaussian * np.sign(U[:, :, 1])
            S[:, :, 2] = 0  # -gaussian * np.sign(U[:, :, 2])
            S[:, :, 3] = 0  # -gaussian

            return S

        def kernel(U):
            return BH_kernel(U, self.x1, self.y1) + BH_kernel(U, self.x2, self.y2)

        self.add_source(kernel)

        self.add_diagnostic("accretion_rate_1", self.get_accr1_rate)
        self.add_diagnostic("accretion_rate_2", self.get_accr2_rate)
        self.add_diagnostic("torque_1", self.get_torque1)
        self.add_diagnostic("torque_2", self.get_torque2)

        """
        TODO:
            Figure out how to save accretion rate and gravitational torque data to file, either
                include it in the hdf file
                just export to txt/csv separately, but continuously throughout the simulation
            Calculate gravitational torque on each black hole, save in same way as above
            If you have more time, look into time average technique described in SB
        """
