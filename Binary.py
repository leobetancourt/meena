import numpy as np
from HD.HD_2D import HD_2D, Boundary
from HD.helpers import E, P, cartesian_to_polar, polar_to_cartesian, get_prims


class Binary(HD_2D):
    def __init__(self, gamma=1.4, coords="cartesian", resolution=(100, 100),
                 x1_range=(-5, 5), x2_range=(-5, 5), logspace=False, solver="hll", high_space=False):
        self.G, self.M = 1, 1
        self.mach = 10
        self.a = 1  # binary separation
        self.R_cav = 2.5 * self.a
        self.R_out = 10 * self.a
        self.delta_0 = 1e-5
        self.Sigma_0 = 1
        self.eps = 0.05 * self.a  # gravitational softening
        self.period = 1
        self.omega = 2 * np.pi / self.period
        if coords == "cartesian":
            self.bh1_x, self.bh1_y = self.a / 2, 0
            self.bh2_x, self.bh2_y = -self.a / 2, 0
        elif coords == "polar":
            self.bh1_r, self.bh1_theta = self.a / 2, 0
            self.bh2_r, self.bh2_theta = self.a / 2, np.pi
        self.ts = []
        self.accr_1 = []
        self.accr_2 = []
        nu = 1e-3 * self.a ** 2 * self.omega

        super().__init__(gamma, nu=nu, coords=coords, resolution=resolution, x1_range=x1_range,
                         x2_range=x2_range, solver=solver, logspace=logspace, high_space=high_space)
        self.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
                     (Boundary.OUTFLOW, Boundary.OUTFLOW))

        if self.coords == "cartesian":
            self.setup_cartesian()
        elif self.coords == "polar":
            self.setup_polar()

        # self.add_diagnostic("accretion_rate_1", self.get_accr1_rate)
        # self.add_diagnostic("accretion_rate_2", self.get_accr2_rate)
        # self.add_diagnostic("torque_1", self.get_torque1)
        # self.add_diagnostic("torque_2", self.get_torque2)

    def gaussian_kernel(self, r):
        A, Delta = 10, self.a / 50
        return A * np.exp(-(r ** 2) / (2 * Delta ** 2))

    def buffer(self):
        g = self.num_g
        if self.coords == "cartesian":
            x, y = self.grid[:, :, 0], self.grid[:, :, 1]
            r, theta = cartesian_to_polar(x, y)
            buff = r >= (0.95 * self.x1_max)

            v = np.sqrt(self.G * self.M / r[buff])
            cs = v / self.mach
            u_k, v_k = - v * np.sin(theta[buff]), v * np.cos(theta[buff])
            rho = self.U[g:-g, g:-g, 0][buff]
            p = rho * (cs ** 2) / self.gamma

            self.U[g:-g, g:-g, 1][buff] = rho * u_k
            self.U[g:-g, g:-g, 2][buff] = rho * v_k
            self.U[g:-g, g:-g, 3][buff] = E(self.gamma, rho, p, u_k, v_k)
        elif self.coords == "polar":  # do i need a buffer in polar coordinates?
            r, theta = self.grid[:, :, 0], self.grid[:, :, 1]
            buff = r >= (0.95 * self.x1_max)

            v = np.sqrt(self.G * self.M / r[buff])
            cs = v / self.mach
            rho = self.U[g:-g, g:-g, 0][buff]
            p = rho * (cs ** 2) / self.gamma

            self.U[g:-g, g:-g, 2][buff] = rho * v
            self.U[g:-g, g:-g, 3][buff] = E(self.gamma, rho, p, 0, v)

    def first_order_step(self, t):
        g = self.num_g
        # update black hole positions
        delta = np.pi
        if self.coords == "cartesian":
            self.bh1_x = (self.a / 2) * np.cos(self.omega * t)
            self.bh1_y = (self.a / 2) * np.sin(self.omega * t)
            self.bh2_x = (self.a / 2) * np.cos(self.omega * t + delta)
            self.bh2_y = (self.a / 2) * np.sin(self.omega * t + delta)
        elif self.coords == "polar":
            self.bh1_theta = self.omega * t
            self.bh2_theta = self.omega * t + delta

        # set buffer velocities
        self.buffer()

        # ensure radial velocity at excised boundary is not positive
        # if self.coords == "polar":
        #     rho, vr, vtheta, p = get_prims(self.gamma, self.U)
        #     vr = np.minimum(vr[:g, :], 0)
        #     self.U[:g, :, 1] = rho[:g, :] * vr

        self.ts.append(t)

        super().first_order_step(t)

    def get_accr1_rate(self):
        g = self.num_g
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.x1, y - self.y1
        r = np.sqrt(dx ** 2 + dy ** 2)
        rho = self.U[:, :, 0]
        rate = np.sum(np.minimum(
            rho[g:-g, g:-g], self.gaussian_kernel(r)) * self.dx1 * self.dx2)

        return rate

    def get_accr2_rate(self):
        g = self.num_g
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.x2, y - self.x2
        r = np.sqrt(dx ** 2 + dy ** 2)
        rho = self.U[:, :, 0]
        rate = np.sum(np.minimum(
            rho[g:-g, g:-g], self.gaussian_kernel(r)) * self.dx1 * self.dx2)
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
        dx, dy = x - self.x1, y - self.x2
        r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
        dA = self.dx1 * self.dx2
        Fg = (self.G * (self.M / 2) / (r ** 2 + self.eps ** 2)) * rho * dA
        Fg_x, Fg_y = (dx / (r + self.eps)) * Fg, (dy / (r + self.eps)) * Fg

        theta = np.arctan2(dy, dx)
        # theta-component of the gravitational force
        Fg_theta = -Fg_x * np.sin(theta) + Fg_y * np.cos(theta)

        T = np.sum((self.a / 2) * Fg_theta)

        return T

    def setup_polar(self):
        r, theta = self.grid[:, :, 0], self.grid[:, :, 1]

        def f(r):
            return 1
        rho = self.Sigma_0 * \
            ((1 - self.delta_0) * np.exp(-(self.R_cav /
                                           (r + self.eps)) ** 12) + self.delta_0) * f(r)
        v_k = np.sqrt(self.G * self.M / (r + self.eps))  # keplerian velocity
        cs = v_k / self.mach
        p = rho * (cs ** 2) / self.gamma

        self.U = np.array([
            rho,
            np.zeros_like(rho),
            rho * v_k,
            E(self.gamma, rho, p, 0, v_k)
        ]).transpose((1, 2, 0))

        def BH_gravity(U, bh_r, bh_theta):
            # distance from each zone to bh
            dist = np.sqrt(r ** 2 + bh_r ** 2 - 2 * r *
                           bh_r * np.cos(theta - bh_theta))
            g = - self.G * (self.M / 2) / (dist ** 2 + self.eps ** 2)

            # angle that gravity vector makes from the x axis
            phi = np.arctan2((r * np.sin(theta) - bh_r * np.sin(bh_theta)), (r * np.cos(theta) - bh_r * np.cos(bh_theta)))
            
            g_r = g * np.cos(phi - theta)
            g_theta = g * np.sin(phi - theta)

            S = np.zeros_like(U)
            rho, u, v, p = get_prims(self.gamma, U)

            S[:, :, 1] = rho * g_r
            S[:, :, 2] = rho * g_theta
            S[:, :, 3] = rho * (u * g_r + v * g_theta)

            return S

        def gravity(U):
            return BH_gravity(U, self.bh1_r, self.bh1_theta) + BH_gravity(U, self.bh2_r, self.bh2_theta)

        self.add_source(gravity)

    def setup_cartesian(self):
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        r, theta = cartesian_to_polar(x, y)

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
            dist = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
            g = - self.G * (self.M / 2) / (dist ** 2 + self.eps ** 2)

            g_x, g_y = (dx / (dist + self.eps)) * \
                g, (dy / (dist + self.eps)) * g
            S = np.zeros_like(U)
            rho, u, v, p = get_prims(self.gamma, U)

            S[:, :, 1] = rho * g_x
            S[:, :, 2] = rho * g_y
            S[:, :, 3] = rho * (u * g_x + v * g_y)

            return S

        def gravity(U):
            return BH_gravity(U, self.bh1_x, self.bh1_y) + BH_gravity(U, self.bh2_x, self.bh2_y)

        self.add_source(gravity)

        def BH_kernel(U, bh_x, bh_y):
            dx, dy = x - bh_x, y - bh_y
            r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
            gaussian = self.gaussian_kernel(r)
            S = np.zeros_like(U)
            S[:, :, 0] = -gaussian

            return S

        def kernel(U):
            return BH_kernel(U, self.bh1_x, self.bh1_y) + BH_kernel(U, self.bh2_x, self.bh2_y)

        self.add_source(kernel)
