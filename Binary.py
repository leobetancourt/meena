import numpy as np
from HD.HD_2D import HD_2D, Boundary
from HD.helpers import E, cartesian_to_polar, get_prims


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
        self.omega_B = 1
        if coords == "cartesian":
            self.bh1_x, self.bh1_y = self.a / 2, 0
            self.bh2_x, self.bh2_y = -self.a / 2, 0
        elif coords == "polar":
            self.bh1_r, self.bh1_theta = self.a / 2, 0
            self.bh2_r, self.bh2_theta = self.a / 2, np.pi
        self.ts = []
        self.accr_1 = []
        self.accr_2 = []
        nu = 1e-3 * self.a ** 2 * self.omega_B

        super().__init__(gamma, nu=nu, coords=coords, resolution=resolution, x1_range=x1_range,
                         x2_range=x2_range, solver=solver, logspace=logspace, high_space=high_space)

        if self.coords == "cartesian":
            self.setup_cartesian()

            self.add_diagnostic("accretion_rate_1", self.get_accr1_rate)
            self.add_diagnostic("accretion_rate_2", self.get_accr2_rate)
            self.add_diagnostic("torque_1", self.get_torque1)
            self.add_diagnostic("torque_2", self.get_torque2)
        elif self.coords == "polar":
            self.setup_polar()

            self.add_diagnostic("accretion_rate", self.get_accr_rate)
            self.add_diagnostic("torque_1", self.get_torque1)
            self.add_diagnostic("torque_2", self.get_torque2)

    def c_s(self, r=None, theta=None):
        if r is None:
            r, theta = self.grid[:, :, 0], self.grid[:, :, 1]
        dist1 = np.sqrt(r ** 2 + self.bh1_r ** 2 - 2 * r *
                        self.bh1_r * np.cos(theta - self.bh1_theta))
        dist2 = np.sqrt(r ** 2 + self.bh2_r ** 2 - 2 * r *
                        self.bh2_r * np.cos(theta - self.bh2_theta))
        return np.sqrt((self.G * (self.M / 2) / np.sqrt(dist1 ** 2 + self.eps ** 2) + self.G * (self.M / 2) / np.sqrt(dist2 ** 2 + self.eps ** 2)) / (self.mach ** 2))

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

    def first_order_step(self, t):
        g = self.num_g
        # update black hole positions
        delta = np.pi
        if self.coords == "cartesian":
            self.bh1_x = (self.a / 2) * np.cos(self.omega_B * t)
            self.bh1_y = (self.a / 2) * np.sin(self.omega_B * t)
            self.bh2_x = (self.a / 2) * np.cos(self.omega_B * t + delta)
            self.bh2_y = (self.a / 2) * np.sin(self.omega_B * t + delta)
        elif self.coords == "polar":
            self.bh1_theta = self.omega_B * t
            self.bh2_theta = self.omega_B * t + delta

        # set buffer velocities
        self.buffer()

        # ensure radial velocity at excised boundary is not positive
        if self.coords == "polar":
            rho, vr, vtheta, p = get_prims(self.gamma, self.U)
            vr[:g, :] = np.minimum(vr[:g, :], 0)
            self.U[:g, :, 1] = rho[:g, :] * vr[:g, :]

        self.ts.append(t)

        super().first_order_step(t)

    def setup_polar(self):
        r, theta = self.grid[:, :, 0], self.grid[:, :, 1]

        def f(r):
            return 1
        # surface density
        rho = self.Sigma_0 * \
            ((1 - self.delta_0) * np.exp(-(self.R_cav /
                                           (r + self.eps)) ** 12) + self.delta_0) * f(r)

        # radial velocity 'kick'
        v_naught = 1e-4 * self.omega_B * self.a
        v_r = v_naught * np.sin(theta) * (r / self.a) * \
            np.exp(-(r / (3.5 * self.a)) ** 6)
        # angular frequency of gas in the disk
        omega_naught = np.sqrt((self.G * self.M / (r ** 3))
                               * (1 - (1 / (self.mach ** 2))))
        omega = ((omega_naught ** -4) + (self.omega_B ** -4)) ** (-1/4)
        v_theta = omega * r

        p = rho * (self.c_s() ** 2) / self.gamma

        self.U = np.array([
            rho,
            rho * v_r,
            rho * v_theta,
            E(self.gamma, rho, p, v_r, v_theta)
        ]).transpose((1, 2, 0))

        def BH_gravity(U, bh_r, bh_theta):
            # distance from each zone to bh
            delta_theta = theta - bh_theta
            dist = np.sqrt(r ** 2 + bh_r ** 2 - 2 * r *
                           bh_r * np.cos(delta_theta))
            g = -self.G * (self.M / 2) / (dist ** 2 + self.eps ** 2)

            g_r = g * (r - bh_r * np.cos(delta_theta)) / dist
            g_theta = g * (bh_r * np.sin(delta_theta)) / dist

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
            g = - 2 * np.pi * self.G * (self.M / 2) / (dist ** 2 + self.eps ** 2)

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

    def get_accr_rate(self):
        g = self.num_g
        if self.coords == "cartesian":
            return self.get_accr1_rate() + self.get_accr2_rate()
        elif self.coords == "polar":
            # calculate amount of mass that will cross excised boundary
            rho, v_r, v_theta, p = get_prims(self.gamma, self.U[g:-g, g:-g])
            rho_in = rho[0, :]
            v_r_in = v_r[0, :]
            dx1 = self.x1_interf[1] - self.x1_interf[0]
            dx2 = self.x2_interf[1] - self.x2_interf[0]

            return np.sum(rho_in[v_r_in < 0] * self.x1_interf[0] * dx1 * dx2)

    def get_accr1_rate(self):
        g = self.num_g
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.bh1_x, y - self.bh1_y
        r = np.sqrt(dx ** 2 + dy ** 2)
        rho = self.U[:, :, 0]
        rate = np.sum(np.minimum(
            rho[g:-g, g:-g], self.gaussian_kernel(r)) * self.dx1 * self.dx2)

        return rate

    def get_accr2_rate(self):
        g = self.num_g
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        dx, dy = x - self.bh2_x, y - self.bh2_y
        r = np.sqrt(dx ** 2 + dy ** 2)
        rho = self.U[:, :, 0]
        rate = np.sum(np.minimum(
            rho[g:-g, g:-g], self.gaussian_kernel(r)) * self.dx1 * self.dx2)

        return rate

    def get_torque1(self):
        g = self.num_g
        rho = self.U[:, :, 0][g:-g, g:-g]

        if self.coords == "cartesian":
            x, y = self.grid[:, :, 0], self.grid[:, :, 1]
            bh_x, bh_y = self.bh1_x, self.bh1_y
            dx, dy = x - bh_x, y - bh_y
            r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
            dA = self.dx1 * self.dx2
            Fg = (self.G * (self.M / 2) / (r ** 2 + self.eps ** 2)) * rho * dA
            Fg_x, Fg_y = (dx / (r + self.eps)) * Fg, (dy / (r + self.eps)) * Fg
            theta = np.arctan2(dy, dx)
            # theta-component of the gravitational force
            Fg_theta = -Fg_x * np.sin(theta) + Fg_y * np.cos(theta)

            T = np.sum((self.a / 2) * Fg_theta)
        elif self.coords == "polar":
            r, theta = self.grid[:, :, 0], self.grid[:, :, 1]
            bh_r, bh_theta = self.bh1_r, self.bh1_theta
            delta_theta = theta - bh_theta
            dist = np.sqrt(r ** 2 + bh_r ** 2 - 2 * r *
                           bh_r * np.cos(delta_theta))
            R_interf, _ = np.meshgrid(self.x1_interf, self.x2, indexing="ij")
            x1_l, x1_r = R_interf[:-1, :], R_interf[1:, :]
            dx1 = x1_r - x1_l
            dx2 = self.x2[1] - self.x2[0]
            dA = r * dx1 * dx2  # dA = rdrdtheta

            Fg = (self.G * (self.M / 2) / (dist ** 2 + self.eps ** 2)) * rho * dA
            Fg_theta = Fg * (bh_r * np.sin(delta_theta)) / dist 
            T = np.sum((self.a / 2) * Fg_theta)

        return T

    def get_torque2(self):
        g = self.num_g
        rho = self.U[:, :, 0][g:-g, g:-g]

        if self.coords == "cartesian":
            x, y = self.grid[:, :, 0], self.grid[:, :, 1]
            bh_x, bh_y = self.bh2_x, self.bh2_y
            dx, dy = x - bh_x, y - bh_y
            r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
            dA = self.dx1 * self.dx2
            Fg = (self.G * (self.M / 2) / (r ** 2 + self.eps ** 2)) * rho * dA
            Fg_x, Fg_y = (dx / (r + self.eps)) * Fg, (dy / (r + self.eps)) * Fg

            theta = np.arctan2(dy, dx)
            # theta-component of the gravitational force
            Fg_theta = -Fg_x * np.sin(theta) + Fg_y * np.cos(theta)

            T = np.sum((self.a / 2) * Fg_theta)
        elif self.coords == "polar":
            r, theta = self.grid[:, :, 0], self.grid[:, :, 1]
            bh_r, bh_theta = self.bh2_r, self.bh2_theta
            delta_theta = theta - bh_theta
            dist = np.sqrt(r ** 2 + bh_r ** 2 - 2 * r *
                           bh_r * np.cos(delta_theta))
            R_interf, _ = np.meshgrid(self.x1_interf, self.x2, indexing="ij")
            x1_l, x1_r = R_interf[:-1, :], R_interf[1:, :]
            dx1 = x1_r - x1_l
            dx2 = self.x2[1] - self.x2[0]
            dA = r * dx1 * dx2  # dA = rdrdtheta

            Fg = (self.G * (self.M / 2) / (dist ** 2 + self.eps ** 2)) * rho * dA
            Fg_theta = Fg * (bh_r * np.sin(delta_theta)) / dist 
            T = np.sum((self.a / 2) * Fg_theta)

        return T
