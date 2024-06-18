import numpy as np
from HD.HD_2D import HD_2D, Boundary
from HD.helpers import E, cartesian_to_polar, get_prims


class Binary(HD_2D):
    def __init__(self, gamma=1.4, resolution=(100, 100),
                 xrange=(-2.5, 2.5), yrange=(-2.5, 2.5), solver="hll", high_space=False):
        super().__init__(gamma, resolution, xrange=xrange,
                         yrange=yrange, solver=solver, high_space=high_space)
        self.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
                     (Boundary.OUTFLOW, Boundary.OUTFLOW))

        self.G, self.M = 1, 1
        self.mach = 10
        self.a = 1 # binary separation
        self.eps = 0.05 * self.a # gravitational softening
        self.period = 1
        self.omega = 2 * np.pi / self.period
        self.x1, self.y1 = self.a / 2, 0
        self.x2, self.y2 = -self.a / 2, 0

        self.setup()
        
    def setup(self):
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        r, theta = cartesian_to_polar(x, y)
        rho = np.ones_like(r) * 1
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
            r = np.sqrt(dx ** 2 + dy ** 2) + self.eps  # distance from each zone to bh
            g = - self.G * (self.M / 2) / ((r + self.eps) ** 2)

            g_x, g_y = (dx / r) * g, (dy / r) * g
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
            A = 10
            Delta = (self.xmax - self.xmin) / 100
            dx, dy = x - bh_x, y - bh_y
            r = np.sqrt(dx ** 2 + dy ** 2)  # distance from each zone to bh
            gaussian = -A * np.exp(-(r ** 2) / (2 * Delta ** 2))
            S = np.zeros_like(U)
            S[:, :, 0] = gaussian
            S[:, :, 1] = 0 # gaussian * np.sign(U[:, :, 1])
            S[:, :, 2] = 0 # gaussian # * np.sign(U[:, :, 2])
            S[:, :, 3] = 0 # gaussian
            return S
        
        def kernel(U):
            return BH_kernel(U, self.x1, self.y1) + BH_kernel(U, self.x2, self.y2)
        
        self.add_source(kernel)
    
    def buffer(self):
        g = self.num_g
        # for zones at r >= 2.5a, make their velocities keplerian
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        r, theta = cartesian_to_polar(x, y)
        
        buff = r >= 2.5 * self.a
        v = np.sqrt(self.G * self.M / (r[buff] + self.eps))
        u_k, v_k = - v * np.sin(theta[buff]), v * np.cos(theta[buff])
        
        rho = self.U[g:-g, g:-g, 0][buff]
        self.U[g:-g, g:-g, 1][buff] = rho * u_k
        self.U[g:-g, g:-g, 2][buff] = rho * v_k

    def first_order_step(self, t):
        # update black hole positions
        delta = np.pi
        self.x1 = (self.a / 2) * np.cos(self.omega * t)
        self.y1 = (self.a / 2) * np.sin(self.omega * t)
        self.x2 = (self.a / 2) * np.cos(self.omega * t + delta)
        self.y2 = (self.a / 2) * np.sin(self.omega * t + delta)
        
        # set buffer velocities
        self.buffer()

        super().first_order_step(t)
