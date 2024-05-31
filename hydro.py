import numpy as np
import matplotlib.pyplot as plt
from sim.sim1D import Sim_1D
from sim.sim2D import Sim_2D, Boundary

if __name__ == "__main__":
    # sim = Sim_1D(gamma=1.4, resolution=200, polar=False, method="HLL", order="first")
    # sim.sod_shock_tube()
    # sim.run(T=0.2, xlabel="x", var="density")

    sim = Sim_2D(gamma=1.4, resolution=(600, 600),
                 xrange=(0, 1), yrange=(0, 1), solver="hllc")
    sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
                (Boundary.PERIODIC, Boundary.PERIODIC))
    sim.sedov_blast(radius=0.3)
    sim.run(T=1, plot="density", filename="sedov", save_interval=0.005)

    # sim = Sim_2D(gamma=5/3, resolution=(400, 400),
    #              xrange=(0, 1), yrange=(0, 1))
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.REFLECTIVE, Boundary.REFLECTIVE))
    # sim.kelvin_helmholtz()
    # sim.run(T=2, plot="density", filename="KH")

    # sim = Sim_2D(gamma=5/3, resolution=(100, 300),
    #              xrange=(0, 0.5), yrange=(0, 1.5), solver="hllc")

    # def gravity(U):
    #     S = np.zeros_like(U)
    #     g = -0.1
    #     rho = U[:, :, 0]
    #     u = U[:, :, 1] / rho
    #     v = U[:, :, 2] / rho
    #     S[:, :, 2] = g * rho
    #     S[:, :, 3] = g * np.multiply(rho, v)
    #     return S

    # sim.add_source(gravity)
    # sim.rayleigh_taylor()
    # sim.run(T=15, plot="density", filename="RT", save_interval=0.1)
