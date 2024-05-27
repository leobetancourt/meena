import numpy as np
import matplotlib.pyplot as plt
from sim.sim1D import Sim_1D
from sim.sim2D import Sim_2D

if __name__ == "__main__":
    # sim = Sim_1D(gamma=1.4, resolution=200, polar=False, method="HLL", order="first")
    # sim.sod_shock_tube()
    # sim.run_simulation(T=0.2, xlabel="x", var="density")

    # sim = Sim_2D(gamma=1.4, resolution=(300, 300), xrange=(0, 1), yrange=(0, 1))
    # sim.sedov_blast(radius=0.5)
    # sim.run_simulation(T=0.1, var="density", filename="sedov")


    sim = Sim_2D(gamma=5/3, resolution=(100, 300),
                 xrange=(0, 0.5), yrange=(0, 1.5))

    def gravity(U):
        S = np.zeros_like(U)
        g = -0.1
        rho = U[:, :, 0]
        u = U[:, :, 1] / rho
        v = U[:, :, 2] / rho
        S[:, :, 2] = g * rho
        S[:, :, 3] = g * np.multiply(rho, v)
        return S
    
    sim.add_source(gravity)
    sim.rayleigh_taylor()
    sim.run_simulation(T=15, var="density", filename="RT")