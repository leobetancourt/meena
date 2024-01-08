import numpy as np
import matplotlib.pyplot as plt
from sim1D import Sim_1D
from sim2D import Sim_2D

if __name__ == "__main__":
    # sim = Sim_1D(gamma=1.4, resolution=200, polar=False, method="HLL", order="first")
    # sim.sod_shock_tube()
    # sim.run_simulation(T=0.2, xlabel="x", var="density")

    # sim = Sim_2D(gamma=1.4, resolution=(200, 200))
    # sim.sedov_blast()
    # sim.run_simulation(T=0.1, var="pressure", filename="sedov")

    def gravity(U):
        S = np.zeros((50, 200, 4))
        g = -1
        rho = U[:, :, 0]
        u = U[:, :, 1] / rho
        v = U[:, :, 2] / rho
        S[:, :, 2] = g * rho
        S[:, :, 3] = g * np.multiply(rho, v)
        return S
        
    sim = Sim_2D(gamma=1.4, resolution=(50, 200),
                 xrange=(-0.25, 0.25), yrange=(0, 1))
    sim.add_source(gravity)
    sim.rayleigh_taylor()
    sim.run_simulation(T=1, var="density", filename="RT")
    

    