import numpy as np
import matplotlib.pyplot as plt
from sim.sim1D import Sim_1D
from sim.sim2D import Sim_2D, Boundary

if __name__ == "__main__":
    # sim = Sim_2D(gamma=5/3, resolution=(200, 200),
    #              xrange=(0, 1), yrange=(0, 1), solver="hll", high_space=True)
    # sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
    #             (Boundary.OUTFLOW, Boundary.OUTFLOW))
    # sim.sedov_blast()
    # sim.run(T=2, plot="density", filename="sedov", save_interval=0.01)
    
    sim = Sim_2D(gamma=5/3, resolution=(200, 200),
                 xrange=(0, 1), yrange=(0, 1), solver="hllc", high_space=True)
    sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
                (Boundary.REFLECTIVE, Boundary.REFLECTIVE))
    sim.kelvin_helmholtz()
    sim.run(T=2, plot="pressure", filename="sedov", save_interval=0.01)

    # sim = Sim_2D(gamma=5/3, resolution=(75, 225),
    #              xrange=(0, 0.5), yrange=(0, 1.5), solver="hllc", high_space=False)
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.REFLECTIVE, Boundary.REFLECTIVE))

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

    # sim = Sim_2D(gamma=5/3, resolution=(300, 300), xrange=(-1, 1),
    #              yrange=(-1, 1), solver="hll", high_space=True)
    # sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
    #             (Boundary.OUTFLOW, Boundary.OUTFLOW))
    # sim.kepler()
    # sim.run(T=2, plot="density", filename="kepler", save_interval=0.01)