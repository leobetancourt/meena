import numpy as np
import matplotlib.pyplot as plt
from HD.HD_1D import HD_1D
from HD.HD_2D import HD_2D, Boundary

if __name__ == "__main__":
    # sim = Sim_2D(gamma=5/3, resolution=(200, 200),
    #              xrange=(0, 1), yrange=(0, 1), solver="hll", high_space=True)
    # sim.set_bcs((Boundary.REFLECTIVE, Boundary.REFLECTIVE),
    #             (Boundary.REFLECTIVE, Boundary.REFLECTIVE))
    # sim.sedov_blast()
    # sim.run(T=2, plot="density", filename="sedov", save_interval=0.01)
    
    # sim = Sim_2D(gamma=5/3, resolution=(200, 200),
    #              xrange=(0, 1), yrange=(0, 1), solver="hllc", high_space=True)
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.REFLECTIVE, Boundary.REFLECTIVE))
    # sim.kelvin_helmholtz()
    # sim.run(T=2, plot="density", filename="KH", save_interval=0.01)

    # sim = Sim_2D(gamma=5/3, resolution=(75, 225),
    #              xrange=(0, 0.5), yrange=(0, 1.5), solver="hllc", high_space=False)
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.REFLECTIVE, Boundary.REFLECTIVE))
    # sim.rayleigh_taylor()
    # sim.run(T=15, plot="density", filename="RT", save_interval=0.1)

    sim = HD_2D(gamma=5/3, resolution=(300, 300), xrange=(-1, 1),
                 yrange=(-1, 1), solver="hll", high_space=False)
    sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
                (Boundary.OUTFLOW, Boundary.OUTFLOW))
    sim.kepler()
    sim.run(T=2, plot="energy", filename="kepler", save_interval=0.01)