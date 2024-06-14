import numpy as np
import matplotlib.pyplot as plt
from HD.HD_1D import HD_1D
from HD.HD_2D import HD_2D, Boundary

from MHD.MHD import MHD

if __name__ == "__main__":
    # sim = HD_2D(gamma=5/3, resolution=(200, 200),
    #              xrange=(0, 1), yrange=(0, 1), solver="hll", high_space=False)
    # sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
    #             (Boundary.OUTFLOW, Boundary.OUTFLOW))
    # sim.sedov_blast()
    # sim.run(T=2, filename="sedov", save_interval=0.01)

    # sim = HD_2D(gamma=5/3, resolution=(200, 200),
    #              xrange=(0, 1), yrange=(0, 1), solver="hllc", high_space=True)
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.REFLECTIVE, Boundary.REFLECTIVE))
    # sim.kelvin_helmholtz()
    # sim.run(T=2, plot="density", filename="KH", save_interval=0.01)

    # sim = HD_2D(gamma=5/3, resolution=(75, 225),
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
    sim.run(T=2, plot="density", filename="kepler", save_interval=0.01)

    # sim = MHD(gamma=2, resolution=(800, 1, 1), xrange=(-1, 1))
    # sim.shock_tube()
    # sim.run(T=0.2, plot="By", filename="BW", save_interval=0.01)

    # sim = MHD(gamma=5/3, resolution=(400, 600, 1), xrange=(-0.5, 0.5), yrange=(-0.75, 0.75))
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC))
    # sim.spherical_blast()
    # sim.run(T=1, plot="density", filename="MHD blast", save_interval=0.01)
