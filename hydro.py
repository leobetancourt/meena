import numpy as np
import matplotlib.pyplot as plt
from HD.HD_1D import HD_1D
from HD.HD_2D import HD_2D, Boundary
from Binary import Binary

from MHD.MHD import MHD

if __name__ == "__main__":

    # sim = HD_2D(gamma=5/3, resolution=(400, 400), xrange=(-1, 1),
    #             yrange=(-1, 1), solver="hll", high_space=False)
    # sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
    #             (Boundary.OUTFLOW, Boundary.OUTFLOW))
    # sim.kepler()
    # sim.run(T=4, plot="density", filename="kepler", save_interval=0.01)
    
    sim = Binary(gamma=5/3, resolution=(400, 400))
    sim.run(T=4, plot="density", filename="binary", save_interval=0.01)

    # sim = MHD(gamma=2, resolution=(800, 1, 1), xrange=(-1, 1))
    # sim.shock_tube()
    # sim.run(T=0.2, plot="By", filename="BW", save_interval=0.01)

    # sim = MHD(gamma=5/3, resolution=(400, 600, 1), xrange=(-0.5, 0.5), yrange=(-0.75, 0.75))
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC))
    # sim.spherical_blast()
    # sim.run(T=1, plot="density", filename="MHD blast", save_interval=0.01)
