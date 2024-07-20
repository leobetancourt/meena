import numpy as np
import matplotlib.pyplot as plt
from HD.HD_1D import HD_1D
from HD.HD_2D import HD_2D, Boundary
from Binary import Binary

from MHD.MHD import MHD

if __name__ == "__main__":

    sim = Binary(coords="polar", resolution=(100, 600),
                 x1_range=(1, 10), x2_range=(0, 2 * np.pi), logspace=True)
    sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
                (Boundary.PERIODIC, Boundary.PERIODIC))
    sim.run(T=10 * 2 * np.pi, dt=0.0024,
            out="./output/binary", save_interval=0.2)

    # sim = MHD(gamma=2, resolution=(800, 1, 1), xrange=(-1, 1))
    # sim.shock_tube()
    # sim.run(T=0.2, plot="By", filename="BW", save_interval=0.01)

    # sim = MHD(gamma=5/3, resolution=(400, 600, 1), xrange=(-0.5, 0.5), yrange=(-0.75, 0.75))
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC))
    # sim.spherical_blast()
    # sim.run(T=1, plot="density", filename="MHD blast", save_interval=0.01)

    # sim = MHD(gamma=5/3, resolution=(512, 512, 1), xrange=(0, 1), yrange=(0, 1))
    # sim.set_bcs((Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC),
    #             (Boundary.PERIODIC, Boundary.PERIODIC))
    # sim.orszag_tang()
    # sim.run(T=1, plot="pressure", filename="Orszag-Tang", save_interval=0.01)

    # sim = HD_2D(gamma=5/3, nu=0, resolution=(200, 300), x1_range=(-1, 1), x2_range=(-1.5, 1.5), solver="hll")
    # sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
    #             (Boundary.OUTFLOW, Boundary.OUTFLOW))
    # sim.sheer()
    # sim.run(T=1, plot="v", filename="sheer1", save_interval=0.01)

    # sim = HD_2D(gamma=5/3, nu=1e-3, coords="polar", resolution=(100, 600),
    #             x1_range=(0.05, 1), x2_range=(0, 2 * np.pi), logspace=True, solver="hll")
    # sim.set_bcs((Boundary.OUTFLOW, Boundary.OUTFLOW),
    #             (Boundary.PERIODIC, Boundary.PERIODIC))
    # sim.sedov_blast(radius=0.1)
    # sim.run(T=1, out="./output", save_interval=0.01)
