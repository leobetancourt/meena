import numpy as np
import matplotlib.pyplot as plt
from sim1D import Sim_1D

if __name__ == "__main__":
    sim = Sim_1D(gamma=1.4, resolution=200, method="HLL", order="high")
    sim.sod_shock_tube()
    sim.run_simulation(T=0.2, xlabel="x", var="density")