import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation_1D

if __name__ == "__main__":
    sim = Simulation_1D(gamma=1.4, resolution=200, dt=0.001, method="HLL", order="first")
    sim.sod_shock_tube()
    sim.run_simulation(T=0.2, xlabel="x", var="density")