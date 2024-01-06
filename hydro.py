import numpy as np
import matplotlib.pyplot as plt
from sim1D import Sim_1D

if __name__ == "__main__":
    sim = Sim_1D(gamma=1.4, resolution=200, polar=True, method="HLL", order="first")
    sim.sedov_blast()
    sim.run_simulation(T=0.179, xlabel="r", var="density")