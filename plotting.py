from HD.helpers import plot_grid, plot_sheer, get_prims, E, P
import h5py
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

parser = ArgumentParser(description="Plot data from an .hdf file.")

parser.add_argument('-f', '--file', type=str, required=True,
                    help='The path to the .hdf file (required)')

parser.add_argument('-o', '--output', type=str, required=True,
                    choices=["accretion_rate_1", "accretion_rate_2", "torque_1", "torque_2"],
                    help='The variable to plot: accretion_rate_1, accretion_rate_2, torque_1, torque_2 (required)')

args = parser.parse_args()

# validate the file argument
if not args.file.endswith('.hdf'):
    parser.error("The file name must end with '.hdf'")

PATH = args.file
var = args.output

with h5py.File(PATH, "r") as f:
    gamma = f.attrs["gamma"]
    xmin, xmax = f.attrs["xrange"]
    ymin, ymax = f.attrs["yrange"]
    if "zrange" in f.attrs: 
        zmin, zmax = f.attrs["zrange"]
    
    t = f["t"][...] # simulation times
    tc = f["tc"][...] # checkpoint times
    data = f[var][...]
    print(data)
    
    if len(data.shape) == 1:
        plt.scatter(t, data, linewidth=1, s=0.5, c="black")
        
    plt.title("Torque (BH 2)")
    plt.xlabel("time (orbits)")
    plt.savefig(f"./visual/{var}.png")
    plt.show()