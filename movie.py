from HD.helpers import plot_grid, plot_sheer, get_prims, E, P
from MHD.helpers import plot_grid as plot_MHD
import h5py
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


parser = ArgumentParser(description="Create a movie from a .hdf file.")

parser.add_argument('-f', '--file', type=str, required=True,
                    help='The path to the .hdf file (required)')

parser.add_argument('-o', '--output', type=str, required=True,
                    choices=['density', 'u', 'v', 'w', 'pressure', 'energy', 'Bx', 'By', 'Bz', 'div'],
                    help='The variable to plot: density, u, v, pressure or energy (required)')

args = parser.parse_args()

# validate the file argument
if not args.file.endswith('.hdf'):
    parser.error("The file name must end with '.hdf'")

PATH = args.file
var = args.output

with h5py.File(PATH, "r") as infile:
    dataset = infile["data"]
    history = dataset[...]
    t_vals = dataset.attrs["t"]
    gamma = dataset.attrs["gamma"]
    xmin, xmax = dataset.attrs["xrange"]
    ymin, ymax = dataset.attrs["yrange"]
    if "zrange" in dataset.attrs: 
        zmin, zmax = dataset.attrs["zrange"]
        
fig = plt.figure()
fps = 24
FFMpegWriter = animation.writers['ffmpeg']
file = os.path.splitext(os.path.basename(PATH))[0]
metadata = dict(title=file, comment='')
writer = FFMpegWriter(fps=fps, metadata=metadata)
PATH = f"./videos/{file}"
if not os.path.exists(PATH):
    os.makedirs(PATH)
cm = writer.saving(fig, f"{PATH}/{var}.mp4", 100)

with cm:
    if history.shape[-1] == 4: # HD
        vmin, vmax = 0, 2
        rho = np.maximum(history[-1, :, :, 0], 1e-6)
        u, v = history[-1, :, :, 1] / rho, history[-1, :, :, 2] / rho
        En = np.maximum(history[-1, :, :, 3], 1e-6)
        p = np.maximum(P(gamma, rho, u, v, En), 1e-6)
        if var == "density":
            vmin, vmax = np.min(rho), np.max(rho)
        elif var == "u":
            vmin, vmax = np.min(u), np.max(u)
        elif var == "v":
            vmin, vmax = np.min(v), np.max(v)
        elif var == "pressure":
            vmin, vmax = np.min(p), np.max(p)
        elif var == "energy":
            vmin, vmax = np.min(En), np.max(En)

    for i in range(len(history)):
        U = history[i]
        t = t_vals[i]
        fig.clear()
        if U.shape[-1] == 4: # HD
            plot_grid(gamma, U, t=t, plot=var, extent=[xmin, xmax, ymin, ymax], vmin=vmin, vmax=vmax)
        else: # MHD
            plot_MHD(gamma, U, t=t, plot=var, extent=[xmin, xmax, ymin, ymax])
        writer.grab_frame()

print("Movie saved to", PATH)
