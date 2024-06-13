from HD.helpers import plot_grid
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
                    choices=['density', 'u', 'v', 'w', 'pressure', 'energy', 'Bx', 'By', 'Bz'],
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
    zmin, zmax = dataset.attrs["zrange"]

fig = plt.figure()
fps = 12
FFMpegWriter = animation.writers['ffmpeg']
file = os.path.splitext(os.path.basename(PATH))[0]
metadata = dict(title=file, comment='')
writer = FFMpegWriter(fps=fps, metadata=metadata)
PATH = f"./videos/{file}"
if not os.path.exists(PATH):
    os.makedirs(PATH)
cm = writer.saving(fig, f"{PATH}/{var}.mp4", 100)

with cm:
    for i in range(len(history)):
        U = history[i]
        t = t_vals[i]
        fig.clear()
        if U.shape[-1] == 4: # HD
            plot_grid(gamma, U, t=t, plot=var, extent=[xmin, xmax, ymin, ymax])
        else: # MHD
            x = np.linspace(xmin, xmax,
                        num=U.shape[0], endpoint=False)
            plot_MHD(gamma, U, t=t, plot=var, extent=[xmin, xmax, ymin, ymax])
        writer.grab_frame()

print("Movie saved to", PATH)
