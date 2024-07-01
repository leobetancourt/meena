from HD.helpers import plot_grid, plot_sheer, get_prims, E, P
from MHD.helpers import plot_grid as plot_MHD
import h5py
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


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

with h5py.File(PATH, "r") as f:
    gamma = f.attrs["gamma"]
    xmin, xmax = f.attrs["xrange"]
    ymin, ymax = f.attrs["yrange"]
    if "zrange" in f.attrs: 
        zmin, zmax = f.attrs["zrange"]
    
    tc = f["tc"][...] # checkpoint times
    rho, momx, momy, En = f["rho"][...], f["momx"][...], f["momy"][...], f["E"][...]
    momz, Bx, By, Bz = None, None, None, None
    if "Bx" in f: # MHD
        momz, Bx, By, Bz = f["momz"], f["Bx"], f["By"], f["Bz"]

        
fig = plt.figure()
fps = 24
FFMpegWriter = animation.writers['ffmpeg']
file = os.path.splitext(os.path.basename(PATH))[0]
metadata = dict(title=file, comment='')
writer = FFMpegWriter(fps=fps, metadata=metadata)
PATH = f"./visual/{file}"
if not os.path.exists(PATH):
    os.makedirs(PATH)
cm = writer.saving(fig, f"{PATH}/{var}.mp4", 300)

with cm:
    if not Bx: # HD
        vmin, vmax = 0, 2
        rho = np.maximum(rho, 1e-6)
        u, v = momx / rho, momy / rho
        En = np.maximum(En, 1e-6)
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

    for i in range(len(tc)): # loop over checkpoints
        if Bx: 
            U = np.array([
                rho[i],
                momx[i],
                momy[i],
                momz[i],
                Bx[i],
                By[i],
                Bz[i],
                En[i]
            ]).transpose((1, 2, 3, 0))
        else:
            U = np.array([
                rho[i],
                momx[i],
                momy[i],
                En[i]
            ]).transpose((1, 2, 0))
            
        fig.clear()
        if Bx: # MHD
            plot_MHD(gamma, U, t=tc[i], plot=var, extent=[xmin, xmax, ymin, ymax])
        else: # HD
            plot_grid(gamma, U, t=tc[i], plot=var, extent=[xmin, xmax, ymin, ymax], vmin=vmin, vmax=vmax)
        writer.grab_frame()

print("Movie saved to", PATH)
