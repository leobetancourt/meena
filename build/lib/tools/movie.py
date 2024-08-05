from common.helpers import plot_grid, print_progress_bar
import os
import h5py
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['figure.dpi'] = 800
plt.rcParams['savefig.dpi'] = 800


def get_h5_files_in_range(directory, t_min, t_max):
    files_in_range = []
    pattern = re.compile(r'out_(\d*\.?\d*)\.h5')

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            try:
                # Extracting the time value from filename
                t = float(match.group(1))
                if t_min <= t <= t_max:
                    files_in_range.append(os.path.join(directory, filename))
            except ValueError:
                pass  # If conversion to float fails, ignore this file

    return sorted(files_in_range, key=lambda x: float(re.match(pattern, os.path.basename(x)).group(1)))


def process_h5_files(file_list):
    with h5py.File(file_list[0], 'r') as f:
        coords = f.attrs["coords"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        t = f.attrs["t"] / (2 * np.pi)
        matrix = np.log10(f["rho"][...])

    vmin, vmax = -3, 0.5
    fig, ax, c, cb = plot_grid(
        matrix, r"$\log_{10} \Sigma$", coords=coords, x1=x1, x2=x2, vmin=vmin, vmax=vmax)
    ax.set_title(f"t = {t:.2f}")
    fps = 24
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=fps)
    PATH = f"./visual"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    cm = writer.saving(fig, f"{PATH}/movie.mp4", 800)

    with cm:
        for i in range(len(file_list)):
            file_path = file_list[i]
            with h5py.File(file_path, 'r') as f:
                t = f.attrs["t"] / (2 * np.pi)
                matrix = np.log10(f["rho"][...])
                c.set_array(matrix.ravel())
                cb.update_normal(c)
                ax.set_title(f"t = {t:.2f}")
                fig.canvas.draw()
                writer.grab_frame()

                print_progress_bar(i, len(file_list),
                                   suffix="complete", length=25)


PATH = "./500x3000/checkpoints"
t_min = 290 * 2 * np.pi
t_max = 300 * 2 * np.pi

files = get_h5_files_in_range(PATH, t_min, t_max)
process_h5_files(files)
