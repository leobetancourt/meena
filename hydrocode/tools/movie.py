from src.common.helpers import plot_grid, print_progress_bar
import os
import h5py
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def generate_movie(checkpoint_path, t_min, t_max, var, title, fps=24, vmin=None, vmax=None):
    file_list = get_h5_files_in_range(checkpoint_path, t_min, t_max)
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$", "v": r"$v$", "energy": r"$E$"}
    with h5py.File(file_list[0], 'r') as f:
        coords = f.attrs["coords"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        t = f.attrs["t"]
        rho, momx1, momx2, e = np.array(f["rho"]), np.array(f["momx1"]), np.array(f["momx2"]), np.array(f["E"])
        if var == "density":
            matrix = rho
        elif var == "log density":
            matrix = np.log10(rho)
        elif var == "u":
            matrix = momx1 / rho
        elif var == "v":
            matrix = momx2 / rho
        elif var == "energy":
            matrix = e

    fig, ax, c, cb = plot_grid(
        matrix, labels[var], coords=coords, x1=x1, x2=x2, vmin=vmin, vmax=vmax)
    ax.set_title(title + f", t = {t:.2f}")
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=fps)
    PATH = checkpoint_path.split("checkpoints/")[0]
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    cm = writer.saving(fig, f"{PATH}/movie.mp4", 800)

    with cm:
        for i in range(len(file_list)):
            file_path = file_list[i]
            with h5py.File(file_path, 'r') as f:
                t = f.attrs["t"]
                rho, momx1, momx2, e = np.array(f["rho"]), np.array(f["momx1"]), np.array(f["momx2"]), np.array(f["E"])
                if var == "density":
                    matrix = rho
                elif var == "log density":
                    matrix = np.log10(rho)
                elif var == "u":
                    matrix = momx1 / rho
                elif var == "v":
                    matrix = momx2 / rho
                elif var == "energy":
                    matrix = e
                if coords == "polar":
                    c.set_array(matrix.ravel())
                elif coords == "cartesian":
                    c.set_data(np.transpose(matrix))
                cb.update_normal(c)
                ax.set_title(title + f", t = {t:.2f}")
                fig.canvas.draw()
                writer.grab_frame()

                print_progress_bar(i, len(file_list),
                                   suffix="complete", length=25)
