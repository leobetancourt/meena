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


def generate_movie(checkpoint_path, t_min, t_max, var, grid_range, title, fps, vmin, vmax, dpi, bitrate, cmap, t_factor, t_units):
    file_list = get_h5_files_in_range(checkpoint_path, t_min, t_max)
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$",
              "u": r"$u$", "v": r"$v$", "energy": r"$E$"}
    with h5py.File(file_list[0], 'r') as f:
        coords = f.attrs["coords"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        t = f.attrs["t"]
        rho, momx1, momx2, e = np.array(f["rho"]), np.array(
            f["momx1"]), np.array(f["momx2"]), np.array(f["E"])
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

        if grid_range:
            x1_min, x1_max = grid_range[0], grid_range[1]
            x2_min, x2_max = grid_range[2], grid_range[3]
            x1_min_i = np.searchsorted(x1, x1_min, side="left")
            x1_max_i = np.searchsorted(x1, x1_max, side="right") - 1
            x2_min_i = np.searchsorted(x2, x2_min, side="left")
            x2_max_i = np.searchsorted(x2, x2_max, side="right") - 1

            matrix = matrix[x1_min_i:x1_max_i+1, x2_min_i:x2_max_i+1]
            x1, x2 = x1[(x1 >= x1_min) & (x1 <= x1_max)], x2[(x2 >= x2_min) & (x2 <= x2_max)]

    fig, ax, c, cb = plot_grid(
        matrix, labels[var], coords, x1, x2, vmin, vmax, cmap)
    ax.set_title(title + f", t = {(t*t_factor):.2f} {t_units}")
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    PATH = checkpoint_path.split("checkpoints/")[0]
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    cm = writer.saving(fig, f"{PATH}/movie.mp4", dpi)

    with cm:
        for i in range(len(file_list)):
            file_path = file_list[i]
            with h5py.File(file_path, 'r') as f:
                t = f.attrs["t"]
                rho, momx1, momx2, e = np.array(f["rho"]), np.array(
                    f["momx1"]), np.array(f["momx2"]), np.array(f["E"])
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

                if grid_range:
                    matrix = matrix[x1_min_i:x1_max_i+1, x2_min_i:x2_max_i+1]
                
                if coords == "polar":
                    c.set_array(matrix.ravel())
                elif coords == "cartesian":
                    c.set_data(np.transpose(matrix))
                cb.update_normal(c)
                ax.set_title(title + f", t = {(t*t_factor):.2f} {t_units}")
                fig.canvas.draw()
                writer.grab_frame()

                print_progress_bar(i, len(file_list),
                                   suffix="complete", length=25)
