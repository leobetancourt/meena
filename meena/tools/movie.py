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
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma / \Sigma_0$", "u": r"$u$", "v": r"$v$", "energy": r"$E$", "pressure": r"$P$", "div B": r"$\nabla \cdot B$"}
    
    with h5py.File(file_list[0], 'r') as f:
        t = f.attrs["t"]
        coords = f.attrs["coords"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        x3 = f.attrs["x3"]
        dx = x1[1] - x1[0]
        if len(x2) > 1:
            dy = x2[1] - x2[0]
        gamma = f.attrs["gamma"]
        rho = np.array(f["rho"])
        u, v, w, e = np.array(f["momx1"]) / rho, np.array(f["momx2"]) / rho, np.array(f["momx3"]) / rho, np.array(f["E"])
        Bx, By, Bz = np.array(f["Bx"]), np.array(f["By"]), np.array(f["Bz"])
            
        if var == "density":
            matrix = rho
        elif var == "log density":
            matrix = np.log10(rho)
        elif var == "u":
            matrix = u
        elif var == "v":
            matrix = v
        elif var == "w":
            matrix = w
        elif var == "energy":
            matrix = e
        elif var == "pressure":
            u2 = u**2 + v**2 + w**2
            B2 = Bx**2 + By**2 + Bz**2
            matrix = (gamma - 1) * (e - 0.5 * (rho * u2 + B2))
        elif var == "bx":
            matrix = Bx
        elif var == "by":
            matrix = By
        elif var == "bz":
            matrix = Bz
        elif var == "div B":
            matrix = (Bx[2:, 1:-1] - Bx[:-2, 1:-1]) / (2 * dx) + (By[1:-1, 2:] - By[1:-1, :-2]) / (2 * dy)
            x1, x2 = x1[1:-1], x2[1:-1]
        
    if len(x2) == 1: # this is a 1D simulation
        fig, ax = plt.subplots()
        c = None
        ax.plot(x1, matrix[:, 0, 0])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(labels[var])
    else:
        fig, ax, c, cb = plot_grid(matrix[..., matrix.shape[-1] // 2], labels[var], coords, x1, x2, vmin, vmax, cmap)
    if title == "":
        ax.set_title(f"t = {(t*t_factor):.2f} {t_units}")
    else:
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
                coords = f.attrs["coords"]
                x1 = f.attrs["x1"]
                x2 = f.attrs["x2"]
                x3 = f.attrs["x3"]
                dx = x1[1] - x1[0]
                if len(x2) > 1:
                    dy = x2[1] - x2[0]
                gamma = f.attrs["gamma"]
                rho = np.array(f["rho"])
                u, v, w, e = np.array(f["momx1"]) / rho, np.array(f["momx2"]) / rho, np.array(f["momx3"]) / rho, np.array(f["E"])
                Bx, By, Bz = np.array(f["Bx"]), np.array(f["By"]), np.array(f["Bz"])
                    
                if var == "density":
                    matrix = rho
                elif var == "log density":
                    matrix = np.log10(rho)
                elif var == "u":
                    matrix = u
                elif var == "v":
                    matrix = v
                elif var == "w":
                    matrix = w
                elif var == "energy":
                    matrix = e
                elif var == "pressure":
                    u2 = u**2 + v**2 + w**2
                    B2 = Bx**2 + By**2 + Bz**2
                    matrix = (gamma - 1) * (e - 0.5 * (rho * u2 + B2))
                elif var == "bx":
                    matrix = Bx
                elif var == "by":
                    matrix = By
                elif var == "bz":
                    matrix = Bz
                elif var == "div B":
                    matrix = (Bx[2:, 1:-1] - Bx[:-2, 1:-1]) / (2 * dx) + (By[1:-1, 2:] - By[1:-1, :-2]) / (2 * dy)
                    x1, x2 = x1[1:-1], x2[1:-1]

                if c:
                    if coords == "polar":
                        c.set_array(matrix.ravel())
                    elif coords == "cartesian":
                        c.set_data(np.transpose(matrix[..., matrix.shape[-1] // 2]))
                    cb.update_normal(c)
                else:
                    ax.cla()
                    ax.plot(x1, matrix[:, 0, 0])
                    ax.set_xlabel(r"$x$")
                    ax.set_ylabel(labels[var])
                if title == "":
                    ax.set_title(f"t = {(t*t_factor):.2f} {t_units}")
                else:
                    ax.set_title(title + f", t = {(t*t_factor):.2f} {t_units}")
                fig.canvas.draw()
                writer.grab_frame()

                print_progress_bar(i, len(file_list),
                                   suffix="complete", length=25)
