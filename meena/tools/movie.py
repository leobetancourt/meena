from src.common.helpers import plot_matrix, print_progress_bar
import os
import h5py
import re
import numpy as np
from scipy.special import iv

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.common.helpers import load_U

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


def generate_movie(checkpoint_path, t_min, t_max, var, grid_range, title, fps, vmin, vmax, dpi, bitrate, cmap, t_factor, t_units, normalize):
    rho_0 = 1
    if normalize:
        prims_0, _, _, _ = load_U(f"{checkpoint_path}/out_0.0000.h5")
        rho = prims_0[..., 0]
        rho_0 = np.max(rho)
    
    file_list = get_h5_files_in_range(checkpoint_path, t_min, t_max)
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma / \Sigma_0$",
              "u": r"$u$", "v": r"$v$", "pressure": r"$P$"}
    with h5py.File(file_list[0], 'r') as f:
        coords = f.attrs["coords"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        t = f.attrs["t"]
        rho, u, v, p = np.array(f["rho"]), np.array(
            f["u"]), np.array(f["v"]), np.array(f["p"])
        if var == "density":
            matrix = rho / rho_0
        elif var == "log density":
            matrix = np.log10(rho / rho_0)
        elif var == "u":
            matrix = u
        elif var == "v":
            matrix = v
        elif var == "pressure":
            matrix = p

        if grid_range:
            x1_min, x1_max = grid_range[0], grid_range[1]
            x2_min, x2_max = grid_range[2], grid_range[3]
            x1_min_i = np.searchsorted(x1, x1_min, side="left")
            x1_max_i = np.searchsorted(x1, x1_max, side="right") - 1
            x2_min_i = np.searchsorted(x2, x2_min, side="left")
            x2_max_i = np.searchsorted(x2, x2_max, side="right") - 1

            matrix = matrix[x1_min_i:x1_max_i+1, x2_min_i:x2_max_i+1]
            x1, x2 = x1[(x1 >= x1_min) & (x1 <= x1_max)], x2[(x2 >= x2_min) & (x2 <= x2_max)]

    fig, ax, c, cb = plot_matrix(
        matrix, labels[var], coords, x1, x2, vmin, vmax, cmap)
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
                rho, u, v, p = np.array(f["rho"]), np.array(f["u"]), np.array(f["v"]), np.array(f["p"])
                if var == "density":
                    matrix = rho / rho_0
                elif var == "log density":
                    matrix = np.log10(rho / rho_0)
                elif var == "u":
                    matrix = u
                elif var == "v":
                    matrix = v
                elif var == "pressure":
                    matrix = p

                if grid_range:
                    matrix = matrix[x1_min_i:x1_max_i+1, x2_min_i:x2_max_i+1]
                
                if coords == "polar":
                    c.set_array(matrix.ravel())
                elif coords == "cartesian":
                    c.set_data(np.transpose(matrix))
                cb.update_normal(c)
                if title == "":
                    ax.set_title(f"t = {(t*t_factor):.2f} {t_units}")
                else:
                    ax.set_title(title + f", t = {(t*t_factor):.2f} {t_units}")
                fig.canvas.draw()
                plt.tight_layout()
                writer.grab_frame()

                print_progress_bar(i, len(file_list),
                                   suffix="complete", length=25)

# def analytic_sigma(x, tau, m, R_0):
#     x = np.maximum(x, 1e-10)
#     ln_prefactor = np.log(m) - np.log(np.pi) - 2 * np.log(R_0) - np.log(tau) - 0.25 * np.log(x)
#     ln_exponential = -(1 + x**2) / tau
    
#     bessel_arg = 2 * x / tau
#     large_bessel = bessel_arg > 100
#     ln_bessel_term = np.where(
#         large_bessel,
#         bessel_arg - 0.5 * np.log(2 * np.pi * bessel_arg),
#         np.log(iv(0.25, bessel_arg))
#     )
    
#     ln_sigma = ln_prefactor + ln_exponential + ln_bessel_term
#     return np.exp(ln_sigma)

# def generate_movie(checkpoint_path, t_min, t_max, var, grid_range, title, fps, vmin, vmax, dpi, bitrate, cmap, t_factor, t_units):
#     file_list = get_h5_files_in_range(checkpoint_path, t_min, t_max)
#     labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma / \Sigma_0$",
#               "u": r"$u$", "v": r"$v$", "pressure": r"$P$"}
#     fig, ax = plt.subplots()

#     FFMpegWriter = animation.writers['ffmpeg']
#     writer = FFMpegWriter(fps=fps, bitrate=bitrate)
#     PATH = checkpoint_path.split("checkpoints/")[0]
#     if not os.path.exists(PATH):
#         os.makedirs(PATH)
#     cm = writer.saving(fig, f"{PATH}/movie.mp4", dpi)

#     title = "Viscous Ring (Single BH)"
#     with cm:
#         for i in range(len(file_list)):
#             file_path = file_list[i]
#             with h5py.File(file_path, 'r') as f:
#                 R_0, nu, m = 1, 1e-3, 1
#                 t = f.attrs["t"]
#                 coords = f.attrs["coords"]
#                 rho = np.array(f["rho"])

#                 x1, x2 = f.attrs['x1'], f.attrs['x2']

#                 X1, X2 = np.meshgrid(x1, x2, indexing='ij')
#                 R = np.sqrt(X1**2 + X2**2)
#                 R_max = np.max(R)
#                 num_bins = 500  # Adjust the number of bins as needed
#                 R_bins = np.linspace(0, R_max, num_bins + 1)

#                 # Digitize the radial distances into bins
#                 r_indices = np.digitize(R, R_bins) - 1  # Bin indices, subtract 1 to make them 0-indexed

#                 # Compute average densities in each radial bin
#                 sigma = np.zeros(num_bins)
#                 for j in range(num_bins):
#                     mask = r_indices == j  # Select points in the current radial bin
#                     if np.any(mask):  # Avoid dividing by zero
#                         sigma[j] = np.mean(rho[mask])
                
#                 tau = 12 * nu * t * (R_0**-2)
#                 x = R_bins[:-1] / R_0
#                 density_norm = np.pi * sigma * (R_0 ** 2) / m
#                 analytic_density = np.pi * analytic_sigma(x, tau, m, R_0) * (R_0 ** 2) / m
#                 ax.plot(x, density_norm, color="red", label="Meena")
#                 ax.plot(x, analytic_density, color="black", alpha=0.5, label="Pringle Eq. (2.13)")
#                 ax.set_xlim(0, 3)
#                 ax.set_xlabel(r"$R/R_0$")
#                 ax.set_yscale("log")
#                 ax.set_ylim(1e-4, 1e1)
#                 sigma_label = r"$\pi \Sigma R_0^2/ m$"
#                 ax.set_ylabel(sigma_label)
#                 ax.legend()
                
#                 ax.set_title(rf"Viscous Ring (Single BH), $R_0=1a, \sigma=0.1a, \tau={tau:.3f}$")
#                 fig.canvas.draw()
#                 plt.tight_layout()
#                 writer.grab_frame()
#                 ax.cla()

#                 print_progress_bar(i, len(file_list),
#                                    suffix="complete", length=25)
