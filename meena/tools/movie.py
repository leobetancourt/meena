from src.common.helpers import plot_matrix, print_progress_bar
import os
import h5py
import re
import numpy as np
from scipy.special import iv

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
    
    # Preload all data
    rho_list, u_list, p_list, p_rad_list = [], [], [], []
    times = []
    radiation = False

    for file in file_list:
        with h5py.File(file, 'r') as f:
            times.append(f.attrs["t"])
            rho = np.array(f["rho"])
            u = np.array(f["u"])
            p = np.array(f["p"])
            rho_list.append(rho)
            u_list.append(u)
            p_list.append(p)

            if "p_rad" in f:
                p_rad = np.array(f["p_rad"])
                p_rad_list.append(p_rad)
                radiation = True

            # Read grid only once (assuming constant across files)
            if len(rho_list) == 1:
                coords = f.attrs["coords"]
                x1 = f.attrs["x1"]
                x2 = f.attrs.get("x2", None)

    # Apply grid range once (assuming same grid across time)
    if grid_range:
        x1_min, x1_max = grid_range[0], grid_range[1]
        x2_min, x2_max = grid_range[2], grid_range[3]
        x1_min_i = np.searchsorted(x1, x1_min, side="left")
        x1_max_i = np.searchsorted(x1, x1_max, side="right") - 1
        x2_min_i = np.searchsorted(x2, x2_min, side="left")
        x2_max_i = np.searchsorted(x2, x2_max, side="right") - 1
        x1 = x1[x1_min_i:x1_max_i+1]
        x2 = x2[x2_min_i:x2_max_i+1] if x2 is not None else None
    else:
        x1_min_i = x2_min_i = 0
        x1_max_i = len(x1) - 1
        x2_max_i = len(x2) - 1 if x2 is not None else 0

    # Clip all arrays to grid range
    def clip(arr):
        return arr[x1_min_i:x1_max_i+1, x2_min_i:x2_max_i+1] if x2 is not None else arr[x1_min_i:x1_max_i+1]

    rho_list = [clip(r) for r in rho_list]
    u_list = [clip(u) for u in u_list]
    p_list = [clip(p) for p in p_list]
    if radiation:
        p_rad_list = [clip(pr) for pr in p_rad_list]

    # Compute dynamic ylims
    def get_min_max(lst):
        data = np.stack(lst)
        return np.min(data), np.max(data)

    rho_min, rho_max = get_min_max(rho_list)
    u_min, u_max = get_min_max(u_list)
    p_min, p_max = get_min_max(p_list)
    if radiation:
        pr_min, pr_max = get_min_max(p_rad_list)

    ylims = [(rho_min, rho_max), (u_min, u_max), (p_min, p_max)]
    if radiation:
        ylims.append((pr_min, pr_max))

    # Set labels
    if radiation:
        matrices = [rho_list[0], u_list[0], p_list[0], p_rad_list[0]]
        labels = [r"$\rho$", r"$u$", r"$p$", r"$p_{\mathrm{rad}}$"]
    else:
        matrices = [rho_list[0], u_list[0], p_list[0]]
        labels = [r"$\rho$", r"$u$", r"$p$"]

    # Plotting
    def plot_grid_1d(matrices, labels, x1, x2=None):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        for i, (matrix, label) in enumerate(zip(matrices, labels)):
            ax = axs[i]
            if x2 is None:
                ax.plot(x1, matrix)
            else:
                im = ax.imshow(matrix, extent=[x1.min(), x1.max(), x2.min(), x2.max()],
                               origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(label)
            ax.set_ylim(ylims[i][0], ylims[i][1])
        return fig, axs

    fig, axs = plot_grid_1d(matrices, labels, x1, x2)
    fig.suptitle(f"t = {times[0] * t_factor:.2f} {t_units}", fontsize=16)

    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    PATH = checkpoint_path.split("checkpoints/")[0]
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    cm = writer.saving(fig, f"{PATH}/movie.mp4", dpi)

    with cm:
        for i in range(len(file_list)):
            matrices = [rho_list[i], u_list[i], p_list[i]]
            if radiation:
                matrices.append(p_rad_list[i])

            for j, (matrix, label) in enumerate(zip(matrices, labels)):
                axs[j].cla()
                if x2 is None:
                    axs[j].plot(x1, matrix)
                else:
                    im = axs[j].imshow(matrix, extent=[x1.min(), x1.max(), x2.min(), x2.max()],
                                       origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                    fig.colorbar(im, ax=axs[j])
                axs[j].set_xlabel(r"$x$")
                axs[j].set_ylabel(label)
                axs[j].set_ylim(ylims[j][0], ylims[j][1])

            fig.suptitle((title + ", " if title else "") + f"t = {times[i] * t_factor:.2f} {t_units}", fontsize=16)
            fig.canvas.draw()
            writer.grab_frame()
            print_progress_bar(i, len(file_list), suffix="complete", length=25)

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
