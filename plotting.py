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
                    choices=["accretion_rate", "accretion_rate_1", "accretion_rate_2", "torque", "torque_1", "torque_2", "ecc_mag", "ecc_phase", "angular_mom_rate"],
                    help='The variable to plot: accretion_rate, accretion_rate_1, accretion_rate_2, torque, torque_1, torque_2, ecc_mag, ecc_phase, angular_mom_rate (required)')

args = parser.parse_args()

# validate the file argument
if not args.file.endswith('.hdf'):
    parser.error("The file name must end with '.hdf'")

PATH = args.file
var = args.output

def gaussian_smooth(timeseries, values, sigma, truncate=3.0):
    """
    Smooths the input function values using a Gaussian kernel with truncated integral.

    Parameters:
    timeseries (np.ndarray): Array of time points.
    values (np.ndarray): Array of function values at each time point.
    sigma (float): Standard deviation of the Gaussian kernel.
    truncate (float): Truncate the Gaussian kernel at this many standard deviations.

    Returns:
    np.ndarray: Smoothed function values.
    """
    if len(timeseries) != len(values):
        raise ValueError("Timeseries and values arrays must have the same length")

    # Precompute constants
    factor = 1 / (sigma * np.sqrt(2 * np.pi))
    sigma_sq = sigma ** 2
    window_radius = truncate * sigma

    smoothed_values = np.zeros_like(values)

    for i, t in enumerate(timeseries):
        print(i / len(timeseries))
        # Select points within the window_radius
        mask = np.abs(timeseries - t) <= window_radius
        times_window = timeseries[mask]
        values_window = values[mask]

        # Vectorized calculation of Gaussian weights
        weights = factor * np.exp(-0.5 * ((t - times_window) ** 2) / sigma_sq)
        smoothed_values[i] = np.sum(weights * values_window) / np.sum(weights)

    return smoothed_values

with h5py.File(PATH, "r") as f:
    gamma = f.attrs["gamma"]
    x1, x2 = f.attrs["x1"], f.attrs["x2"]
    if "x3" in f.attrs:
        x3 = f.attrs["x3"]
    
    t = f["t"][...] / (2 * np.pi) # simulation times
    if var == "torque":
        data = np.array(f["torque_1"][...]) + np.array(f["torque_2"][...])
    elif var == "ecc_mag":
        ecc_x, ecc_y = f["eccentricity_x"][...], f["eccentricity_y"][...]
        data = np.sqrt(ecc_x ** 2 + ecc_y ** 2)
    elif var == "ecc_phase":
        ecc_x, ecc_y = f["eccentricity_x"][...], f["eccentricity_y"][...]
        data = np.arctan2(ecc_y, ecc_x)
    else:
        data = f[var][...]
        
    dA = 0.0001088292186894373
    data *= dA
    smoothed_data = gaussian_smooth(t, data, sigma=2, truncate=4)
        
    print(data, smoothed_data)
    
    if len(data.shape) == 1:
        plt.plot(t, smoothed_data, linewidth=1, c="black")
        # plt.yscale("log")
        # plt.ylim((-4, 4))
        
    plt.title("Excised Torque")
    plt.xlabel("Time (Orbits)")
    plt.ylabel(r"Torque [$\Sigma_0 Gma$]")
    plt.savefig(f"./visual/{var}.png", bbox_inches="tight")
    plt.show()
    
