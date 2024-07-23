import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


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
        raise ValueError(
            "Timeseries and values arrays must have the same length")

    # Precompute constants
    factor = 1 / (sigma * jnp.sqrt(2 * jnp.pi))
    sigma_sq = sigma ** 2
    window_radius = truncate * sigma

    smoothed_values = jnp.zeros_like(values)

    for i, t in enumerate(timeseries):
        print(i / len(timeseries))
        # Select points within the window_radius
        mask = jnp.abs(timeseries - t) <= window_radius
        times_window = timeseries[mask]
        values_window = values[mask]

        # Vectorized calculation of Gaussian weights
        weights = factor * jnp.exp(-0.5 * ((t - times_window) ** 2) / sigma_sq)
        smoothed_values[i] = jnp.sum(
            weights * values_window) / jnp.sum(weights)

    return smoothed_values


PATH = "/Volumes/T7/research/out.h5"
var = "torque"

with h5py.File(PATH, "r") as f:
    gamma = f.attrs["gamma"]
    x1, x2 = f.attrs["x1"], f.attrs["x2"]
    if "x3" in f.attrs:
        x3 = f.attrs["x3"]

    tc = f["tc"][...] / (2 * jnp.pi)  # simulation times
    if var == "torque":
        data = jnp.array(f["torque_1"][...]) + jnp.array(f["torque_2"][...])
    elif var == "ecc_mag":
        ecc_x, ecc_y = f["eccentricity_x"][...], f["eccentricity_y"][...]
        data = jnp.sqrt(ecc_x ** 2 + ecc_y ** 2)
    elif var == "ecc_phase":
        ecc_x, ecc_y = f["eccentricity_x"][...], f["eccentricity_y"][...]
        data = jnp.arctan2(ecc_y, ecc_x)
    else:
        data = f[var][...]

    # smoothed_data = gaussian_smooth(t, data, sigma=2, truncate=4)

    # print(data, smoothed_data)

    if len(data.shape) == 1:
        plt.plot(tc, data, linewidth=1, c="black")
        # plt.yscale("log")
        # plt.ylim((-4, 4))

    plt.title("Excised Torque")
    plt.xlabel("Time (Orbits)")
    plt.ylabel(r"Torque [$\Sigma_0 Gma$]")
    plt.savefig(f"./{var}.png", bbox_inches="tight")
    plt.show()
