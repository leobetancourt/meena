from functools import partial
import os
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helpers import print_progress_bar, read_csv
from jax import vmap, jit
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def gaussian_smooth(timeseries, values, output_times, sigma=2, truncate=3.0):
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
    factor = 1 / (sigma * np.sqrt(2 * np.pi))
    sigma_sq = sigma ** 2
    window_radius = truncate * sigma

    smoothed_values = np.zeros_like(output_times)

    for i, t in enumerate(output_times):
        # Select points within the window_radius
        distances = np.abs(timeseries - t)
        within_window = distances <= window_radius

        values_window = values[within_window]

        # Vectorized calculation of Gaussian weights
        weights = factor * \
            np.exp(-0.5 * (distances[within_window] ** 2) / sigma_sq)
        smoothed_values[i] = np.sum(weights * values_window) / np.sum(weights)

        print_progress_bar(i, len(output_times), length=25, suffix="complete")

    return smoothed_values


def torque(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    output_t = np.linspace(t[0], t[-1], 500)
    torque = gaussian_smooth(t, diagnostics["torque"], output_t, sigma=2, truncate=3.0)
    plt.plot(output_t, torque, linewidth=1, label=label)
    plt.axhline(linewidth=1, color="black")
    plt.title("Excised Torque")
    plt.xlabel(r"Time (Orbits)")
    plt.ylabel(r"Torque [$\Sigma_0 Gma]")
    plt.ylim(-0.015, 0.015)
    plt.legend()


def m_dot(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    output_t = np.linspace(t[0], t[-1], 500)
    m_dot = gaussian_smooth(t, diagnostics["m_dot"], output_t, sigma=2, truncate=3.0)
    plt.plot(output_t, m_dot, linewidth=1, label=label)
    plt.axhline(linewidth=1, color="black")
    plt.title(r"$\dot{M}$")
    plt.xlabel(r"Time (Orbits)")
    plt.legend()


def m_dot_lombscargle(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    # frequencies (orbits^-1) for which to compute the periodogram
    w = np.linspace(0.1, 5, 1000)
    vals = diagnostics["m_dot"]
    pgram = signal.lombscargle(t, vals, w)
    pgram = pgram / max(pgram)

    plt.plot(w, pgram, linewidth=1, label=label)
    plt.title(r"$\dot{M}$ Periodogram")
    plt.xlabel(r"Variability Frequency (Orbits$^{-1}$)")
    plt.ylabel("Power")
    plt.legend()


def m_dot_periodogram(diagnostics, label=""):
    t = diagnostics["t"]
    t_min = 0 * 2 * np.pi
    t_max = 300 * 2 * np.pi
    m_dot = diagnostics["m_dot"][(t <= t_max) & (t > t_min)]
    dt = diagnostics["dt"][0]
    N = len(m_dot)
    # Compute the periodogram
    frequencies, power = signal.periodogram(m_dot, 1 / (dt / (2 * np.pi)))
    # ft = np.fft.fft(m_dot)
    # power = np.abs(ft) ** 2
    # freq = np.fft.fftfreq(N, dt)
    # freq_orbits = freq * (2 * np.pi)
    
    x_min, xmax = 0.1, 2.5
    power = power / max(power[(frequencies >= x_min) & (frequencies <= xmax)])
    plt.plot(frequencies, power, label=label)
    plt.xlabel(r'Variability Frequency (Orbits$^{-1}$)')
    plt.ylabel("Power")
    plt.title(r"$\dot{M}$ Periodogram (fixed timestep)")
    plt.ylim(0, 1)
    plt.xlim(0, xmax)  # Optionally, limit the x-axis


def a_dot(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    output_t = np.linspace(t[0], t[-1], 500)
    torque = gaussian_smooth(t, diagnostics["torque"], output_t, sigma=2, truncate=4)
    mu = (0.5 * 0.5)
    a_dot_torque = 2 * torque / mu

    L_dot = gaussian_smooth(t, diagnostics["L_dot"], output_t, sigma=2, truncate=4)
    a_dot_L = 2 * L_dot / mu

    # plt.plot(output_t, a_dot_L, linewidth=1, c="blue", label=r"due to $\dot{L}$")
    # plt.plot(output_t, a_dot_torque, linewidth=1, c="red",
    #          label=r"due to $\tau_{grav}$")
    plt.plot(output_t, a_dot_torque + a_dot_L, linewidth=1, label=label)
    plt.axhline(linewidth=1, color="black")
    plt.title(r"$\dot{a},\ 500(r)x3000(\theta)$")
    plt.xlabel(r"Time (Orbits)")
    plt.legend()


def eccentricity_mag(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    mag = np.sqrt(diagnostics["e_x"] ** 2 + diagnostics["e_y"] ** 2)
    plt.plot(t, mag, linewidth=1, label=label)
    plt.yscale("log")
    plt.axhline(linewidth=1, color="black")
    plt.title("Eccentricity Magnitude")
    plt.xlabel(r"Time (Orbits)")
    plt.legend()


def eccentricity_phase(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    phase = np.arctan2(diagnostics["e_y"], diagnostics["e_x"])
    plt.plot(t, phase, linewidth=1, label=label)
    plt.axhline(linewidth=1, color="black")
    plt.ylim((-4, 4))
    plt.title("Eccentriticy Phase (Radians)")
    plt.xlabel(r"Time (Orbits)")
    plt.legend()
    
def growth_rate(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    e_x = diagnostics["e_x"]
    e_y = diagnostics["e_y"]
    e_x_dot = np.gradient(e_x, varargs=[t])
    e_y_dot = np.gradient(e_y, varargs=[t])
    
    growth_rate = (e_x_dot * e_x + e_y_dot * e_y) / (e_x ** 2 + e_y ** 2)
    precession_rate = (e_y_dot * e_x - e_x_dot * e_y) / (e_x ** 2 + e_y ** 2)
    
    omega_B = 1

# diagnostics_high = read_csv("./500x3000/diagnostics.csv", compress=True)
# diagnostics_mid = read_csv("./300x1800/diagnostics.csv", compress=True)
# diagnostics_low = read_csv("./100x600/diagnostics.csv", compress=True)

# m_dot_lombscargle(diagnostics_high, label=r"$500 (r) x 3000 (\theta)$")
# m_dot_lombscargle(diagnostics_mid, label=r"$300 (r) x 1800 (\theta)$")
# m_dot_lombscargle(diagnostics_low, label=r"$100 (r) x 600 (\theta)$")

# torque(diagnostics_high, label=r"$500 (r) x 3000 (\theta)$")
# torque(diagnostics_mid, label=r"$300 (r) x 1800 (\theta)$")
# torque(diagnostics_low, label=r"$100 (r) x 600 (\theta)$")

diagnostics = read_csv("./500x3000_fixed/diagnostics.csv")
m_dot_periodogram(diagnostics)


plt.savefig(f"./visual/fig.png", bbox_inches="tight")
plt.show()
