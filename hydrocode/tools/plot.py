from functools import partial
import os
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from common.helpers import print_progress_bar, read_csv
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
    t_min = 200 * 2 * np.pi
    t_max = 300 * 2 * np.pi
    m_dot = diagnostics["m_dot"][(t <= t_max) & (t > t_min)]
    dt = diagnostics["dt"][0]
    # Compute the periodogram
    frequencies, power = signal.periodogram(m_dot, 1 / (dt / (2 * np.pi)))
    # ft = np.fft.fft(m_dot)
    # power = np.abs(ft) ** 2
    # freq = np.fft.fftfreq(N, dt)
    # freq_orbits = freq * (2 * np.pi)
    
    x_min, x_max = 0.1, 2.5
    power = power / max(power[(frequencies >= x_min) & (frequencies <= x_max)])
    fig, ax = plt.subplots()
    plt.plot(frequencies, power, label=label)
    plt.xlabel(r'Variability Frequency (Orbits$^{-1}$)')
    plt.ylabel("Power")
    plt.title(r"$\dot{M}$ Periodogram (fixed timestep)")
    plt.ylim(0, 1)
    plt.xlim(0, x_max)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))


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
    
def smoothed_derivative(times, x, index, window_size):
    # Ensure the window size is positive and within bounds
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    
    # Define the start and end indices for the window
    start = max(index - window_size, 0)
    end = min(index + window_size + 1, len(times))

    # Slice the arrays for the window
    times_window = times[start:end]
    x_window = x[start:end]

    # Calculate the differences in eccentricity and time
    delta_x = np.diff(x_window)
    delta_time = np.diff(times_window)

    # Compute the local derivatives
    local_derivatives = delta_x / delta_time

    # Compute the smoothed derivative by averaging
    smoothed_derivative = np.mean(local_derivatives)

    return smoothed_derivative

    
def instability_growth_rate(diagnostics, dx):
    t = diagnostics["t"] / (2 * np.pi)
    e_x = diagnostics["e_x"]
    e_y = diagnostics["e_y"]
    e = e_x ** 2 + e_y ** 2
    idx = np.argmax(e > 3e-3)
    
    # e_dot = smoothed_derivative(t, e, idx, 1000)
    dist = 5
    e_dot = (e[idx + dist] - e[idx - dist]) / (t[idx + dist] - t[idx - dist])

    growth_rate = e_dot / e[idx]
    
    fig, ax = plt.subplots()
    print(growth_rate)
    plt.scatter(dx, growth_rate, s=3, c="black")
    plt.ylim((0, 0.02))
    plt.xscale("log")
    ax.set_xticks([0.01, 0.1])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim((0.005, 1))
    plt.title("Instability Growth Rate")
    plt.xlabel(r"Resolution ($dx/a$)")
    plt.ylabel(r"$\Gamma_c / \Omega_B$")

def cavity_precession_rate(diagnostics, label=""):
    t = diagnostics["t"] / (2 * np.pi)
    e_x = diagnostics["e_x"]
    e_y = diagnostics["e_y"]
    e_x_dot = np.gradient(e_x, t)
    e_y_dot = np.gradient(e_y, t)
    
    precession_rate = (e_y_dot * e_x - e_x_dot * e_y) / (e_x ** 2 + e_y ** 2)
    
    omega_B = 1
    

# diagnostics_high = read_csv("./500x3000/diagnostics.csv")
# diagnostics_mid = read_csv("./300x1800/diagnostics.csv")
# diagnostics_low = read_csv("./100x600/diagnostics.csv")

# m_dot_lombscargle(diagnostics_high, label=r"$500 (r) x 3000 (\theta)$")
# m_dot_lombscargle(diagnostics_mid, label=r"$300 (r) x 1800 (\theta)$")
# m_dot_lombscargle(diagnostics_low, label=r"$100 (r) x 600 (\theta)$")

# torque(diagnostics_high, label=r"$500 (r) x 3000 (\theta)$")
# torque(diagnostics_mid, label=r"$300 (r) x 1800 (\theta)$")
# torque(diagnostics_low, label=r"$100 (r) x 600 (\theta)$")

# instability_growth_rate(diagnostics_low, dx=(30/100))
# instability_growth_rate(diagnostics_mid, dx=(30/300))
# instability_growth_rate(diagnostics_high, dx=(30/500))

diagnostics = read_csv("./500x3000_fixed/diagnostics.csv")
m_dot_periodogram(diagnostics)
# stats = read_csv("./RT/stats.csv")
# print(stats["n_zones"])
# plt.scatter(stats["n_zones"], stats["M_zones_per_sec"], s=1, c="black")
# plt.title("Resolution vs. Speed (V100)")
# plt.xlabel("number of zones")
# plt.ylabel("million zone updates/second")

plt.savefig(f"./visual/fig.png", bbox_inches="tight")
plt.show()
