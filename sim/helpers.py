import numpy as np


def E(gamma, rho, p, u, v):
    return (p / (gamma - 1)) + (0.5 * rho * (u ** 2 + v ** 2))


def P(gamma, rho, u, v, E):
    return (gamma - 1) * (E - (0.5 * rho * (u ** 2 + v ** 2)))


def enthalpy(rho, p, E):
    return (E + p) / rho


def cartesian_to_polar(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return (r, theta)


def polar_to_cartesian(r, theta):
    return (r * np.cos(theta), r * np.sin(theta))


def c_s(gamma, P, rho):
    return np.sqrt(gamma * P / rho)
