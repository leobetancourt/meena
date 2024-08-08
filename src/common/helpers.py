from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
import h5py
import csv


def linspace_cells(min, max, num):
    interfaces = jnp.linspace(min, max, num + 1)
    centers = (interfaces[:-1] + interfaces[1:]) / 2

    return centers, interfaces

def logspace_cells(min, max, num):
    interfaces = jnp.logspace(jnp.log10(min), jnp.log10(max), num + 1)
    centers = (interfaces[:-1] + interfaces[1:]) / 2

    return centers, interfaces

def cartesian_to_polar(x, y):
    r = jnp.sqrt(x ** 2 + y ** 2)
    theta = jnp.arctan2(y, x)
    return r, theta

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # print new line on complete
    if iteration == total:
        print()

def read_csv(path, compress=False):
    # Read the CSV file using pandas
    df = pd.read_csv(path)

    # Initialize a dictionary to store the numpy arrays
    numpy_arrays = {}

    # Iterate over each column in the dataframe and save it as a numpy array
    for column in df.columns:
        if compress:
            numpy_arrays[column] = df[column].values[::2]
        else:
            numpy_arrays[column] = df[column].values

    return numpy_arrays

def save_to_h5(filename, t, U, coords, gamma, x1, x2):
    rho, momx1, momx2, E = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    with h5py.File(filename, "w") as f:
        # metadata
        f.attrs["coords"] = coords
        f.attrs["gamma"] = gamma
        f.attrs["x1"] = x1
        f.attrs["x2"] = x2
        f.attrs["t"] = t

        # create h5 datasets for conserved variables
        f.create_dataset("rho", data=rho, dtype="float64")
        f.create_dataset("momx1", data=momx1, dtype="float64")
        f.create_dataset("momx2", data=momx2, dtype="float64")
        f.create_dataset("E", data=E, dtype="float64")


def create_csv_file(filename, headers):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def append_row_csv(filename, row):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def load_U(file):
    with h5py.File(file, 'r') as f:
        t = f.attrs["t"]
        rho, momx1, momx2, e = f["rho"], f["momx1"], f["momx2"], f["E"]

        U = jnp.array([
            rho,
            momx1,
            momx2,
            e
        ]).transpose((1, 2, 0))

        return U, t


def plot_grid(matrix, label, coords, x1, x2, vmin=None, vmax=None):
    extent = [x1[0], x1[-1], x2[0], x2[-1]]

    if coords == "cartesian":
        fig, ax = plt.subplots()
        if vmin is None:
            vmin, vmax = jnp.min(matrix), jnp.max(matrix)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        c = ax.imshow(jnp.transpose(matrix), cmap="magma", interpolation='nearest',
                      origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    elif coords == "polar":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        if vmin is None:
            vmin, vmax = jnp.min(matrix), jnp.max(matrix)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_ylim(0, jnp.max(x1))
        ax.set_ylim(0, 5)
        ax.set_facecolor("black")
        circle_r = jnp.min(x1) - (x1[1] - x1[0]) / 2
        circle = Circle((0, 0), radius=circle_r, transform=ax.transData._b,
                        color='blue', fill=False, linewidth=1)
        ax.add_patch(circle)
        R, Theta = jnp.meshgrid(x1, x2, indexing="ij")
        c = ax.pcolormesh(Theta, R, matrix, shading='auto',
                          cmap="magma", vmin=vmin, vmax=vmax)

    cb = plt.colorbar(c, ax=ax, label=label)

    return fig, ax, c, cb


def cartesian_to_polar(x, y):
    r = jnp.sqrt(x ** 2 + y ** 2)
    theta = jnp.arctan2(y, x)
    return r, theta


def add_ghost_cells(arr, num_g, axis=0):
    if axis == 0:
        # add ghost cells to the first coordinate direction
        return jnp.vstack((jnp.repeat(arr[:1, :, :], num_g, axis), arr, jnp.repeat(
            arr[-1:, :, :], num_g, axis)))
    elif axis == 1:
        # add ghost cells to the second coordinate direction
        return jnp.hstack((jnp.repeat(arr[:, :1, :], num_g, axis), arr, jnp.repeat(
            arr[:, -1:, :], num_g, axis)))


def apply_bcs(lattice, U):
    g = lattice.num_g
    bc_x1, bc_x2 = lattice.bc_x1, lattice.bc_x2
    if bc_x1[0] == "outflow":
        U = U.at[:g, :, :].set(U[g:(g+1), :, :])
    elif bc_x1[0] == "reflective":
        U = U.at[:g, :, :].set(jnp.flip(U[g:(2*g), :, :], axis=0))
        # invert x1 momentum
        U = U.at[:g, :, 1].set(-jnp.flip(U[g:(2*g), : 1], axis=0))
    elif bc_x1[0] == "periodic":
        U = U.at[:g, :, :].set(U[(-2*g):(-g), :, :])

    if bc_x1[1] == "outflow":
        U = U.at[-g:, :, :].set(U[-(g+1):-g, :, :])
    elif bc_x1[1] == "reflective":
        U = U.at[-g:, :, :].set(jnp.flip(U[-(2*g):-g, :, :], axis=0))
        # invert x1 momentum
        U = U.at[-g:, :, 1].set(-jnp.flip(U[-(2*g):-g, : 1], axis=0))
    elif bc_x1[1] == "periodic":
        U = U.at[-g:, :, :].set(U[g:(2*g), :, :])

    if bc_x2[0] == "outflow":
        U = U.at[:, :g, :].set(U[:, g:(g+1), :])
    elif bc_x2[0] == "reflective":
        U = U.at[:, :g, :].set(jnp.flip(U[:, g:(2*g), :], axis=1))
        # invert x2 momentum
        U = U.at[:, :g, 2].set(-jnp.flip(U[:, g:(2*g), 2], axis=1))
    elif bc_x2[0] == "periodic":
        U = U.at[:, :g, :].set(U[:, (-2*g):(-g), :])

    if bc_x2[1] == "outflow":
        U = U.at[:, -g:, :].set(U[:, -(g+1):-g, :])
    elif bc_x2[1] == "reflective":
        U = U.at[:, -g:, :].set(jnp.flip(U[:, -(2*g):-g, :], axis=1))
        # invert x2 momentum
        U = U.at[:, -g:, 2].set(-jnp.flip(U[:, -(2*g):-g, 2], axis=1))
    elif bc_x2[1] == "periodic":
        U = U.at[:, -g:, :].set(U[:, g:(2*g), :])

    return U


def get_prims(hydro, U, X1, X2, t):
    rho = U[:, :, 0]
    u, v = U[:, :, 1] / rho, U[:, :, 2] / rho
    e = U[:, :, 3]
    p = hydro.P((rho, u, v, e), X1, X2, t)
    return rho, u, v, p

def F_from_prim(hydro, prims, X1, X2, t):
    rho, u, v, p = prims
    e = hydro.E(prims, X1, X2, t)
    return jnp.array([
        rho * u,
        rho * (u ** 2) + p,
        rho * u * v,
        (e + p) * u
    ]).transpose((1, 2, 0))

def G_from_prim(hydro, prims, X1, X2, t):
    rho, u, v, p = prims
    e = hydro.E(prims, X1, X2, t)
    return jnp.array([
        rho * v,
        rho * u * v,
        rho * (v ** 2) + p,
        (e + p) * v
    ]).transpose((1, 2, 0))
