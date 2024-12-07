from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import pandas as pd
import h5py
import csv
import os
import glob
import re


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

def cartesian_to_spherical(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arctan2(y, x)  # Azimuthal angle
    phi = jnp.arccos(jnp.clip(z / r, -1.0, 1.0)) if r != 0 else 0.0  # Zenith angle
    return r, theta, phi

def add_ghost_coords(x, num_g):
    x_left = x[0] - (x[1] - x[0]) * jnp.arange(num_g, 0, -1)
    x_right = x[-1] + (x[-1] - x[-2]) * jnp.arange(1, num_g + 1)
    return jnp.concatenate([x_left, x, x_right])

def add_ghost_cells(arr, num_g, axis=0):
    if axis == 0:
        return jnp.vstack((
            jnp.repeat(arr[:1], num_g, axis), 
            arr, 
            jnp.repeat(arr[-1:, :, :], num_g, axis)))
    elif axis == 1:
        return jnp.hstack((
            jnp.repeat(arr[:, :1], num_g, axis), 
            arr, 
            jnp.repeat(arr[:, -1:, :], num_g, axis)))
    elif axis == 2:
        return jnp.dstack((
            jnp.repeat(arr[:, :, :1], num_g, axis), 
            arr, 
            jnp.repeat(arr[:, :, -1:], num_g, axis)))
    else:
        raise ValueError("Axis must be 0, 1, or 2.")


def apply_bcs(lattice, U):
    g = lattice.num_g
    bc_x1, bc_x2 = lattice.bc_x1, lattice.bc_x2
    
    if bc_x1[0] == "outflow":
        U = U.at[:g].set(U[g:(g+1)])
    elif bc_x1[0] == "reflective":
        U = U.at[:g].set(jnp.flip(U[g:(2*g)], axis=0))
        # invert x1 momentum
        U = U.at[:g, :, 1].set(-jnp.flip(U[g:(2*g), :, 1], axis=0))
    elif bc_x1[0] == "periodic":
        U = U.at[:g].set(U[(-2*g):(-g)])

    if bc_x1[1] == "outflow":
        U = U.at[-g:].set(U[-(g+1):-g])
    elif bc_x1[1] == "reflective":
        U = U.at[-g:].set(jnp.flip(U[-(2*g):-g], axis=0))
        # invert x1 momentum
        U = U.at[-g:, :, 1].set(-jnp.flip(U[-(2*g):-g, :, 1], axis=0))
    elif bc_x1[1] == "periodic":
        U = U.at[-g:].set(U[g:(2*g)])

    if bc_x2[0] == "outflow":
        U = U.at[:, :g].set(U[:, g:(g+1)])
    elif bc_x2[0] == "reflective":
        U = U.at[:, :g].set(jnp.flip(U[:, g:(2*g)], axis=1))
        # invert x2 momentum
        U = U.at[:, :g, 2].set(-jnp.flip(U[:, g:(2*g), 2], axis=1))
    elif bc_x2[0] == "periodic":
        U = U.at[:, :g].set(U[:, (-2*g):(-g)])

    if bc_x2[1] == "outflow":
        U = U.at[:, -g:].set(U[:, -(g+1):-g])
    elif bc_x2[1] == "reflective":
        U = U.at[:, -g:].set(jnp.flip(U[:, -(2*g):-g], axis=1))
        # invert x2 momentum
        U = U.at[:, -g:, 2].set(-jnp.flip(U[:, -(2*g):-g, 2], axis=1))
    elif bc_x2[1] == "periodic":
        U = U.at[:, -g:].set(U[:, g:(2*g)])

    return U

def apply_bcs_mhd(lattice, U):
    g = lattice.num_g
    bc_x1, bc_x2, bc_x3 = lattice.bc_x1, lattice.bc_x2, lattice.bc_x3
    
    if bc_x1[0] == "outflow":
        U = U.at[:g].set(U[g:(g+1)])
    elif bc_x1[0] == "reflective":
        U = U.at[:g].set(jnp.flip(U[g:(2*g)], axis=0))
        # invert x1 momentum
        U = U.at[:g, :, :, 1].set(-jnp.flip(U[g:(2*g), :, :, 1], axis=0))
    elif bc_x1[0] == "periodic":
        U = U.at[:g].set(U[(-2*g):(-g)])

    if bc_x1[1] == "outflow":
        U = U.at[-g:].set(U[-(g+1):-g])
    elif bc_x1[1] == "reflective":
        U = U.at[-g:].set(jnp.flip(U[-(2*g):-g], axis=0))
        # invert x1 momentum
        U = U.at[-g:, :, :, 1].set(-jnp.flip(U[-(2*g):-g, :, :, 1], axis=0))
    elif bc_x1[1] == "periodic":
        U = U.at[-g:].set(U[g:(2*g)])

    if bc_x2[0] == "outflow":
        U = U.at[:, :g].set(U[:, g:(g+1)])
    elif bc_x2[0] == "reflective":
        U = U.at[:, :g].set(jnp.flip(U[:, g:(2*g)], axis=1))
        # invert x2 momentum
        U = U.at[:, :g, :, 2].set(-jnp.flip(U[:, g:(2*g), :, 2], axis=1))
    elif bc_x2[0] == "periodic":
        U = U.at[:, :g].set(U[:, (-2*g):(-g)])

    if bc_x2[1] == "outflow":
        U = U.at[:, -g:].set(U[:, -(g+1):-g])
    elif bc_x2[1] == "reflective":
        U = U.at[:, -g:].set(jnp.flip(U[:, -(2*g):-g], axis=1))
        # invert x2 momentum
        U = U.at[:, -g:, :, 2].set(-jnp.flip(U[:, -(2*g):-g, :, 2], axis=1))
    elif bc_x2[1] == "periodic":
        U = U.at[:, -g:].set(U[:, g:(2*g)])
    
    if bc_x3[0] == "outflow":
        U = U.at[:, :, :g].set(U[:, :, g:(g+1)])
    elif bc_x3[0] == "reflective":
        U = U.at[:, :, :g].set(jnp.flip(U[:, :, g:(2*g)], axis=2))
        # invert x2 momentum
        U = U.at[:, :, :g, 3].set(-jnp.flip(U[:, :, g:(2*g), 3], axis=2))
    elif bc_x3[0] == "periodic":
        U = U.at[:, :, :g].set(U[:, :, (-2*g):(-g)])

    if bc_x3[1] == "outflow":
        U = U.at[:, :, -g:].set(U[:, :, -(g+1):-g])
    elif bc_x3[1] == "reflective":
        U = U.at[:, :, -g:].set(jnp.flip(U[:, :, -(2*g):-g], axis=2))
        # invert x2 momentum
        U = U.at[:, :, -g:, 3].set(-jnp.flip(U[:, :, -(2*g):-g, 3], axis=2))
    elif bc_x3[1] == "periodic":
        U = U.at[:, :, -g:].set(U[:, :, g:(2*g)])

    return U

def apply_bcs_B(lattice, g, U):
    bc_x1, bc_x2, bc_x3 = lattice.bc_x1, lattice.bc_x2, lattice.bc_x3
    
    if bc_x1[0] == "outflow":
        U = U.at[:g].set(U[g:(g+1)])
    elif bc_x1[0] == "reflective":
        U = U.at[:g].set(jnp.flip(U[g:(2*g)], axis=0))
    elif bc_x1[0] == "periodic":
        U = U.at[:g].set(U[(-2*g):(-g)])

    if bc_x1[1] == "outflow":
        U = U.at[-g:].set(U[-(g+1):-g])
    elif bc_x1[1] == "reflective":
        U = U.at[-g:].set(jnp.flip(U[-(2*g):-g], axis=0))
    elif bc_x1[1] == "periodic":
        U = U.at[-g:].set(U[g:(2*g)])

    if bc_x2[0] == "outflow":
        U = U.at[:, :g].set(U[:, g:(g+1)])
    elif bc_x2[0] == "reflective":
        U = U.at[:, :g].set(jnp.flip(U[:, g:(2*g)], axis=1))
    elif bc_x2[0] == "periodic":
        U = U.at[:, :g].set(U[:, (-2*g):(-g)])

    if bc_x2[1] == "outflow":
        U = U.at[:, -g:].set(U[:, -(g+1):-g])
    elif bc_x2[1] == "reflective":
        U = U.at[:, -g:].set(jnp.flip(U[:, -(2*g):-g], axis=1))
    elif bc_x2[1] == "periodic":
        U = U.at[:, -g:].set(U[:, g:(2*g)])
    
    if bc_x3[0] == "outflow":
        U = U.at[:, :, :g].set(U[:, :, g:(g+1)])
    elif bc_x3[0] == "reflective":
        U = U.at[:, :, :g].set(jnp.flip(U[:, :, g:(2*g)], axis=2))
    elif bc_x3[0] == "periodic":
        U = U.at[:, :, :g].set(U[:, :, (-2*g):(-g)])

    if bc_x3[1] == "outflow":
        U = U.at[:, :, -g:].set(U[:, :, -(g+1):-g])
    elif bc_x3[1] == "reflective":
        U = U.at[:, :, -g:].set(jnp.flip(U[:, :, -(2*g):-g], axis=2))
    elif bc_x3[1] == "periodic":
        U = U.at[:, :, -g:].set(U[:, :, g:(2*g)])

    return U

def enthalpy(rho: ArrayLike, p: ArrayLike, e: ArrayLike):
    return (e + p) / rho
    
def finite_difference_x1(lattice, u: ArrayLike, x1_g: ArrayLike, x2_g: ArrayLike) -> Array:
    g = lattice.num_g
    if lattice.coords == "cartesian":
        dx = lattice.x1[1] - lattice.x1[0]
        du = (u[(g):-(g-1), g:-g] - u[(g-1):-(g), g:-g]) / (dx)
    elif lattice.coords == "polar":
        X1, _ = jnp.meshgrid(x1_g[1:-1], lattice.x2, indexing="ij")
        dR = jnp.diff(X1, axis=0)
        du = jnp.diff(u[(g-1):-(g-1), g:-g], axis=0) / dR
    return du


def finite_difference_x2(lattice, u: ArrayLike, x1_g: ArrayLike, x2_g: ArrayLike) -> Array:
    g = lattice.num_g
    if lattice.coords == "cartesian":
        dy = lattice.x2[1] - lattice.x2[0]
        du = (u[g:-g, (g):-(g-1)] - u[g:-g, (g-1):-(g)]) / (dy)
    elif lattice.coords == "polar":
        dtheta = lattice.x2[1] - lattice.x2[0]
        X1, _ = jnp.meshgrid(lattice.x1, lattice.x2_intf, indexing="ij")
        du = jnp.diff(u[g:-g, (g-1):-(g-1)], axis=1) / (X1 * dtheta)
    return du

def minmod(x, y, z):
    return (1 / 4) * jnp.absolute(jnp.sign(x) + jnp.sign(y)) * (jnp.sign(x) + jnp.sign(z)) * jnp.minimum(jnp.minimum(jnp.absolute(x), jnp.absolute(y)), jnp.absolute(z))

""" plotting and I/O """

def find_latest_checkpoint(output_dir):
    """
    Function to find the latest checkpoint in the specified output directory by parsing the time in the filename.
    Assumes checkpoint files are named like 'out_x.xx.h5', where x.xx is the time.
    """
    # Find all .h5 files in the output directory
    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoints/out_*.h5"))
    
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the output directory.")
    
    # Regular expression to extract the time from the filename (e.g., 'out_1.24.h5' -> 1.24)
    pattern = re.compile(r'out_(\d+\.\d+)\.h5')

    # Extract times from the filenames and sort by time
    checkpoints_with_times = []
    for file in checkpoint_files:
        match = pattern.search(file)
        if match:
            time = float(match.group(1))  # Extract the time part as a float
            checkpoints_with_times.append((time, file))
    
    if not checkpoints_with_times:
        raise ValueError("No valid checkpoint files found with the expected naming convention.")
    
    # Sort by time and select the file with the largest time
    latest_checkpoint = max(checkpoints_with_times, key=lambda x: x[0])[1]
    
    return latest_checkpoint

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
    # Read the CSV file and treat 'nan' as missing values
    df = pd.read_csv(path, na_values=['nan', 'inf', '-inf'], on_bad_lines="skip")
    # Convert all columns to numeric, forcing errors='coerce' to convert any bad strings to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    # Replace inf and -inf with NaN explicitly (if they somehow sneak through)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows containing any NaN values (including those converted from inf)
    df = df.dropna()
    
    # Initialize a dictionary to store the numpy arrays
    numpy_arrays = {}

    # Iterate over each column in the dataframe and save it as a numpy array
    for column in df.columns:
        if compress:
            numpy_arrays[column] = df[column].values[::2]
        else:
            numpy_arrays[column] = df[column].values
    
    return numpy_arrays


def save_to_h5(filename, t, U, hydro, lattice):
    rho, momx1, momx2, E = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    with h5py.File(filename, "w") as f:
        # metadata
        f.attrs["coords"] = lattice.coords
        f.attrs["gamma"] = hydro.gamma()
        f.attrs["x1"] = lattice.x1
        f.attrs["x2"] = lattice.x2
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


def plot_grid(matrix, label, coords, x1, x2, vmin=None, vmax=None, cmap="magma"):
    extent = [x1[0], x1[-1], x2[0], x2[-1]]

    if coords == "cartesian":
        fig, ax = plt.subplots()
        if vmin is None:
            vmin, vmax = jnp.min(matrix), jnp.max(matrix)
        c = ax.imshow(jnp.transpose(matrix), cmap=cmap, interpolation='nearest',
                      origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    elif coords == "polar":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        if vmin is None:
            vmin, vmax = jnp.min(matrix), jnp.max(matrix)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 5)
        ax.set_facecolor("black")
        circle_r = jnp.min(x1) - (x1[1] - x1[0]) / 2
        circle = Circle((0, 0), radius=circle_r, transform=ax.transData._b,
                        color='blue', fill=False, linewidth=1)
        ax.add_patch(circle)
        R, Theta = jnp.meshgrid(x1, x2, indexing="ij")
        c = ax.pcolormesh(Theta, R, matrix, shading='auto',
                          cmap=cmap, vmin=vmin, vmax=vmax)

    cb = plt.colorbar(c, ax=ax, label=label)

    return fig, ax, c, cb