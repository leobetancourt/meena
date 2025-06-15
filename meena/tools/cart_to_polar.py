import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
import os

def load_cartesian_checkpoint(file):
    with h5py.File(file, 'r') as f:
        t = f.attrs["t"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        rho = f["rho"][()]
        u = f["u"][()]
        v = f["v"][()]
        p = f["p"][()]

    # Shape: (nx, ny)
    prims = np.stack([rho, u, v, p], axis=-1)
    return t, x1, x2, prims

def cartesian_to_polar_grid(x1, x2, prims, r_min, r_max, nr, theta_min, theta_max, ntheta):
    # Cartesian grid points
    X, Y = np.meshgrid(x1, x2, indexing='ij')

    # Target polar grid
    r = np.linspace(r_min, r_max, nr)
    theta = np.linspace(theta_min, theta_max, ntheta)
    R, Theta = np.meshgrid(r, theta, indexing='ij')

    # Convert polar to Cartesian coordinates
    Xp = R * np.cos(Theta)
    Yp = R * np.sin(Theta)

    # Interpolators (one for each primitive variable)
    nx, ny, _ = prims.shape
    interpolators = [
        RegularGridInterpolator((x1, x2), prims[..., i], bounds_error=False, fill_value=None)
        for i in range(4)
    ]

    # Flatten (r, θ) grid for interpolation
    coords = np.stack([Xp.ravel(), Yp.ravel()], axis=-1)

    # Interpolate
    polar_prims = np.stack([interp(coords).reshape((nr, ntheta)) for interp in interpolators], axis=-1)

    return r, theta, polar_prims

def save_polar_checkpoint(output_file, t, r, theta, prims):
    with h5py.File(output_file, 'w') as f:
        f.attrs["t"] = t
        f.attrs["x1"] = r
        f.attrs["x2"] = theta
        f.attrs["coords"] = "polar"
        
        f.create_dataset("rho", data=prims[..., 0])
        f.create_dataset("u",   data=prims[..., 1])
        f.create_dataset("v",   data=prims[..., 2])
        f.create_dataset("p",   data=prims[..., 3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Cartesian checkpoint file (HDF5)")
    parser.add_argument("output", help="Output polar checkpoint file (HDF5)")
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, required=True)
    parser.add_argument("--nr", type=int, default=256)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=2*np.pi)
    parser.add_argument("--ntheta", type=int, default=256)
    args = parser.parse_args()

    t, x1, x2, prims = load_cartesian_checkpoint(args.input)
    r, theta, polar_prims = cartesian_to_polar_grid(
        x1, x2, prims,
        args.r_min, args.r_max, args.nr,
        args.theta_min, args.theta_max, args.ntheta
    )
    save_polar_checkpoint(args.output, t, r, theta, polar_prims)
    print(f"Saved polar checkpoint to {args.output}")

if __name__ == "__main__":
    main()