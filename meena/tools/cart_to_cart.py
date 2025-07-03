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

    prims = np.stack([rho, u, v, p], axis=-1)
    return t, x1, x2, prims

def regrid_cartesian_to_cartesian(x1, x2, prims, x_min, x_max, nx, y_min, y_max, ny):
    # Original grid
    X, Y = np.meshgrid(x1, x2, indexing='ij')

    # New grid
    x_new = np.linspace(x_min, x_max, nx)
    y_new = np.linspace(y_min, y_max, ny)
    Xn, Yn = np.meshgrid(x_new, y_new, indexing='ij')

    # Interpolators
    interpolators = [
        RegularGridInterpolator((x1, x2), prims[..., i], bounds_error=False, fill_value=np.nan)
        for i in range(4)
    ]

    coords = np.stack([Xn.ravel(), Yn.ravel()], axis=-1)
    new_prims = np.stack([interp(coords).reshape((nx, ny)) for interp in interpolators], axis=-1)

    return x_new, y_new, new_prims

def save_cartesian_checkpoint(output_file, t, x1, x2, prims):
    with h5py.File(output_file, 'w') as f:
        f.attrs["t"] = t
        f.attrs["x1"] = x1
        f.attrs["x2"] = x2
        f.attrs["coords"] = "cartesian"
        
        f.create_dataset("rho", data=prims[..., 0])
        f.create_dataset("u",   data=prims[..., 1])
        f.create_dataset("v",   data=prims[..., 2])
        f.create_dataset("p",   data=prims[..., 3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Cartesian checkpoint file (HDF5)")
    parser.add_argument("output", help="Output regridded checkpoint file (HDF5)")
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--ny", type=int, default=256)
    args = parser.parse_args()

    t, x1, x2, prims = load_cartesian_checkpoint(args.input)
    x_new, y_new, new_prims = regrid_cartesian_to_cartesian(
        x1, x2, prims,
        args.x_min, args.x_max, args.nx,
        args.y_min, args.y_max, args.ny
    )
    save_cartesian_checkpoint(args.output, t, x_new, y_new, new_prims)
    print(f"Saved regridded Cartesian checkpoint to {args.output}")

if __name__ == "__main__":
    main()
