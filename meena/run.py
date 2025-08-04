import sys
import importlib
import inspect
import os
import jax.numpy as jnp

from pathlib import Path
from .detail import Hydro, Lattice
from src.common.helpers import load_U
from src.hydro.main import run

def load_config(config_file):
    config_path = Path(config_file)
    sys.path.append(str(config_path.parent.parent / 'configs'))
    config_name = config_path.stem
    config_module = importlib.import_module(config_name)
    
    for name_local in dir(config_module):
        obj = getattr(config_module, name_local)
        if inspect.isclass(obj):
            if obj.__name__ != 'Hydro' and obj.__module__ != "builtins" and issubclass(obj, Hydro):
                return obj
    return None

def run_config(config_file, checkpoint, plot, save_plots, plot_range, output_dir, **kwargs):
    config_class = load_config(config_file)
    hydro = config_class(**kwargs)
    
    lattice = Lattice(
        coords=hydro.coords(),
        bc_x1=hydro.bc_x1(),
        bc_x2=hydro.bc_x2(),
        nx1=hydro.resolution()[0],
        nx2=hydro.resolution()[1],
        x1_range=hydro.range()[0],
        x2_range=hydro.range()[1],
        num_g=hydro.num_g(),
        log_x1=hydro.log_x1(),
        log_x2=hydro.log_x2()
    )
    
    out = output_dir if output_dir else f"./output/{Path(config_file).stem}"
    os.makedirs(out, exist_ok=True)

    if checkpoint:  # user specifies a checkpoint file to run from
        prims, _, _, t = load_U(checkpoint)
        # If any invalid values (NaNs), fall back to initialized values for those
        prims_init = hydro.initialize(lattice)
        bad_mask = jnp.any(jnp.isnan(prims), axis=-1)  # shape (NX1, NX2)

        # Replace bad cells with initialized values
        prims = jnp.where(bad_mask[..., None], prims_init, prims)
    else:
        prims, t = hydro.initialize(
            lattice), hydro.t_start()

        with open(Path(out) / "setup.txt", "w") as f:
            for key, value in vars(hydro).items():
                f.write(f"{key} = {value}\n")

    run(
        hydro,
        lattice,
        prims=prims,
        t=t,
        T=hydro.t_end(),
        N=None,
        plot=plot,
        save_plots=save_plots,
        plot_range=plot_range,
        out=out,
        save_interval=hydro.save_interval(),
        diagnostics=hydro.diagnostics()
    )