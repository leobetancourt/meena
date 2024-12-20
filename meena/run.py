import sys
import importlib
import inspect
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

def run_config(config_file, checkpoint, plot, plot_range, output_dir, **kwargs):
    config_class = load_config(config_file)
    hydro = config_class(**kwargs)
    
    if isinstance(hydro.resolution(), int):
        dims = 1
    else:
        dims = len(hydro.resolution())
    nx1, nx2, nx3 = 1, 1, 1
    x1_range, x2_range, x3_range = (0, 1), (0, 1), (0, 1)
    if dims == 1:
        nx1 = hydro.resolution()
        x1_range = hydro.range()
    elif dims == 2:
        nx1, nx2 = hydro.resolution()
        x1_range, x2_range = hydro.range()
    elif dims == 3:
        nx1, nx2, nx3 = hydro.resolution()
        x1_range, x2_range, x3_range = hydro.range()
            
    lattice = Lattice(
        coords=hydro.coords(),
        dims = dims,
        bc_x1=hydro.bc_x1(),
        bc_x2=hydro.bc_x2(),
        bc_x3=hydro.bc_x3(),
        nx1=nx1,
        nx2=nx2,
        nx3=nx3,
        x1_range=x1_range,
        x2_range=x2_range,
        x3_range=x3_range,
        num_g=hydro.num_g(),
        log_x1=hydro.log_x1(),
        log_x2=hydro.log_x2(),
        log_x3=hydro.log_x3(),
    )

    B = None
    if checkpoint:  # user specifies a checkpoint file to run from
        if hydro.regime() == "HD":
            U, t = load_U(checkpoint)
        elif hydro.regime() == "MHD":
            U, B, t = load_U(checkpoint)
    else:
        if hydro.regime() == "HD":
            U = hydro.initialize(lattice)
        elif hydro.regime() == "MHD":
            U, B = hydro.initialize(lattice)
        t = hydro.t_start()

    out = output_dir if output_dir else f"./output/{Path(config_file).stem}"

    run(
        hydro,
        lattice,
        U=U,
        B=B,
        t=t,
        T=hydro.t_end(),
        N=None,
        plot=plot,
        plot_range=plot_range,
        out=out,
        save_interval=hydro.save_interval(),
        diagnostics=hydro.diagnostics()
    )