import sys
import importlib
import inspect
from pathlib import Path
from .detail import Hydro, Lattice
from src import run
from src.common.helpers import load_U


def run_config(config_file, checkpoint, plot, plot_range, output_dir):
    config_path = Path(config_file)
    sys.path.append(str(config_path.parent.parent / 'configs'))
    config_name = config_path.stem
    config_module = importlib.import_module(config_name)

    for name_local in dir(config_module):
        if inspect.isclass(getattr(config_module, name_local)):
            Config = getattr(config_module, name_local)
            # check if Config is a subclass of Hydro (exclude Hydro itself)
            if Config != Hydro and issubclass(Config, Hydro):
                # TODO: make BaseConfig a separate class to Hydro
                hydro = Config()

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

                if checkpoint:  # user specified a checkpoint file to run from
                    U, t = load_U(checkpoint)
                    print(t)
                else:
                    U, t = hydro.initialize(
                        lattice.X1, lattice.X2), hydro.t_start()

                out = output_dir if output_dir else f"./output/{config_name}"

                run(
                    hydro,
                    lattice,
                    U=U,
                    t=t,
                    T=hydro.t_end(),
                    N=None,
                    plot=plot,
                    plot_range=plot_range,
                    out=out,
                    save_interval=hydro.save_interval(),
                    diagnostics=hydro.diagnostics()
                )
