import click
import h5py
import numpy as np
import matplotlib.pyplot as plt

from . import run_config, load_config
from .tools import generate_movie
from src.common.helpers import plot_grid

@click.group()
def cli():
    pass

class DynamicCommand(click.Command):
    def parse_args(self, ctx, args):
        # Extract config_file argument from args
        config_file = args[0]

        # Load the config class
        config_class = load_config(config_file)
        self.original_names = {}
        for name, value in vars(config_class()).items():
            self.params.append(click.Option([f"--{name}"], default=value))
            self.original_names[name.lower()] = name
        return super().parse_args(ctx, args)

@click.command(cls=DynamicCommand)
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path())
@click.option("--plot", type=click.Choice(["density", "log density", "u", "v", "pressure", "energy"]))
@click.option("--plot-range", type=(float, float))
@click.option("--output-dir", type=click.Path())
def run(config_file, checkpoint, plot, plot_range, output_dir, **kwargs):
    ctx = click.get_current_context()
    dynamic_command = ctx.command
    # Recover original variable names
    original_kwargs = {}
    for k, v in kwargs.items():
        original_key = dynamic_command.original_names[k]
        original_kwargs[original_key] = v
    run_config(config_file, checkpoint, plot, plot_range, output_dir, **original_kwargs)

@click.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option("-v", "--var", type=click.Choice(["density", "log density", "u", "v", "energy"]), default="density")
@click.option("-r", "--range", type=(float, float, float, float), default=None)
@click.option("--title", type=str, default="")
@click.option("--dpi", type=int, default=500)
@click.option("--cmap", type=str, default="magma")
@click.option("--c-range", type=(float, float))
def plot(checkpoint_file, var, range, title, dpi, cmap, c_range):
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$", "v": r"$v$", "energy": r"$E$"}
    vmin, vmax = None, None
    if c_range:
        vmin, vmax = c_range
    
    with h5py.File(checkpoint_file, 'r') as f:
        t = f.attrs["t"]
        coords = f.attrs["coords"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        rho, momx1, momx2, e = np.array(f["rho"]), np.array(f["momx1"]), np.array(f["momx2"]), np.array(f["E"])
        
        if var == "density":
            matrix = rho
        elif var == "log density":
            matrix = np.log10(rho)
        elif var == "u":
            matrix = momx1 / rho
        elif var == "v":
            matrix = momx2 / rho
        elif var == "energy":
            matrix = e

        if range:
            x1_min, x1_max = range[0], range[1]
            x2_min, x2_max = range[2], range[3]
            x1_min_i = np.searchsorted(x1, x1_min, side="left")
            x1_max_i = np.searchsorted(x1, x1_max, side="right") - 1
            x2_min_i = np.searchsorted(x2, x2_min, side="left")
            x2_max_i = np.searchsorted(x2, x2_max, side="right") - 1

            matrix = matrix[x1_min_i:x1_max_i+1, x2_min_i:x2_max_i+1]
            x1, x2 = x1[(x1 >= x1_min) & (x1 <= x1_max)], x2[(x2 >= x2_min) & (x2 <= x2_max)]
        
        fig, ax, c, cb = plot_grid(matrix, labels[var], coords, x1, x2, vmin, vmax, cmap)
        if title != "":
            ax.set_title(title + f", t = {t:.2f}")
        else:
            ax.set_title(f"t = {t:.2f}")
        PATH = checkpoint_file.split("checkpoints/")[0]
        plt.savefig(f"{PATH}/t={t:.2f}.png", bbox_inches="tight", dpi=dpi)
        plt.show()
        
@click.command()
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option("-t", "--t-range", type=(float, float))
@click.option("-v", "--var", type=click.Choice(["density", "log density", "u", "v", "energy"]), default="density")
@click.option("-r", "--range", type=(float, float, float, float), default=None)
@click.option("--title", type=str, default="")
@click.option("--fps", type=int, default=24)
@click.option("--dpi", type=int, default=200)
@click.option("--cmap", type=str, default="magma")
@click.option("--c-range", type=(float, float))
def movie(checkpoint_path, t_range, var, range, title, fps, dpi, cmap, c_range):
    vmin, vmax = None, None
    if c_range:
        vmin, vmax = c_range
    t_min, t_max = t_range
        
    generate_movie(checkpoint_path, t_min, t_max, var, range, title, fps, vmin, vmax, dpi, cmap)

cli.add_command(run)
cli.add_command(plot)
cli.add_command(movie)

if __name__ == "__main__":
    cli()