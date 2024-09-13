import click
import h5py
import numpy as np
import matplotlib.pyplot as plt

from . import run_config, load_config
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
        for name, value in vars(config_class()).items():
            self.params.append(click.Option([f"--{name}"], default=value))
        
        return super().parse_args(ctx, args)

@click.command(cls=DynamicCommand)
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path())
@click.option("--plot", type=click.Choice(["density", "log density", "u", "v", "pressure", "energy"]))
@click.option("--plot-range", type=(float, float))
@click.option("--output-dir", type=click.Path())
def run(config_file, checkpoint, plot, plot_range, output_dir, **kwargs):
    run_config(config_file, checkpoint, plot, plot_range, output_dir, **kwargs)

@click.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option("-v", "--var", type=click.Choice(["density", "log density", "u", "v", "energy"]), default="density")
@click.option("--plot-range", type=(float, float))
def plot(checkpoint_file, var, plot_range):
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma$", "u": r"$u$", "v": r"$v$", "energy": r"$E$"}
    vmin, vmax = None, None
    if plot_range:
        vmin, vmax = plot_range
        
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
        
        fig, ax, c, cb = plot_grid(matrix, labels[var], coords, x1, x2, vmin, vmax)
        ax.set_title(f"t = {t:.2f}")
        plt.show()

cli.add_command(run)
cli.add_command(plot)


if __name__ == "__main__":
    cli()