import click
import h5py
import numpy as np
import matplotlib.pyplot as plt

from . import run_config, load_config
from .tools import generate_movie
from src.common.helpers import plot_grid, find_latest_checkpoint

@click.group()
def cli():
    pass

class DynamicCommand(click.Command):
    def parse_args(self, ctx, args):
        # Extract config_file argument from args
        config_file = args[0]

        # Load the config class
        config_class = load_config(config_file)
        self.og_params = {}
        for param, value in vars(config_class()).items():
            new_param = param.replace("_", "-")
            self.params.append(click.Option([f"--{new_param}"], default=value))
            self.og_params[new_param.lower()] = param
        return super().parse_args(ctx, args)

@click.command(cls=DynamicCommand)
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(), help="Path to a specific checkpoint file.")
@click.option("--plot", type=click.Choice(["density", "log density", "u", "v", "pressure", "energy"]))
@click.option("--plot-range", type=(float, float))
@click.option("--output-dir", type=click.Path())
@click.option("--resume", is_flag=True, help="Resume simulation from the latest checkpoint in given output directory.")
def run(config_file, checkpoint, plot, plot_range, output_dir, resume, **kwargs):
    ctx = click.get_current_context()
    dynamic_command = ctx.command
    og_kwargs = {}
    
    for k, v in kwargs.items():
        og_key = dynamic_command.og_params[k.replace("_", "-")]
        og_kwargs[og_key] = v
        
    if resume:
        checkpoint = find_latest_checkpoint(output_dir)    
    
    run_config(config_file, checkpoint, plot, plot_range, output_dir, **og_kwargs)

@click.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option("-v", "--var", type=click.Choice(["density", "log density", "u", "v", "w", "energy", "pressure", "bx", "by", "bz", "div B"]), default="density")
@click.option("-r", "--range", type=(float, float, float, float), default=None)
@click.option("--title", type=str, default="")
@click.option("--dpi", type=int, default=500)
@click.option("--cmap", type=str, default="magma")
@click.option("--c-range", type=(float, float))
@click.option("--t-factor", type=float, default=1)
@click.option("--t-units", type=str, default="")
def plot(checkpoint_file, var, range, title, dpi, cmap, c_range, t_factor, t_units):
    labels = {"density": r"$\rho$", "log density": r"$\log_{10} \Sigma / \Sigma_0$", "u": r"$u$", "v": r"$v$", "energy": r"$E$", "pressure": r"$P$", "bx": r"$B_x$", "by": r"$B_y$", "bz": r"$B_z$", "div B": r"$\nabla \cdot B$"}
    vmin, vmax = None, None
    if c_range:
        vmin, vmax = c_range
    
    with h5py.File(checkpoint_file, 'r') as f:
        t = f.attrs["t"]
        coords = f.attrs["coords"]
        x1 = f.attrs["x1"]
        x2 = f.attrs["x2"]
        x3 = f.attrs["x3"]
        dx = x1[1] - x1[0]
        if len(x2) > 1:
            dy = x2[1] - x2[0]
        gamma = f.attrs["gamma"]
        rho = np.array(f["rho"])
        u, v, w, e = np.array(f["momx1"]) / rho, np.array(f["momx2"]) / rho, np.array(f["momx3"]) / rho, np.array(f["E"])
        Bx, By, Bz = np.array(f["Bx"]), np.array(f["By"]), np.array(f["Bz"])
            
        if var == "density":
            matrix = rho
        elif var == "log density":
            matrix = np.log10(rho)
        elif var == "u":
            matrix = u
        elif var == "v":
            matrix = v
        elif var == "w":
            matrix = w
        elif var == "energy":
            matrix = e
        elif var == "pressure":
            u2 = u**2 + v**2 + w**2
            B2 = Bx**2 + By**2 + Bz**2
            matrix = (gamma - 1) * (e - 0.5 * (rho * u2 + B2))
        elif var == "bx":
            matrix = Bx
        elif var == "by":
            matrix = By
        elif var == "bz":
            matrix = Bz
        elif var == "div B":
            matrix = (Bx[2:, 1:-1] - Bx[:-2, 1:-1]) / (2 * dx) + (By[1:-1, 2:] - By[1:-1, :-2]) / (2 * dy)
            x1, x2 = x1[1:-1], x2[1:-1]
        
        if len(x2) == 1: # this is a 1D simulation
            fig, ax = plt.subplots()
            ax.plot(x1, matrix[:, 0, 0])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(labels[var])
        else:
            fig, ax, c, cb = plot_grid(matrix[..., matrix.shape[-1] // 2], labels[var], coords, x1, x2, vmin, vmax, cmap)
        if title == "":
            ax.set_title(f"t = {(t*t_factor):.2f} {t_units}")
        else:
            ax.set_title(title + f", t = {(t*t_factor):.2f} {t_units}")
        PATH = checkpoint_file.split("checkpoints/")[0]
        plt.savefig(f"{PATH}/t={t:.2f}.png", bbox_inches="tight", dpi=dpi)
        plt.show()
        
@click.command()
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option("-t", "--t-range", type=(float, float))
@click.option("-v", "--var", type=click.Choice(["density", "log density", "u", "v", "w", "energy", "pressure", "bx", "by", "bz", "div B"]), default="density")
@click.option("-r", "--range", type=(float, float, float, float), default=None)
@click.option("--title", type=str, default="")
@click.option("--fps", type=int, default=24)
@click.option("--dpi", type=int, default=300)
@click.option("--bitrate", type=int, default=-1)
@click.option("--cmap", type=str, default="magma")
@click.option("--c-range", type=(float, float))
@click.option("--t-factor", type=float, default=1)
@click.option("--t-units", type=str, default="")
def movie(checkpoint_path, t_range, var, range, title, fps, dpi, bitrate, cmap, c_range, t_factor, t_units):
    vmin, vmax = None, None
    if c_range:
        vmin, vmax = c_range
    t_min, t_max = t_range
        
    generate_movie(checkpoint_path, t_min, t_max, var, range, title, fps, vmin, vmax, dpi, bitrate, cmap, t_factor, t_units)

cli.add_command(run)
cli.add_command(plot)
cli.add_command(movie)

if __name__ == "__main__":
    cli()