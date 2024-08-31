from . import run_config
import click

@click.group()
def cli():
    pass

@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path())
@click.option("--plot", type=click.Choice(["density", "log density", "u", "v", "pressure", "energy"]))
@click.option("--plot-range", type=(float, float))
@click.option("--output-dir", type=click.Path())
def run(config_file, checkpoint, plot, plot_range, output_dir):
    run_config(config_file, checkpoint, plot, plot_range, output_dir)


@click.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
def plot(checkpoint_file):
    pass


cli.add_command(run)
cli.add_command(plot)


if __name__ == "__main__":
    cli()