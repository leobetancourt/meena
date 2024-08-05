from . import run_config
import click

@click.group()
def cli():
    pass

@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--plot", type=click.Choice(["density", "log density", "u", "v", "pressure", "energy"]))
@click.option("--output_dir", type=click.Path())
def run(config_file, plot, output_dir):
    run_config(config_file, plot, output_dir)


@click.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
def plot(checkpoint_file):
    pass


cli.add_command(run)
cli.add_command(plot)


if __name__ == "__main__":
    cli()