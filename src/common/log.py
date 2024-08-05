import time

import jax.numpy as jnp

from rich.console import Console
from rich.theme import Theme
from rich.live import Live
from rich.panel import Panel
from rich.table import Table, Column
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn


class Logger(Live):
    def __init__(self):
        self.log_freq = 1000
        complete_column = MofNCompleteColumn(
            table_column=Column(justify="left"))
        bar_column = BarColumn(
            bar_width=None, table_column=Column(justify="right"))
        self.progress = Progress(bar_column, complete_column, expand=True)
        self.task = self.progress.add_task("", total=self.log_freq)
        self.n_start = 1
        self.min_dt = 1
        self.run_start = time.time()
        self.log_start = time.time()
        
        super().__init__(console=Console(theme=Theme({"bar.complete": "red"})), refresh_per_second=4)
        
    def panel(self, lattice, n, t):
        elapsed = time.time() - self.log_start
        mzps = (lattice.nx1 * lattice.nx2 * (n - self.n_start) / elapsed) / 1e6

        left_grid = Table.grid(expand=True)
        left_grid.add_column(ratio=1, justify="left")
        left_grid.add_column(ratio=1, justify="right")
        left_grid.add_row("t", f"{t:.2f}")
        left_grid.add_row("min timestep", f"{self.min_dt:.2e}")
        right_grid = Table.grid(expand=True)
        right_grid.add_column(ratio=1, justify="left")
        right_grid.add_column(ratio=1, justify="right")
        right_grid.add_row("time elapsed", time.strftime(
            "%H:%M:%S", time.gmtime(elapsed)))
        right_grid.add_row("mzps", f"{mzps:.2e}")
        stats_grid = Table.grid(expand=True, padding=(0, 10))
        stats_grid.add_column(ratio=1, justify="left")
        stats_grid.add_column(ratio=1, justify="right")
        stats_grid.add_row(left_grid, right_grid)
        grid = Table.grid(expand=True)
        grid.add_row(self.progress)
        grid.add_row(stats_grid)
        return Panel(grid, title=f"timesteps {self.n_start}-{self.n_start + (self.log_freq - 1)}", border_style="grey50")

    def update_logs(self, lattice, n, t, dt):
        self.min_dt = jnp.minimum(self.min_dt, dt)
        self.update(self.panel(lattice, n, t))
        self.progress.update(self.task, advance=1)
        if (n - 1) % self.log_freq == 0:
            self.reset(lattice, n, t)

    def reset_progress(self):
        self.progress.remove_task(self.task)
        self.task = self.progress.add_task("", total=self.log_freq)

    def reset(self, lattice, n, t):
        self.console.print(self.panel(lattice, n, t))
        self.n_start = n
        self.reset_progress()
        self.log_start = time.time()
        self.min_dt = 1
    
    def print_summary(self, lattice, n):
        elapsed = time.time() - self.run_start
        mzps = (lattice.nx1 * lattice.nx2 * n / elapsed) / 1e6
        self.console.print(f"[bold]time elapsed[/bold] {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        self.console.print(f"[bold]average speed[/bold] {mzps:.2e} mzps")