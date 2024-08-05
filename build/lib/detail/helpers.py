import jax.numpy as jnp

def linspace_cells(min, max, num):
    interfaces = jnp.linspace(min, max, num + 1)
    centers = (interfaces[:-1] + interfaces[1:]) / 2

    return centers, interfaces


def logspace_cells(min, max, num):
    interfaces = jnp.logspace(jnp.log10(min), jnp.log10(max), num + 1)
    centers = (interfaces[:-1] + interfaces[1:]) / 2

    return centers, interfaces