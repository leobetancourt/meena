# Meena



A GPU-accelerated Godunov hydrodynamics code written in [JAX](https://github.com/jax-ml/jax).

https://github.com/user-attachments/assets/4a9abd93-4475-446b-8e73-1dcc3d937822

This code supports Newtonian hydrodynamics up to 2D and compiles on CPU/GPU/TPU from the same Python code base. See [just-in-time compilation](https://jax.readthedocs.io/en/latest/jit-compilation.html).

Meena implements the [HLL and HLLC Riemann solvers](https://link.springer.com/chapter/10.1007/978-3-662-03490-3_10) and is second order accurate in space using piecewise-linear spatial reconstruction. 


## Quick-start

Install with:

```bash
cd meena
python setup.py install
```

Run a configuration script:

```bash
meena run configs/RayleighTaylor.py --nx 1000 --gamma-ad 1.4
```
Command line arguments `nx` and `gamma-ad` are dynamically parsed from the config class (a subclass of `Hydro`). See the `configs/` directory for examples.

## Notes

This code was adapted from Weiqun Zhang's [How To Write A Hydrodynamics Code](http://duffell.org/media/hydro.pdf). 
