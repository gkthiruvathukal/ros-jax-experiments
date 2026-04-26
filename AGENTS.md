# Agent Instructions

## Project overview

This repo benchmarks JAX on Apple Silicon (macOS) with CPU and Metal GPU backends.
ROS experiments will be added in a future phase — do not create ROS-related code yet.

## Environment setup

Always use the Python 3.12 virtual environment:

```bash
make install       # creates .venv and installs pinned deps
```

Run scripts via `.venv/bin/python`, not the system Python. Never upgrade JAX or
jaxlib without checking jax-metal compatibility first — see the version pinning
note in README.md.

## Key commands

```bash
make test          # correctness checks + quick CPU vs Metal table
make benchmark     # full size sweep → benchmark.png
make clean         # remove .venv
```

## Critical constraint: JAX version pinning

`jax==0.4.34` and `jaxlib==0.4.34` must stay in sync with `jax-metal==0.1.1`.
Newer JAX versions (0.5+) break the Metal backend with an `UNIMPLEMENTED:
default_memory_space is not supported` error. Do not bump these versions unless
a compatible jax-metal release is available.

## Adding benchmarks

- New operations go in `benchmark_plot.py` — add an `@jit` function and append
  to the `OPS` list.
- New correctness checks go in `jax_test.py`.
- Always test on both CPU and Metal devices using explicit `jax.device_put`.

## Style

- No comments unless the why is non-obvious.
- No docstrings beyond the module-level one-liner.
- Keep benchmark output human-readable — progress lines during the run, summary at the end.
