# Agent Instructions

## Project overview

This repo benchmarks JAX on Apple Silicon (macOS) with CPU and Metal GPU backends,
and on Linux with CUDA. ROS experiments will be added in a future phase — do not
create ROS-related code yet.

## Environment setup

Always use the Python 3.12 virtual environment:

```bash
make install              # Metal backend (default, Apple Silicon)
make install BACKEND=cuda # CUDA backend (Linux + NVIDIA)
make install BACKEND=cpu  # CPU-only (any platform)
```

Run scripts via `.venv/bin/python`, not the system Python. Never upgrade JAX or
jaxlib without checking jax-metal compatibility first — see the version pinning
note in README.md.

## Key commands

```bash
make test          # correctness checks + quick benchmark table
make benchmark     # full size sweep → benchmark.png
make clean         # remove .venv
```

Both scripts also accept CLI flags — run with `--help` for details.

## Critical constraint: JAX version pinning

`jax==0.4.34` and `jaxlib==0.4.34` must stay in sync with `jax-metal==0.1.1`
(see `requirements-metal.txt`). Newer JAX versions (0.5+) break the Metal backend
with an `UNIMPLEMENTED: default_memory_space is not supported` error. Do not bump
these versions unless a compatible jax-metal release is available.

This constraint applies to the Metal backend only. The CUDA backend
(`requirements-cuda.txt`) is unpinned and can use a current JAX release.

## Requirements files

| File | Backend | Platform |
|------|---------|----------|
| `requirements-metal.txt` | Metal (jax-metal) | macOS / Apple Silicon |
| `requirements-cuda.txt`  | CUDA (jax[cuda12]) | Linux + NVIDIA |
| `requirements-cpu.txt`   | CPU only | any |

Do not install `jax-metal` and the CUDA jaxlib in the same environment —
they are mutually exclusive.

## Adding benchmarks

- New operations go in `benchmark_plot.py` — add an `@jit` function and add an
  entry to the `ALL_OPS` dict.
- New correctness checks go in `jax_test.py` inside `run_correctness()`.
- Always test on both CPU and the available accelerator using explicit
  `jax.device_put`. Device detection is handled by `detect_devices()` in each
  script — do not hard-code device references.

## Style

- No comments unless the why is non-obvious.
- No docstrings beyond the module-level one-liner.
- Keep benchmark output human-readable — progress lines during the run, summary at the end.
