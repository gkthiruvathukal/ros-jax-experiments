"""JAX correctness checks and quick CPU vs accelerator benchmark.

JAX 0.4.34 + jax-metal 0.1.1 required on Apple Silicon — newer JAX versions
have a plugin API change (default_memory_space) that jax-metal has not yet
implemented.

Usage examples
--------------
  # default: correctness on cpu, benchmark on all detected devices
  python jax_test.py

  # benchmark cpu and metal only
  python jax_test.py --devices cpu metal

  # skip correctness checks, run benchmark on cuda
  python jax_test.py --devices cuda --no-correctness

  # skip benchmark, just run correctness
  python jax_test.py --no-benchmark
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap


# ── Device detection ──────────────────────────────────────────────────────────

def detect_devices():
    """Return a dict of available named devices: {'cpu': ..., 'metal': ..., 'cuda': ...}."""
    found = {}
    for d in jax.devices():
        s = str(d).upper()
        if "METAL" in s:
            found.setdefault("metal", d)
        elif "CUDA" in s or "GPU" in s:
            found.setdefault("cuda", d)
    found["cpu"] = jax.devices("cpu")[0]
    return found


def resolve_devices(requested, available):
    resolved = {}
    for name in requested:
        if name not in available:
            print(f"WARNING: device '{name}' not available — skipping.")
        else:
            resolved[name] = available[name]
    if not resolved:
        sys.exit("ERROR: none of the requested devices are available. Aborting.")
    return resolved


# ── Ops ───────────────────────────────────────────────────────────────────────

@jit
def op_matmul(a):    return jnp.dot(a, a)

@jit
def op_elemwise(a):  return jnp.exp(jnp.sin(a) + jnp.cos(a))

@jit
def op_reduction(a): return jnp.sum(a ** 2)

BENCHMARKS = [
    ("matmul",    op_matmul,    [(256, 256), (1024, 1024), (4096, 4096)]),
    ("elem-wise", op_elemwise,  [(256, 256), (1024, 1024), (4096, 4096)]),
    ("reduction", op_reduction, [(256, 256), (1024, 1024), (4096, 4096)]),
]


# ── Correctness checks ────────────────────────────────────────────────────────

def run_correctness():
    print("=== Correctness ===")

    x = jnp.arange(1, 6, dtype=jnp.float32)
    assert float(jnp.sum(x)) == 15.0
    print(f"[1] arange + sum    : {x}  sum={jnp.sum(x)}")

    def loss(w):
        return jnp.sum((w - 3.0) ** 2)

    w = jnp.array([1.0, 2.0, 4.0])
    g = jit(grad(loss))(w)
    assert list(g) == [-4.0, -2.0, 2.0], f"unexpected grad: {g}"
    print(f"[2] autograd        : grad={g}  (expected [-4, -2, 2])")

    batched_dot = jit(vmap(lambda row, v: jnp.dot(row, v), in_axes=(0, None)))
    key = jax.random.PRNGKey(0)
    mat = jax.random.normal(key, (8, 4))
    vec = jax.random.normal(key, (4,))
    out = batched_dot(mat, vec)
    assert out.shape == (8,)
    print(f"[3] vmap            : {mat.shape} · {vec.shape} -> {out.shape}")
    print()


# ── Benchmark ─────────────────────────────────────────────────────────────────

def bench(fn, x, repeats=20):
    fn(x).block_until_ready()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def make_input(shape, device):
    k = jax.random.PRNGKey(0)
    return jax.device_put(jax.random.normal(k, shape), device)


def run_benchmark(devices, repeats=20):
    dev_names = list(devices)

    header = f"{'operation':<12}  {'shape':<14}" + "".join(f"  {n:>9}" for n in dev_names)
    if len(dev_names) > 1:
        header += f"  {'speedup':>9}"
    print(f"=== Quick benchmark (median ms/call, {repeats} reps) ===")
    print(header)
    print("-" * (30 + 11 * len(dev_names) + (9 if len(dev_names) > 1 else 0)))

    for name, fn, shapes in BENCHMARKS:
        for shape in shapes:
            label = f"{shape[0]}×{shape[1]}"
            ms = {}
            for dev_name, device in devices.items():
                ms[dev_name] = bench(fn, make_input(shape, device), repeats)

            row = f"{name:<12}  {label:<14}" + "".join(f"  {ms[n]:>8.2f}" for n in dev_names)
            if len(dev_names) == 2:
                d0, d1 = dev_names
                speedup = ms[d0] / ms[d1] if ms[d1] else float("nan")
                row += f"  {speedup:>8.1f}×"
            print(row)
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    available = detect_devices()
    default_devs = [d for d in ("cpu", "metal", "cuda") if d in available]

    p = argparse.ArgumentParser(
        description="JAX correctness checks and quick benchmark across devices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--devices", nargs="+",
        choices=["cpu", "metal", "cuda"],
        default=default_devs,
        metavar="DEVICE",
        help="Devices to benchmark. Choices: cpu, metal, cuda.",
    )
    p.add_argument(
        "--repeats", type=int, default=20,
        help="Number of timed repetitions per measurement.",
    )
    p.add_argument(
        "--no-correctness", action="store_true",
        help="Skip correctness checks.",
    )
    p.add_argument(
        "--no-benchmark", action="store_true",
        help="Skip benchmark; run correctness checks only.",
    )
    return p.parse_args(), available


def main():
    args, available = parse_args()

    devices = resolve_devices(args.devices, available)

    print(f"JAX {jax.__version__}  |  " +
          "  |  ".join(f"{n}: {d}" for n, d in devices.items()))
    print()

    if not args.no_correctness:
        run_correctness()

    if not args.no_benchmark:
        run_benchmark(devices, repeats=args.repeats)


if __name__ == "__main__":
    main()
