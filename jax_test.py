"""JAX correctness checks and CPU vs Metal GPU benchmark on Apple Silicon.

JAX 0.4.34 + jax-metal 0.1.1 required — newer JAX versions have a plugin API
change (default_memory_space) that jax-metal has not yet implemented.
"""

import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# ── Device setup ────────────────────────────────────────────────────────────

CPU   = jax.devices('cpu')[0]
METAL = next((d for d in jax.devices() if 'METAL' in str(d).upper()), None)

print(f"JAX {jax.__version__}  |  cpu: {CPU}  |  metal: {METAL or 'not available'}")
print()

# ── Correctness checks (run on default backend) ──────────────────────────────

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

# ── Benchmark helpers ────────────────────────────────────────────────────────

def bench(fn, *args, repeats=20):
    """Return median wall-clock ms per call after one warmup."""
    fn(*args).block_until_ready()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def make_pair(shape, device):
    k = jax.random.PRNGKey(0)
    x = jax.random.normal(k, shape)
    return jax.device_put(x, device)


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

# ── Run benchmarks ───────────────────────────────────────────────────────────

print("=== CPU vs Metal benchmark (median ms/call) ===")
print(f"{'operation':<12}  {'shape':<14}  {'CPU':>9}  {'Metal':>9}  {'speedup':>9}")
print("-" * 60)

for name, fn, shapes in BENCHMARKS:
    for shape in shapes:
        cpu_ms   = bench(fn, make_pair(shape, CPU))
        label    = f"{shape[0]}×{shape[1]}"
        if METAL:
            metal_ms = bench(fn, make_pair(shape, METAL))
            speedup  = f"{cpu_ms / metal_ms:.1f}×"
        else:
            metal_ms, speedup = float('nan'), "n/a"
        print(f"{name:<12}  {label:<14}  {cpu_ms:>8.2f}  {metal_ms:>8.2f}  {speedup:>9}")
    print()
