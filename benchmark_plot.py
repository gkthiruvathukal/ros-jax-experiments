"""Benchmark CPU vs Metal across matrix sizes and save a chart."""

import subprocess
import time
import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _sysctl(key, default=None):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return default


def chip_name():
    return _sysctl("machdep.cpu.brand_string", "Apple Silicon")


def system_info():
    cpu_total = int(_sysctl("hw.physicalcpu", 0))
    cpu_perf  = int(_sysctl("hw.perflevel0.physicalcpu", 0))
    cpu_eff   = int(_sysctl("hw.perflevel1.physicalcpu", 0))
    try:
        gpu_cores = int(subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True
        ).split("Total Number of Cores:")[1].split()[0])
    except Exception:
        gpu_cores = None

    return {
        "cpu_total": cpu_total,
        "cpu_perf":  cpu_perf,
        "cpu_eff":   cpu_eff,
        "gpu_cores": gpu_cores,
    }

CPU   = jax.devices('cpu')[0]
METAL = next((d for d in jax.devices() if 'METAL' in str(d).upper()), None)

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
REPEATS = 20

@jit
def op_matmul(a):    return jnp.dot(a, a)

@jit
def op_elemwise(a):  return jnp.exp(jnp.sin(a) + jnp.cos(a))

@jit
def op_reduction(a): return jnp.sum(a ** 2)

OPS = [
    ("Matrix multiply", op_matmul),
    ("Element-wise (exp·sin·cos)", op_elemwise),
    ("Reduction (sum x²)", op_reduction),
]


def bench(fn, x):
    fn(x).block_until_ready()          # warmup / compile
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(x).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_benchmarks():
    key = jax.random.PRNGKey(0)
    results = {}
    for op_name, fn in OPS:
        cpu_ms, metal_ms = [], []
        for n in SIZES:
            x = jax.random.normal(key, (n, n))
            cpu_ms.append(bench(fn, jax.device_put(x, CPU)))
            if METAL:
                metal_ms.append(bench(fn, jax.device_put(x, METAL)))
            print(f"  {op_name}  {n}×{n}  cpu={cpu_ms[-1]:.2f}ms" +
                  (f"  metal={metal_ms[-1]:.2f}ms" if METAL else ""))
        results[op_name] = (cpu_ms, metal_ms if METAL else None)
    return results


def plot(results, out="benchmark.png"):
    info = system_info()
    cpu_label = (f"CPU ({info['cpu_perf']}P+{info['cpu_eff']}E cores)"
                 if info['cpu_perf'] else f"CPU ({info['cpu_total']} cores)")
    gpu_label = (f"Metal GPU ({info['gpu_cores']} cores)"
                 if info['gpu_cores'] else "Metal GPU")

    n_ops = len(OPS)
    fig, axes = plt.subplots(1, n_ops, figsize=(5 * n_ops, 4.5), sharey=False)
    fig.suptitle(f"JAX: CPU vs Metal GPU — {chip_name()}\n(median ms/call, lower is better)",
                 fontsize=13, fontweight="bold", y=1.02)

    colors = {"CPU": "#4C72B0", "Metal": "#DD8452"}

    for ax, (op_name, fn) in zip(axes, OPS):
        cpu_ms, metal_ms = results[op_name]
        ax.plot(SIZES, cpu_ms,   "o-", color=colors["CPU"],   label=cpu_label, linewidth=2)
        if metal_ms:
            ax.plot(SIZES, metal_ms, "s-", color=colors["Metal"], label=gpu_label, linewidth=2)
        ax.set_title(op_name, fontsize=11)
        ax.set_xlabel("Matrix size (N×N)")
        ax.set_ylabel("Median time (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xticks(SIZES)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    hw_text = (
        f"CPU: {info['cpu_perf']}P + {info['cpu_eff']}E cores ({info['cpu_total']} total)   "
        f"GPU: {info['gpu_cores']} Metal cores   "
        f"JAX {jax.__version__}"
    )
    fig.text(0.5, -0.02, hw_text, ha="center", fontsize=9, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc"))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    info = system_info()
    print(f"JAX {jax.__version__}  |  {chip_name()}")
    print(f"CPU: {info['cpu_perf']}P + {info['cpu_eff']}E cores ({info['cpu_total']} total)  |  "
          f"GPU: {info['gpu_cores']} Metal cores\n")
    results = run_benchmarks()
    plot(results)
