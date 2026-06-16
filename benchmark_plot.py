"""Benchmark JAX ops across devices and (optionally) save a chart.

Usage examples
--------------
  # default: cpu + metal (or cuda if detected), all ops, plot saved
  python benchmark_plot.py

  # explicit devices, subset of ops, custom sizes
  python benchmark_plot.py --devices cpu metal --ops matmul elemwise

  # table only, no chart
  python benchmark_plot.py --no-plot

  # CUDA run on a Linux machine
  python benchmark_plot.py --devices cpu cuda --out cuda_results.png
"""

import argparse
import subprocess
import time
import sys

import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Hardware info helpers ─────────────────────────────────────────────────────

import platform

def _sysctl(key, default=None):
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return default


def _proc_cpuinfo_field(field):
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(field):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _nvidia_gpu_name():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
    except Exception:
        return None


def _nvidia_gpu_count():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return len(out)
    except Exception:
        return None


def chip_name():
    # macOS
    name = _sysctl("machdep.cpu.brand_string")
    if name:
        return name
    # Linux: try /proc/cpuinfo, then nvidia-smi, then platform fallback
    return (
        _proc_cpuinfo_field("model name")
        or _nvidia_gpu_name()
        or platform.processor()
        or "Unknown CPU"
    )


def system_info():
    # macOS sysctl paths
    cpu_total = int(_sysctl("hw.physicalcpu", 0) or 0)
    cpu_perf  = int(_sysctl("hw.perflevel0.physicalcpu", 0) or 0)
    cpu_eff   = int(_sysctl("hw.perflevel1.physicalcpu", 0) or 0)

    # Linux fallback for physical CPU count
    if cpu_total == 0:
        try:
            import os
            cpu_total = os.cpu_count() or 0
        except Exception:
            pass

    # Metal GPU cores (macOS only)
    metal_cores = None
    try:
        metal_cores = int(subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True, stderr=subprocess.DEVNULL,
        ).split("Total Number of Cores:")[1].split()[0])
    except Exception:
        pass

    # NVIDIA GPU count (Linux)
    cuda_gpus = _nvidia_gpu_count()

    return {
        "cpu_total":  cpu_total,
        "cpu_perf":   cpu_perf,
        "cpu_eff":    cpu_eff,
        "gpu_cores":  metal_cores,   # Metal cores (macOS) or None
        "cuda_gpus":  cuda_gpus,     # NVIDIA GPU count (Linux) or None
    }


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
    """Validate requested device names against what is actually available."""
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

ALL_OPS = {
    "matmul":    ("Matrix multiply",             op_matmul),
    "elemwise":  ("Element-wise (exp·sin·cos)",  op_elemwise),
    "reduction": ("Reduction (sum x²)",           op_reduction),
}

DEFAULT_SIZES   = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_REPEATS = 20


# ── Benchmark core ────────────────────────────────────────────────────────────

def bench(fn, x, repeats):
    fn(x).block_until_ready()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_benchmarks(ops, devices, sizes, repeats):
    """
    Returns:
        results[op_key][device_name] = list of median-ms values (one per size)
    """
    key = jax.random.PRNGKey(0)
    results = {op_key: {dev: [] for dev in devices} for op_key in ops}

    for op_key, (op_label, fn) in ops.items():
        for n in sizes:
            x = jax.random.normal(key, (n, n))
            row_parts = [f"{op_label}  {n}×{n}"]
            for dev_name, device in devices.items():
                ms = bench(fn, jax.device_put(x, device), repeats)
                results[op_key][dev_name].append(ms)
                row_parts.append(f"{dev_name}={ms:.2f}ms")
            print("  " + "  ".join(row_parts))

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "cpu":    "#4C72B0",
    "metal":  "#DD8452",
    "cuda":   "#55A868",
}

MARKERS = {
    "cpu":   "o",
    "metal": "s",
    "cuda":  "^",
}


def device_label(dev_name, info):
    if dev_name == "cpu":
        if info["cpu_perf"]:
            return f"CPU ({info['cpu_perf']}P+{info['cpu_eff']}E cores)"
        return f"CPU ({info['cpu_total']} cores)"
    if dev_name == "metal":
        return f"Metal GPU ({info['gpu_cores']} cores)" if info["gpu_cores"] else "Metal GPU"
    if dev_name == "cuda":
        gpu_name = _nvidia_gpu_name()
        return f"CUDA GPU ({gpu_name})" if gpu_name else "CUDA GPU"
    return dev_name.upper()


def _hw_footer(info):
    parts = []
    if info["cpu_perf"]:
        parts.append(f"CPU: {info['cpu_perf']}P + {info['cpu_eff']}E cores ({info['cpu_total']} total)")
    elif info["cpu_total"]:
        parts.append(f"CPU: {info['cpu_total']} cores")
    if info["gpu_cores"]:
        parts.append(f"GPU: {info['gpu_cores']} Metal cores")
    if info["cuda_gpus"]:
        gpu_name = _nvidia_gpu_name() or "NVIDIA"
        parts.append(f"GPU: {info['cuda_gpus']}× {gpu_name}")
    parts.append(f"JAX {jax.__version__}")
    return "   ".join(parts)


def plot(results, ops, devices, sizes, out):
    info = system_info()
    n_ops = len(ops)
    fig, axes = plt.subplots(1, n_ops, figsize=(5 * n_ops, 4.5), sharey=False)
    if n_ops == 1:
        axes = [axes]

    fig.suptitle(
        f"JAX: {' vs '.join(d.upper() for d in devices)} — {chip_name()}\n"
        f"(median ms/call, lower is better)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (op_key, (op_label, _)) in zip(axes, ops.items()):
        for dev_name in devices:
            ms_list = results[op_key][dev_name]
            ax.plot(
                sizes, ms_list,
                f"{MARKERS.get(dev_name, 'o')}-",
                color=COLORS.get(dev_name, "#888888"),
                label=device_label(dev_name, info),
                linewidth=2,
            )
        ax.set_title(op_label, fontsize=11)
        ax.set_xlabel("Matrix size (N×N)")
        ax.set_ylabel("Median time (ms)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xticks(sizes)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    hw_text = _hw_footer(info)
    fig.text(0.5, -0.02, hw_text, ha="center", fontsize=9, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc"))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    available = detect_devices()
    default_devs = [d for d in ("cpu", "metal", "cuda") if d in available]

    p = argparse.ArgumentParser(
        description="Benchmark JAX ops across CPU / Metal / CUDA devices.",
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
        "--ops", nargs="+",
        choices=list(ALL_OPS),
        default=list(ALL_OPS),
        metavar="OP",
        help=f"Ops to run. Choices: {', '.join(ALL_OPS)}.",
    )
    p.add_argument(
        "--sizes", nargs="+", type=int,
        default=DEFAULT_SIZES,
        metavar="N",
        help="Matrix sizes (N×N) to sweep.",
    )
    p.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="Number of timed repetitions per measurement.",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip chart generation; print table only.",
    )
    p.add_argument(
        "--out", default="benchmark.png",
        help="Output filename for the chart.",
    )
    return p.parse_args(), available


def main():
    args, available = parse_args()

    devices = resolve_devices(args.devices, available)
    ops     = {k: ALL_OPS[k] for k in args.ops}

    info = system_info()
    print(f"JAX {jax.__version__}  |  {chip_name()}")
    print(_hw_footer(info))
    print(f"Devices : {', '.join(devices)}")
    print(f"Ops     : {', '.join(ops)}")
    print(f"Sizes   : {args.sizes}")
    print(f"Repeats : {args.repeats}\n")

    results = run_benchmarks(ops, devices, args.sizes, args.repeats)

    if not args.no_plot:
        plot(results, ops, devices, args.sizes, args.out)


if __name__ == "__main__":
    main()
