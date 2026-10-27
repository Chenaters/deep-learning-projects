from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "cuda_deep_learning_benchmark.ipynb"
RESULTS_JSON = ROOT / "results" / "benchmark_results.json"


def load_summary_markdown() -> str:
    if not RESULTS_JSON.exists():
        return (
            "# CUDA GPU Deep Learning Benchmark\n\n"
            "## tl;dr\n\n"
            "Run this notebook top-to-bottom to capture CUDA availability, matrix multiply throughput, "
            "and synthetic ConvNet training throughput on this machine.\n\n"
            "## Context & Methods\n\n"
            "This notebook uses synthetic image batches so the timing focuses on GPU compute rather than "
            "network downloads, disk reads, or data preprocessing. The training benchmark runs a compact "
            "convolutional network in both float32/TF32-enabled mode and CUDA AMP float16 mode, then compares "
            "the best GPU result with a small CPU baseline."
        )

    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    env = payload.get("environment", {})
    gpu = env.get("gpu") or {}
    best_training = summary.get("best_cuda_training") or {}
    best_cpu = summary.get("best_cpu_training") or {}
    best_matrix = summary.get("best_matrix_multiply") or {}
    speedup = summary.get("training_speedup_vs_cpu")

    lines = [
        "# CUDA GPU Deep Learning Benchmark",
        "",
        "## tl;dr",
        "",
        f"- GPU tested: {gpu.get('name', 'CUDA unavailable')} with {gpu.get('total_memory_gb', 'unknown')} GB VRAM.",
        f"- PyTorch saw CUDA successfully: {env.get('torch_cuda_available')} "
        f"(torch {env.get('torch_version')}, CUDA wheel {env.get('torch_cuda_version')}).",
    ]

    if best_training:
        lines.append(
            "- Best synthetic ConvNet training result: "
            f"{best_training['images_per_second']:.1f} images/sec "
            f"({best_training['ms_per_step']:.1f} ms/step), "
            f"batch {best_training['batch_size']}, {best_training['precision']}."
        )
    if best_cpu and speedup:
        lines.append(
            f"- Best GPU training throughput was {speedup:.1f}x the small CPU baseline "
            f"({best_cpu['images_per_second']:.1f} images/sec)."
        )
    if best_matrix:
        lines.append(
            "- Best matrix multiply result: "
            f"{best_matrix['tflops']:.2f} TFLOP/s "
            f"({best_matrix['dtype']}, size {best_matrix['matrix_size']})."
        )

    lines.extend(
        [
            "",
            "## Context & Methods",
            "",
            "This notebook uses synthetic image batches so the timing focuses on GPU compute rather than "
            "network downloads, disk reads, or data preprocessing. The training benchmark runs a compact "
            "convolutional network in both float32/TF32-enabled mode and CUDA AMP float16 mode, then compares "
            "the best GPU result with a small CPU baseline.",
            "",
            "### Key Assumptions",
            "",
            "- Synthetic data measures training compute throughput, not end-to-end data-loader performance.",
            "- Laptop GPU performance depends strongly on power mode, thermals, and whether the machine is plugged in.",
            "- The CUDA wheel is cu128; the installed NVIDIA driver advertises CUDA 12.9 and is backward-compatible with this wheel.",
        ]
    )
    return "\n".join(lines)


def make_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python (cuda_gpu_benchmark)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    cells = [
        nbf.v4.new_markdown_cell(load_summary_markdown()),
        nbf.v4.new_markdown_cell("## Setup"),
        nbf.v4.new_code_cell(
            """from __future__ import annotations

import json
import sys
from pathlib import Path

from IPython.display import Markdown, display

ROOT = Path.cwd()
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gpu_benchmark

CONFIG = gpu_benchmark.merge_config()
CONFIG"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """environment = gpu_benchmark.collect_environment()

compact_environment = {
    key: value
    for key, value in environment.items()
    if key != "nvidia_smi"
}

print(json.dumps(compact_environment, indent=2))
print("\\n--- nvidia-smi ---")
print(environment["nvidia_smi"])"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """payload = gpu_benchmark.run_all_benchmarks(config=CONFIG)
gpu_benchmark.print_summary(payload)"""
        ),
        nbf.v4.new_code_cell(
            """def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\\n".join([header, divider, *body])


training_columns = [
    "device",
    "precision",
    "batch_size",
    "timed_steps",
    "ms_per_step",
    "images_per_second",
    "peak_memory_gb",
    "error",
]

display(Markdown("### Training Throughput"))
display(Markdown(markdown_table(payload["training_results"], training_columns)))"""
        ),
        nbf.v4.new_code_cell(
            """matrix_columns = [
    "dtype",
    "matrix_size",
    "repeats",
    "elapsed_seconds",
    "tflops",
    "error",
]

display(Markdown("### Matrix Multiply Throughput"))
display(Markdown(markdown_table(payload["matrix_results"], matrix_columns)))"""
        ),
        nbf.v4.new_code_cell(
            """import matplotlib.pyplot as plt

plot_rows = [
    row
    for row in payload["training_results"]
    if row.get("device") == "cuda" and "images_per_second" in row
]

if plot_rows:
    labels = [f"{row['precision']}\\nbatch {row['batch_size']}" for row in plot_rows]
    values = [row["images_per_second"] for row in plot_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color=["#3772ff" if "amp" in label else "#2d936c" for label in labels])
    ax.set_title("Synthetic ConvNet Training Throughput")
    ax.set_ylabel("Images per second")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    chart_path = RESULTS_DIR / "training_throughput.png"
    fig.savefig(chart_path, dpi=150)
    plt.show()
    print(f"Saved chart to {chart_path}")
else:
    print("No CUDA training rows were available to plot.")"""
        ),
        nbf.v4.new_markdown_cell("## Takeaways"),
        nbf.v4.new_code_cell(
            """summary = payload["summary"]
best_training = summary.get("best_cuda_training")
best_cpu = summary.get("best_cpu_training")
best_matrix = summary.get("best_matrix_multiply")
speedup = summary.get("training_speedup_vs_cpu")

takeaways = []
if best_training:
    takeaways.append(
        f"- Best GPU training result was **{best_training['images_per_second']:.1f} images/sec** "
        f"at batch {best_training['batch_size']} using `{best_training['precision']}`."
    )
if best_cpu and speedup:
    takeaways.append(
        f"- That was **{speedup:.1f}x faster** than the CPU baseline in this notebook "
        f"({best_cpu['images_per_second']:.1f} images/sec)."
    )
if best_matrix:
    takeaways.append(
        f"- Peak matrix multiply throughput was **{best_matrix['tflops']:.2f} TFLOP/s** "
        f"for `{best_matrix['dtype']}` at size {best_matrix['matrix_size']}."
    )
if not takeaways:
    takeaways.append("- CUDA results were not produced; check the environment output above.")

display(Markdown("\\n".join(takeaways)))"""
        ),
    ]

    notebook["cells"] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    make_notebook()
