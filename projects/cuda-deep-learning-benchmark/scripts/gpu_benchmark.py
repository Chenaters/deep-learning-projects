from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"
RESULTS_CSV = RESULTS_DIR / "training_results.csv"


DEFAULT_CONFIG = {
    "seed": 20260626,
    "input_size": 128,
    "num_classes": 100,
    "gpu_batch_sizes": [16, 32, 64, 96],
    "cpu_batch_sizes": [8],
    "warmup_steps": 5,
    "timed_steps": 25,
    "cpu_warmup_steps": 2,
    "cpu_timed_steps": 8,
    "matrix_sizes": [1024, 2048, 4096],
    "matrix_repeats": 20,
    "matrix_warmup": 5,
}


QUICK_CONFIG_OVERRIDES = {
    "gpu_batch_sizes": [16, 32, 64],
    "warmup_steps": 3,
    "timed_steps": 12,
    "cpu_timed_steps": 4,
    "matrix_sizes": [1024, 2048],
    "matrix_repeats": 10,
    "matrix_warmup": 3,
}


def merge_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def run_nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"
    return (completed.stdout or completed.stderr).strip()


def collect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_info: dict[str, Any] | None = None

    if cuda_available:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu_info = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "system_memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": gpu_info,
        "nvidia_smi": run_nvidia_smi(),
    }


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or isinstance(
        exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SyntheticConvNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def enable_fast_cuda_defaults(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def matrix_multiply_benchmark(
    device: torch.device,
    sizes: list[int],
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    if device.type != "cuda":
        return []

    results: list[dict[str, Any]] = []
    dtype_specs = [
        ("float32_tf32_allowed", torch.float32),
        ("float16", torch.float16),
    ]

    for dtype_name, dtype in dtype_specs:
        for size in sizes:
            try:
                torch.cuda.empty_cache()
                a = torch.randn((size, size), device=device, dtype=dtype)
                b = torch.randn((size, size), device=device, dtype=dtype)
                for _ in range(warmup):
                    _ = a @ b
                sync_if_needed(device)

                start = time.perf_counter()
                for _ in range(repeats):
                    _ = a @ b
                sync_if_needed(device)
                elapsed = time.perf_counter() - start

                operations = 2 * size**3 * repeats
                results.append(
                    {
                        "test": "matrix_multiply",
                        "device": device.type,
                        "dtype": dtype_name,
                        "matrix_size": size,
                        "repeats": repeats,
                        "elapsed_seconds": elapsed,
                        "tflops": operations / elapsed / 1e12,
                    }
                )
            except RuntimeError as exc:
                if is_oom_error(exc):
                    results.append(
                        {
                            "test": "matrix_multiply",
                            "device": device.type,
                            "dtype": dtype_name,
                            "matrix_size": size,
                            "repeats": repeats,
                            "error": "out_of_memory",
                        }
                    )
                    torch.cuda.empty_cache()
                    continue
                raise
    return results


def training_benchmark(
    device: torch.device,
    batch_size: int,
    input_size: int,
    num_classes: int,
    warmup_steps: int,
    timed_steps: int,
    precision: str,
) -> dict[str, Any]:
    use_amp = device.type == "cuda" and precision == "amp_float16"
    model = SyntheticConvNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(batch_size, 3, input_size, input_size, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def run_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, targets)
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    for _ in range(warmup_steps):
        run_step()
    sync_if_needed(device)

    start = time.perf_counter()
    last_loss = 0.0
    for _ in range(timed_steps):
        last_loss = run_step()
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    peak_memory_gb = None
    if device.type == "cuda":
        peak_memory_gb = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)

    return {
        "test": "synthetic_convnet_training",
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": precision,
        "batch_size": batch_size,
        "input_size": input_size,
        "num_classes": num_classes,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "elapsed_seconds": elapsed,
        "ms_per_step": elapsed * 1000 / timed_steps,
        "images_per_second": batch_size * timed_steps / elapsed,
        "final_loss": last_loss,
        "peak_memory_gb": peak_memory_gb,
    }


def run_training_suite(config: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
        for precision in ("float32_tf32_allowed", "amp_float16"):
            for batch_size in config["gpu_batch_sizes"]:
                try:
                    results.append(
                        training_benchmark(
                            device=device,
                            batch_size=batch_size,
                            input_size=config["input_size"],
                            num_classes=config["num_classes"],
                            warmup_steps=config["warmup_steps"],
                            timed_steps=config["timed_steps"],
                            precision=precision,
                        )
                    )
                except RuntimeError as exc:
                    if is_oom_error(exc):
                        results.append(
                            {
                                "test": "synthetic_convnet_training",
                                "device": "cuda",
                                "precision": precision,
                                "batch_size": batch_size,
                                "error": "out_of_memory",
                            }
                        )
                        torch.cuda.empty_cache()
                        continue
                    raise

    cpu_device = torch.device("cpu")
    for batch_size in config["cpu_batch_sizes"]:
        results.append(
            training_benchmark(
                device=cpu_device,
                batch_size=batch_size,
                input_size=config["input_size"],
                num_classes=config["num_classes"],
                warmup_steps=config["cpu_warmup_steps"],
                timed_steps=config["cpu_timed_steps"],
                precision="float32",
            )
        )

    return results


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    training_rows = [
        row
        for row in payload["training_results"]
        if "images_per_second" in row and row.get("device") == "cuda"
    ]
    cpu_rows = [
        row
        for row in payload["training_results"]
        if "images_per_second" in row and row.get("device") == "cpu"
    ]
    matrix_rows = [row for row in payload["matrix_results"] if "tflops" in row]

    best_training = max(training_rows, key=lambda row: row["images_per_second"], default=None)
    best_cpu = max(cpu_rows, key=lambda row: row["images_per_second"], default=None)
    best_matrix = max(matrix_rows, key=lambda row: row["tflops"], default=None)

    speedup_vs_cpu = None
    if best_training and best_cpu:
        speedup_vs_cpu = best_training["images_per_second"] / best_cpu["images_per_second"]

    return {
        "best_cuda_training": best_training,
        "best_cpu_training": best_cpu,
        "training_speedup_vs_cpu": speedup_vs_cpu,
        "best_matrix_multiply": best_matrix,
    }


def save_payload(payload: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload["summary"] = summarize_payload(payload)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "test",
        "device",
        "device_name",
        "precision",
        "batch_size",
        "input_size",
        "timed_steps",
        "elapsed_seconds",
        "ms_per_step",
        "images_per_second",
        "peak_memory_gb",
        "error",
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["training_results"])


def run_all_benchmarks(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = merge_config(config)
    enable_fast_cuda_defaults(config["seed"])

    environment = collect_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_results = matrix_multiply_benchmark(
        device=device,
        sizes=config["matrix_sizes"],
        repeats=config["matrix_repeats"],
        warmup=config["matrix_warmup"],
    )
    training_results = run_training_suite(config)

    payload = {
        "config": config,
        "environment": environment,
        "matrix_results": matrix_results,
        "training_results": training_results,
    }
    save_payload(payload)
    return payload


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    env = payload["environment"]
    gpu = env.get("gpu") or {}
    print(f"GPU: {gpu.get('name', 'CUDA unavailable')}")
    print(f"PyTorch: {env['torch_version']} | CUDA wheel: {env['torch_cuda_version']}")

    best_training = summary.get("best_cuda_training")
    if best_training:
        print(
            "Best CUDA training: "
            f"{best_training['images_per_second']:.1f} images/sec, "
            f"{best_training['ms_per_step']:.1f} ms/step, "
            f"batch {best_training['batch_size']}, "
            f"{best_training['precision']}"
        )

    best_cpu = summary.get("best_cpu_training")
    speedup = summary.get("training_speedup_vs_cpu")
    if best_cpu and speedup:
        print(
            "CPU baseline: "
            f"{best_cpu['images_per_second']:.1f} images/sec; "
            f"best CUDA was {speedup:.1f}x faster."
        )

    best_matrix = summary.get("best_matrix_multiply")
    if best_matrix:
        print(
            "Best matrix multiply: "
            f"{best_matrix['tflops']:.2f} TFLOP/s "
            f"({best_matrix['dtype']}, size {best_matrix['matrix_size']})."
        )

    print(f"Results JSON: {RESULTS_JSON}")
    print(f"Training CSV: {RESULTS_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CUDA deep learning benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run a shorter smoke-test benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = QUICK_CONFIG_OVERRIDES if args.quick else None
    payload = run_all_benchmarks(config=overrides)
    print_summary(payload)


if __name__ == "__main__":
    main()
