# CUDA GPU Benchmark

This folder contains a local CUDA and deep learning benchmark for this machine.

## What It Tests

- NVIDIA driver visibility through `nvidia-smi`
- PyTorch CUDA availability and GPU properties
- Matrix multiply throughput in float32/TF32-enabled mode and float16 mode
- Synthetic ConvNet training throughput on CUDA
- A small CPU training baseline for rough speedup comparison

The benchmark uses synthetic image batches. This keeps the test focused on compute throughput instead of data download, disk, or data-loader performance.

## Files

- `cuda_deep_learning_benchmark.ipynb`: reader-facing notebook
- `scripts/gpu_benchmark.py`: reusable benchmark runner
- `scripts/make_notebook.py`: notebook generator
- `results/benchmark_results.json`: full benchmark output after execution
- `results/training_results.csv`: training result table after execution
- `results/training_throughput.png`: chart after notebook execution

## Run From PowerShell

Use the local virtual environment that was created for this benchmark:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab cuda_deep_learning_benchmark.ipynb
```

Run the notebook top-to-bottom from the Jupyter UI, or execute it from the terminal:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --execute --to notebook --inplace cuda_deep_learning_benchmark.ipynb
```

If a restricted shell blocks Jupyter from writing under your home directory, pin Jupyter state to this folder first:

```powershell
New-Item -ItemType Directory -Force -Path .jupyter, .ipython, .jupyter_data, .jupyter_runtime | Out-Null
$env:IPYTHONDIR = (Resolve-Path .\.ipython).Path
$env:JUPYTER_CONFIG_DIR = (Resolve-Path .\.jupyter).Path
$env:JUPYTER_DATA_DIR = (Resolve-Path .\.jupyter_data).Path
$env:JUPYTER_RUNTIME_DIR = (Resolve-Path .\.jupyter_runtime).Path
```

For a shorter smoke test, run the script directly:

```powershell
.\.venv\Scripts\python.exe scripts\gpu_benchmark.py --quick
```

## Notes

Laptop GPU results vary with AC power, battery mode, thermal state, and background GPU usage. For the cleanest comparison, plug in the laptop, set Windows/NVIDIA power mode to performance, close other GPU-heavy apps, and rerun the notebook.
