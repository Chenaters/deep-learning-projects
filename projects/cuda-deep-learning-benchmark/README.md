# CUDA Deep Learning Benchmark

PyTorch CUDA benchmark for matrix multiplication and synthetic convolutional network training, with a CPU comparison and notebook interface.

## Run locally

The original requirements target a CUDA 12.8 PyTorch build. Use an NVIDIA GPU and a compatible driver for GPU measurements.

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python scripts/gpu_benchmark.py --quick
python -m jupyter lab cuda_deep_learning_benchmark.ipynb
```

## Publication and reproducibility

No benchmark results are bundled: hardware and driver diagnostics may identify the original workstation. The script regenerates results locally. Synthetic batches measure compute throughput rather than real dataset loading. The exact original dependency pins are retained.

This is a source-only publication copy prepared on October 26, 2026. Datasets,
trained weights, notebook outputs, local environment files, and Git history are
excluded. Preparation and validation details are in `PUBLICATION_NOTES.md`.
