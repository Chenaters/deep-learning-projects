# IMSA Deep Learning Labs

A collection of IMSA AI II learning exercises covering PyTorch fundamentals, convolutional classification, reconstruction and transfer learning. These are coursework experiments, not claims of novel methods.

## Run locally

Use a fresh Python environment, install requirements, then open the notebooks and run from the top:

```sh
python -m pip install -r requirements.txt
python -m jupyter lab
```

Fashion-MNIST and ImageNet backbone weights are downloaded by the corresponding framework calls.

## Publication and reproducibility

Saved outputs were cleared. The publication copy fixes ResNet50 preprocessing to use its expected input transform; previous discussion describes the historical experiment and requires rerunning. GPU training, downloads, and full notebook execution were not performed during preparation. Course-provided prompts remain credited as IMSA coursework. No dataset images are included.

This is a source-only publication copy prepared on October 26, 2026. Datasets,
trained weights, notebook outputs, local environment files, and Git history are
excluded. Preparation and validation details are in `PUBLICATION_NOTES.md`.
