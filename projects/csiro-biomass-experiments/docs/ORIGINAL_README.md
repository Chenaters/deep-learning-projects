> Historical preparation guide. The current README and PUBLICATION_NOTES describe the published experimental snapshot.

# CSIRO Biomass Experiments

ResNet/EfficientNet and small DINO biomass regression experiments.

## Run locally

Read the limitations below before execution. Install `requirements.txt` in a fresh environment, obtain the referenced data independently, and open the notebooks with JupyterLab.

## Publication and reproducibility

The original notebooks contain missing data/model errors and a DINO state-dict normalization mismatch. Resolve backbone/checkpoint compatibility, regenerate comparisons, and run from a clean environment. No trained weights or competition data are bundled.

This is a source-only publication copy prepared on October 26, 2026. Datasets,
trained weights, notebook outputs, local environment files, and Git history are
excluded. Preparation and validation details are in `PUBLICATION_NOTES.md`.
