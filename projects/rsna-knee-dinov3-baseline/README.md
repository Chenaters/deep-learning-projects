# RSNA Knee MRI DINOv3 Baseline

A source baseline that combines frozen DINOv3 image features with study-level attention pooling to predict twelve MRI labels. Includes unit tests with a dummy backbone.

## Run locally

Use Python 3.11 or 3.12. Installation, data hierarchy, training and inference commands are in [the original project guide](docs/ORIGINAL_README.md).

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## Publication and reproducibility

No patient data, DICOM files, model weights, fold predictions or competition submissions are included. Acquire competition data and backbone weights separately under their respective access terms. The source package does not establish competition performance; full training and end-to-end inference were not rerun. Competition-specific requirements in the original guide have not been reverified.

This is a source-only publication copy prepared on October 26, 2026. Datasets,
trained weights, notebook outputs, local environment files, and Git history are
excluded. Preparation and validation details are in `PUBLICATION_NOTES.md`.
