# RSNA Knee Abnormality Detection — DINOv3 baseline

This repository contains a study-level DINOv3 baseline for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data). It predicts the 12 competition labels from the MRI DICOM series and produces a correctly ordered `submission.csv`.

## Baseline design

Each study can contain several acquisitions in sagittal, coronal, and axial planes. The pipeline:

1. selects up to 6 series with round-robin plane coverage;
2. samples 4 positions from each series;
3. builds a 2.5D image from the previous, current, and next slices;
4. applies study-series percentile windowing, preserves aspect ratio, and uses ImageNet normalization;
5. extracts frozen DINOv3 ViT-S/16 features from every sampled image;
6. adds plane, fluid-sensitive, and fat-suppression embeddings;
7. performs gated-attention pooling across all images in the study;
8. predicts all 12 labels with a masked, class-balanced BCE loss.

Only rows with at least one known target enter supervised training. Missing per-condition labels remain masked rather than being treated as negatives. Reports are deliberately unused because reports are absent from the test set. Cross-validation stratifies both positive labels and label availability. The validation metric is the same macro-averaged ROC-AUC used by the competition.

The backbone is frozen by design. This makes a first run feasible on an 8–16 GB GPU and follows Meta's recommendation to start with frozen DINOv3 features before fine-tuning. Fold checkpoints contain only the lightweight pooling/classification layers; inference also needs the original DINOv3 backbone.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Expected competition layout:

```text
DATA_DIR/
├── train.csv
├── train_series.csv
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
├── test.csv
├── test_series.csv
├── test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
└── sample_submission.csv
```

DINOv3 weights require accepting Meta's DINOv3 license. Request access to `facebook/dinov3-vits16-pretrain-lvd1689m` on Hugging Face, then either authenticate for training or download a local snapshot. For Kaggle's internet-disabled inference, upload that snapshot as a Kaggle Model/Dataset and pass its mounted directory with `--model-path`.

## Train

Start with one fold:

```bash
python train.py \
  --data-dir /path/to/rsna-knee-abnormality-detection \
  --model-path facebook/dinov3-vits16-pretrain-lvd1689m \
  --folds 0 \
  --epochs 8
```

Train the full five-fold ensemble:

```bash
python train.py \
  --data-dir /path/to/rsna-knee-abnormality-detection \
  --model-path /path/to/local/dinov3-vits16-snapshot \
  --folds 0 1 2 3 4
```

Defaults use 24 images per study, batch size 2, and two-step gradient accumulation. If memory is tight, use `--batch-size 1`. The best checkpoint per fold and `folds.csv` are written under `outputs/`.

## Validate the pipeline quickly

Before a full run, point `--data-dir` at a small copy/symlink of the competition hierarchy and use:

```bash
python train.py --data-dir /path/to/data --folds 0 --epochs 1 \
  --max-series 2 --slices-per-series 2 --batch-size 1 --workers 0
```

Run unit tests with:

```bash
pytest -q
```

## Inference and submission

```bash
python infer.py \
  --data-dir /path/to/rsna-knee-abnormality-detection \
  --checkpoints "outputs/dinov3_fold*.pt" \
  --model-path /path/to/local/dinov3-vits16-snapshot \
  --output submission.csv
```

The script averages fold logits, applies sigmoid once, restores the exact `test.csv` order, verifies there are no missing predictions, and writes the required 13 columns.

For a Kaggle submission notebook, attach three inputs: the competition data, this source tree plus trained fold checkpoints, and the local DINOv3 snapshot. With internet disabled, the notebook's final cells can be:

```bash
pip install --no-deps -e /kaggle/input/rsna-dinov3-source
python /kaggle/input/rsna-dinov3-source/infer.py \
  --data-dir /kaggle/input/rsna-knee-abnormality-detection \
  --checkpoints "/kaggle/input/rsna-dinov3-checkpoints/dinov3_fold*.pt" \
  --model-path /kaggle/input/dinov3-vits16-pretrain-lvd1689m \
  --output /kaggle/working/submission.csv
```

Kaggle requires GPU/CPU notebook runtime of at most nine hours, internet disabled, and the output filename `submission.csv`. The ViT-S backbone and capped series/slice sampling are chosen to stay comfortably inside that constraint; actual time should still be measured on the complete hidden-test notebook.

## Practical next improvements

- Pretrain the head with report-derived pseudo-labels, while holding out all studies in the validation fold before pseudo-label generation.
- Cache frozen DINOv3 slice features for faster experiments.
- Add target-specific plane pooling (for example, sagittal emphasis for ACL and menisci).
- Fine-tune only the final DINOv3 block with a very low learning rate after the frozen baseline is stable.
- Ensemble different slice counts or DINOv3 ViT-S and ViT-B backbones.
