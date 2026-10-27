# CSIRO Biomass: CNN Baselines and DINO Experiments

IMSA AI II coursework investigating image-based prediction of pasture biomass.
This folder contains both the original CNN baseline/exploration work and the
small DINO vision-transformer experiments.

The model predicts five continuous quantities: `Dry_Clover_g`, `Dry_Dead_g`,
`Dry_Green_g`, `Dry_Total_g`, and `GDM_g`.

## Approaches and files

| Approach | Files | Purpose |
| --- | --- | --- |
| Data exploration | [LouisChen_dataExploration.ipynb](LouisChen_dataExploration.ipynb) | Inspect metadata, targets, sample images, and image preprocessing |
| CNN baselines | [LouisChen_baseline_model.ipynb](LouisChen_baseline_model.ipynb) | ResNet50/EfficientNet experiments, fine-tuning, saved-model comparisons, and submission generation |
| Small DINO regression | [dino_model/dino_small_csiro_setup.ipynb](dino_model/dino_small_csiro_setup.ipynb) and its [Python export](dino_model/dino_small_csiro_setup.py) | Pretrained small DINO backbone, MLP regression head, target normalization, and staged fine-tuning |
| DINO validation | [evaluate_dino_validation.py](dino_model/evaluate_dino_validation.py) | Evaluate a compatible saved checkpoint on the configured validation split |
| DINO submission | [make_dino_submission.py](dino_model/make_dino_submission.py) | Load a compatible checkpoint and generate predictions in submission format |

## Setup and data

Use a fresh Python environment and run from this project directory:

```sh
python -m pip install -r requirements.txt
python -m jupyter lab
```

Obtain the [CSIRO Biomass competition data](https://www.kaggle.com/competitions/csiro-biomass)
independently under its applicable terms. The notebooks expect the local
`csiro-biomass/` hierarchy, including `train.csv`, `test.csv`,
`sample_submission.csv`, and images at the paths listed in those CSVs.
Review each notebook's data/configuration cells before execution.

The DINO helpers accept `--dataset-dir` and checkpoint arguments. Read their
available flags with:

```sh
python dino_model/evaluate_dino_validation.py --help
python dino_model/make_dino_submission.py --help
```

The training Python file is a notebook export with top-level execution; importing
it can begin data loading/training. Use its notebook/configuration deliberately.
Dependencies are inferred from imports and are not a tested lockfile.

## Experimental status and reproducibility

This is published as experimental coursework. Source syntax and notebook formats
were checked; complete training and end-to-end prediction were not rerun for this
publication. The original notebooks contain saved missing-data/model errors and
a DINO checkpoint normalization-key mismatch. The cleaned copies have no saved
outputs. Resolve checkpoint/backbone compatibility and regenerate model
comparisons before treating results as reproduced.

Datasets, `.keras`/`.pt` weights, prediction CSVs, figures containing dataset
images, local environments and private audit files are excluded. The separate
historical write-ups below preserve the original coursework discussion and do
not establish new benchmark or leaderboard results.

## Historical coursework write-ups

- [Proposal](docs/coursework/Proposal.md)
- [Data exploration report](docs/coursework/DataExplorationReport.md)
- [Final model discussion](docs/coursework/LouisChen_finalModel.md)

See [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md),
[DATA_SOURCES.md](DATA_SOURCES.md), and [VALIDATION.md](VALIDATION.md).
