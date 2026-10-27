# IMSA Named Entity Recognition

IMSA AI II coursework implementing a spaCy named entity recognizer on CoNLL-style
BIO annotations. Includes sentence parsing, BIO-to-span conversion, model
training, validation/test precision/recall/F1, and model export.

## Run

From this project folder in a fresh Python environment:

```sh
python -m pip install -r requirements.txt
python train_simple_ner.py --help
python -m jupyter lab conll_ner.ipynb
```

The notebook obtains its referenced dataset through KaggleHub. The script accepts
explicit `--train-path`, `--valid-path`, and `--test-path` arguments and otherwise
looks for local files or the KaggleHub cache. Obtain data independently under its
applicable terms. Datasets and trained model directories are excluded.

## Scope and validation

This is a learning exercise, not a new NER method. The notebook uses capped
split sizes for a small experiment. Source syntax and notebook format were
checked; the complete training run was not repeated during publication.
The separate IMDb sentiment experiment remains unpublished while its
embedding/data-split leakage is being corrected.
