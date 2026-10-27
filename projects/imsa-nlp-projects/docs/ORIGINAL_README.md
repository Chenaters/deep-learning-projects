# IMSA NLP Projects

Two IMSA NLP exercises: a spaCy named entity recognizer with a reusable command-line trainer, and an IMDb sentiment classifier using Word2Vec embeddings and bidirectional LSTMs.

## Run locally

Install the source-inferred requirements. Open `conll_ner.ipynb` for a dataset download and train/validation/test example. Run the trainer help to see its CLI:

```sh
python -m pip install -r requirements.txt
python train_simple_ner.py --help
python -m jupyter lab
```

## Publication and reproducibility

The NER example limits each split for a small learning run. CoNLL and IMDb data and trained models are excluded. The IMDb notebook is in `drafts/` because Word2Vec/vocabulary are learned from the full corpus before the split; fit them on training text only before citing held-out results. Keras Embedding arguments and NLTK tokenizer resources also require a fresh-environment run. No historical accuracy is advertised.

This is a source-only publication copy prepared on October 26, 2026. Datasets,
trained weights, notebook outputs, local environment files, and Git history are
excluded. Preparation and validation details are in `PUBLICATION_NOTES.md`.
