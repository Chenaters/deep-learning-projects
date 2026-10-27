import argparse
import random
from pathlib import Path

import spacy
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch


def parse_conll_text(raw_text):
    """Parse CoNLL text into sentence-level (tokens, tags)."""
    sentences = []
    tokens = []
    ner_tags = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            if tokens:
                sentences.append((tokens, ner_tags))
                tokens, ner_tags = [], []
            continue

        if line.startswith("-DOCSTART-"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        tokens.append(parts[0])
        ner_tags.append(parts[-1])

    if tokens:
        sentences.append((tokens, ner_tags))

    return sentences


def bio_to_spans(tags):
    """Convert BIO tags to token index spans: (start_tok, end_tok, label)."""
    spans = []
    start = None
    label = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                spans.append((start, i - 1, label))
                start, label = None, None
            continue

        prefix, ent = tag.split("-", 1)
        if prefix == "B" or (prefix == "I" and (start is None or ent != label)):
            if start is not None:
                spans.append((start, i - 1, label))
            start, label = i, ent

    if start is not None:
        spans.append((start, len(tags) - 1, label))

    return spans


def sentence_to_text_offsets(tokens):
    """Build sentence text and token-level character offsets."""
    text = ""
    offsets = []

    for token in tokens:
        if text:
            text += " "
        start = len(text)
        text += token
        end = len(text)
        offsets.append((start, end))

    return text, offsets


def build_examples(nlp, sentences):
    examples = []

    for tokens, tags in sentences:
        text, offsets = sentence_to_text_offsets(tokens)
        ents = []
        for start_tok, end_tok, label in bio_to_spans(tags):
            start_char = offsets[start_tok][0]
            end_char = offsets[end_tok][1]
            ents.append((start_char, end_char, label))

        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, {"entities": ents}))

    return examples


def evaluate_split(nlp, examples):
    pred_examples = [Example(nlp(ex.reference.text), ex.reference) for ex in examples]
    scores = Scorer().score(pred_examples)
    return {
        "precision": scores["ents_p"],
        "recall": scores["ents_r"],
        "f1": scores["ents_f"],
    }


def train_simple_ner(
    data_train,
    data_valid,
    data_test,
    n_epochs=6,
    max_train_sentences=3000,
    max_valid_sentences=600,
    max_test_sentences=600,
    batch_size=8,
    dropout=0.25,
    seed=42,
):
    fix_random_seed(seed)
    random.seed(seed)

    train_sents = parse_conll_text(data_train)[:max_train_sentences]
    valid_sents = parse_conll_text(data_valid)[:max_valid_sentences]
    test_sents = parse_conll_text(data_test)[:max_test_sentences]

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    for _, tags in train_sents:
        for tag in tags:
            if tag != "O":
                ner.add_label(tag.split("-", 1)[1])

    train_examples = build_examples(nlp, train_sents)
    valid_examples = build_examples(nlp, valid_sents)
    test_examples = build_examples(nlp, test_sents)

    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    print(f"spaCy version: {spacy.__version__}")
    print(f"Labels: {sorted(ner.labels)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Valid examples: {len(valid_examples)}")
    print(f"Test examples:  {len(test_examples)}")

    for epoch in range(1, n_epochs + 1):
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=batch_size):
            nlp.update(batch, sgd=optimizer, losses=losses, drop=dropout)
        print(f"Epoch {epoch} | NER loss: {losses.get('ner', 0.0):.3f}")

    valid_metrics = evaluate_split(nlp, valid_examples)
    test_metrics = evaluate_split(nlp, test_examples)

    return nlp, valid_metrics, test_metrics


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def resolve_default_paths():
    cwd = Path(".")
    direct = {
        "train": cwd / "train.txt",
        "valid": cwd / "valid.txt",
        "test": cwd / "test.txt",
    }
    if all(p.exists() for p in direct.values()):
        return direct

    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "alaakhaled" / "conll003-englishversion" / "versions" / "1"
    cached = {
        "train": cache_root / "train.txt",
        "valid": cache_root / "valid.txt",
        "test": cache_root / "test.txt",
    }
    return cached


def main():
    defaults = resolve_default_paths()

    parser = argparse.ArgumentParser(description="Train a simple spaCy NER model from CoNLL text data.")
    parser.add_argument("--train-path", type=Path, default=defaults["train"])
    parser.add_argument("--valid-path", type=Path, default=defaults["valid"])
    parser.add_argument("--test-path", type=Path, default=defaults["test"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-valid", type=int, default=600)
    parser.add_argument("--max-test", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("simple_conll_ner_model_v2"))
    args = parser.parse_args()

    for path in [args.train_path, args.valid_path, args.test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    data_train = read_text(args.train_path)
    data_valid = read_text(args.valid_path)
    data_test = read_text(args.test_path)

    nlp, valid_metrics, test_metrics = train_simple_ner(
        data_train=data_train,
        data_valid=data_valid,
        data_test=data_test,
        n_epochs=args.epochs,
        max_train_sentences=args.max_train,
        max_valid_sentences=args.max_valid,
        max_test_sentences=args.max_test,
        batch_size=args.batch_size,
        dropout=args.dropout,
        seed=args.seed,
    )

    print("\nValidation metrics")
    print(f"Precision: {valid_metrics['precision']:.4f}")
    print(f"Recall:    {valid_metrics['recall']:.4f}")
    print(f"F1:        {valid_metrics['f1']:.4f}")

    print("\nTest metrics")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1:        {test_metrics['f1']:.4f}")

    nlp.to_disk(args.output_dir)
    print(f"\nSaved model to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
