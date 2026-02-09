from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class TrainResult:
    model: Pipeline
    report: str


def build_pipeline(*, min_df: int = 2) -> Pipeline:
    """
    Strong baseline for short noisy strings:
    - char ngrams handle OCR/abbrev better than word ngrams
    """
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        lowercase=True,
    )
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,  # keep compatible across sklearn versions
        class_weight="balanced",
    )
    return Pipeline([("tfidf", vec), ("clf", clf)])


def train_from_csv(
    csv_path: Path,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{csv_path} must have columns: {text_col}, {label_col}")

    df = df[[text_col, label_col]].dropna()
    X = df[text_col].astype(str).tolist()
    y = df[label_col].astype(str).tolist()

    # Stratified splits are ideal, but they fail when some classes are very small.
    # For tiny/sample datasets we fall back to a non-stratified split, or skip the
    # holdout split entirely.

    n = len(X)
    unique = len(set(y))
    counts = pd.Series(y).value_counts(dropna=False)
    can_stratify = unique >= 2 and int(counts.min()) >= 2

    # For very small datasets, a holdout test set is not meaningful.
    # Train on all data and return a note.
    min_test = int(round(n * test_size))
    if n < 10 or unique < 2 or min_test < 1:
        model = build_pipeline(min_df=1)
        model.fit(X, y)
        return TrainResult(
            model=model,
            report=(
                "Not enough data for a holdout test split; trained on all rows.\n"
                f"rows={n} unique_labels={unique}"
            ),
        )

    strat = y if can_stratify else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    except ValueError:
        # As a last resort, do a plain split.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    # If the dataset is small, min_df=2 can wipe out the vocabulary.
    model = build_pipeline(min_df=1 if n < 50 else 2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=2, zero_division=0)

    if not can_stratify:
        report = "WARNING: non-stratified split (some classes have <2 samples).\n" + report

    return TrainResult(model=model, report=report)


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def _get_class_labels(model: Pipeline) -> list[str]:
    # Pipeline exposes estimator attributes through the final step.
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]
    if (
        hasattr(model, "named_steps")
        and "clf" in model.named_steps
        and hasattr(model.named_steps["clf"], "classes_")
    ):
        return [str(c) for c in model.named_steps["clf"].classes_]
    raise ValueError("Model is missing classes_ attribute; is it fitted?")


def predict_topk(model: Pipeline, text: str, k: int = 3) -> list[tuple[str, float]]:
    """
    Returns top-k (label, probability) sorted descending.
    Requires a probabilistic classifier (predict_proba).
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba; cannot compute top-k.")

    probs = model.predict_proba([text])[0]
    labels = _get_class_labels(model)

    k = max(1, min(int(k), len(labels)))
    # indices of top-k probabilities
    top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    return [(labels[i], float(probs[i])) for i in top_idx]


def predict_one(model: Pipeline, text: str) -> tuple[str, float]:
    """
    Returns (predicted_label, confidence).
    Confidence is max predicted probability. Works for probabilistic models.
    """
    top1 = predict_topk(model, text, k=1)[0]
    return top1[0], top1[1]
