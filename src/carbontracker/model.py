from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


@dataclass
class TrainResult:
    model: Pipeline
    report: str


def build_pipeline() -> Pipeline:
    """
    Strong baseline for short noisy strings:
    - char ngrams handle OCR/abbrev better than word ngrams
    """
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=2)

    return TrainResult(model=model, report=report)


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def predict_one(model: Pipeline, text: str) -> Tuple[str, float]:
    """
    Returns (predicted_label, confidence).
    Confidence is max predicted probability. Works for probabilistic models.
    """
    label = model.predict([text])[0]
    conf = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        conf = float(max(proba))
    return label, conf
