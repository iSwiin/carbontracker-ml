from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from carbontracker.config import Paths


@dataclass
class EvalRun:
    timestamp_utc: str
    data_path: str
    model_path: str
    test_size: float
    random_state: int
    stratify: bool
    threshold_unknown: float | None
    n_total: int
    n_test: int
    labels: list[str]
    accuracy: float
    f1_macro: float
    f1_weighted: float
    data_sha256: str
    model_sha256: str
    per_class: dict[str, dict[str, float]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_train_test_split(
    X: pd.Series,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    """
    Stratified split is ideal, but it can fail if some classes have too few samples.
    This falls back to non-stratified split when needed.
    """
    if stratify:
        try:
            return train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            ) + (True,)
        except Exception:
            pass

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )
    return X_train, X_test, y_train, y_test, False


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained receipt item classifier.")
    parser.add_argument("--data", default=None, help="Path to items.csv (default: from Paths)")
    parser.add_argument(
        "--model", default=None, help="Path to saved model .joblib (default: from Paths)"
    )
    parser.add_argument("--outdir", default="reports", help="Output dir for reports")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified split")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, predictions below this max-proba become 'unknown'",
    )

    args = parser.parse_args()

    paths = Paths()
    data_path = Path(args.data) if args.data else Path(paths.train_csv)
    model_path = Path(args.model) if args.model else Path(paths.model_path)
    outdir = Path(args.outdir)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("items.csv must have columns: text,label")

    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df["label"] = df["label"].astype(str).fillna("").str.strip()

    df = df[(df["text"] != "") & (df["label"] != "")]
    if df.empty:
        raise ValueError("No valid labeled rows in items.csv after cleaning.")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test, used_stratify = safe_train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=not args.no_stratify,
    )

    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Optional threshold -> unknown
    threshold = args.threshold
    if threshold is not None:
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support predict_proba; cannot threshold.")
        proba = model.predict_proba(X_test)
        conf = proba.max(axis=1)
        y_pred = np.where(conf < threshold, "unknown", y_pred)

    labels = sorted(list(set(y_test) | set(y_pred)))

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0))
    f1w = float(f1_score(y_test, y_pred, average="weighted", labels=labels, zero_division=0))

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Per-class metrics only (filter out summary keys)
    per_class = {
        k: {
            "precision": float(v.get("precision", 0.0)),
            "recall": float(v.get("recall", 0.0)),
            "f1": float(v.get("f1-score", 0.0)),
            "support": float(v.get("support", 0.0)),
        }
        for k, v in report_dict.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }

    # Save confusion matrix CSV
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{label}" for label in labels],
        columns=[f"pred:{label}" for label in labels],
    )

    cm_path = outdir / "confusion.csv"
    cm_df.to_csv(cm_path)

    # Append metrics.json history
    metrics_path = outdir / "metrics.json"
    run = EvalRun(
        timestamp_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        data_path=str(data_path),
        model_path=str(model_path),
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=bool(used_stratify),
        threshold_unknown=threshold if threshold is not None else None,
        n_total=int(len(df)),
        n_test=int(len(X_test)),
        labels=labels,
        accuracy=acc,
        f1_macro=f1m,
        f1_weighted=f1w,
        data_sha256=sha256_file(data_path),
        model_sha256=sha256_file(model_path),
        per_class=per_class,
    )

    history: list[dict[str, Any]] = []
    if metrics_path.exists():
        try:
            history = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []

    history.append(asdict(run))
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Console output
    print("=== Evaluation (held-out test set) ===")
    print(f"Data:  {data_path}")
    print(f"Model: {model_path}")
    print(f"Split: test_size={args.test_size}, seed={args.seed}, stratify={used_stratify}")
    if threshold is not None:
        print(f"Threshold: < {threshold} => unknown")
    print("")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
