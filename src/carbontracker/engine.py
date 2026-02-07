from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_CONF_THRESHOLD, UNKNOWN_LABEL, Paths
from .factors import load_category_factors
from .model import load_model, predict_one
from .receipt_cleaning import is_junk_line, normalize_text


@dataclass
class ScoredLine:
    text: str
    price: float
    category: str
    confidence: float
    kgco2e: float


def score_dataframe(
    df: pd.DataFrame,
    model_path: Path,
    factors_path: Path,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    drop_junk: bool = True,
    score_unknown: bool = False,
) -> dict[str, Any]:
    """
    df must have columns: text, price
    Returns dict with items, totals, breakdown.
    """
    if "text" not in df.columns or "price" not in df.columns:
        raise ValueError("Input df must have columns: text, price")

    df = df.copy()
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    if drop_junk:
        df["is_junk"] = df["text"].map(is_junk_line)
        df = df[~df["is_junk"]].copy()
    else:
        df["is_junk"] = False

    model = load_model(model_path)
    factors = load_category_factors(factors_path)

    scored: list[ScoredLine] = []
    for _, row in df.iterrows():
        text = row["text"]
        price = float(row["price"])

        pred, conf = predict_one(model, text)

        # apply threshold
        if conf < conf_threshold:
            pred = UNKNOWN_LABEL

        # CO2 scoring
        if pred == UNKNOWN_LABEL and not score_unknown:
            kg = 0.0
        else:
            kg = price * float(factors.get(pred, 0.0))

        scored.append(
            ScoredLine(
                text=text,
                price=price,
                category=pred,
                confidence=round(conf, 3),
                kgco2e=kg,
            )
        )

    out_df = pd.DataFrame([s.__dict__ for s in scored])

    total_kg = float(out_df["kgco2e"].sum()) if len(out_df) else 0.0
    total_spend = float(out_df["price"].sum()) if len(out_df) else 0.0
    unclassified_spend = (
        float(out_df.loc[out_df["category"] == UNKNOWN_LABEL, "price"].sum()) if len(out_df) else 0.0
    )

    by_category = (
        out_df.groupby("category")["kgco2e"].sum().sort_values(ascending=False).round(3).to_dict()
        if len(out_df)
        else {}
    )

    return {
        "items": out_df,
        "total_kgco2e": round(total_kg, 3),
        "total_spend": round(total_spend, 2),
        "unclassified_spend": round(unclassified_spend, 2),
        "by_category": by_category,
        "num_lines_scored": int(len(out_df)),
        "conf_threshold": conf_threshold,
        "drop_junk": drop_junk,
        "score_unknown": score_unknown,
    }


def score_receipt_csv(
    csv_path: Path,
    paths: Paths | None = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    drop_junk: bool = True,
    score_unknown: bool = False,
) -> dict[str, Any]:
    paths = paths or Paths()
    df = pd.read_csv(csv_path)
    return score_dataframe(
        df=df,
        model_path=paths.model_path,
        factors_path=paths.factors_csv,
        conf_threshold=conf_threshold,
        drop_junk=drop_junk,
        score_unknown=score_unknown,
    )
