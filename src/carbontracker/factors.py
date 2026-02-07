from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import UNKNOWN_LABEL


def load_category_factors(path: Path) -> dict[str, float]:
    """
    Loads a CSV with columns:
      category, kgco2e_per_usd
    Returns dict {category: factor}.
    """
    df = pd.read_csv(path)
    if "category" not in df.columns or "kgco2e_per_usd" not in df.columns:
        raise ValueError(f"{path} must have columns: category, kgco2e_per_usd")

    df["category"] = df["category"].astype(str).str.strip()
    df["kgco2e_per_usd"] = pd.to_numeric(df["kgco2e_per_usd"], errors="coerce")

    factors = {r["category"]: float(r["kgco2e_per_usd"]) for _, r in df.dropna().iterrows()}
    # If unknown exists, keep it. Otherwise we treat unknown as 0 in engine.
    factors.setdefault(UNKNOWN_LABEL, 0.0)
    return factors
