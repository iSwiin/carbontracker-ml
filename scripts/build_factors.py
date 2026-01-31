from pathlib import Path
import pandas as pd

from carbontracker.config import Paths, UNKNOWN_LABEL


# Category -> exact "2017 NAICS Title" string in the EPA CSV
MAP = {
    "beef": "Beef Cattle Ranching and Farming",
    "poultry": "Other Poultry Production",
    "dairy": "Dairy Cattle and Milk Production",
    "produce": "Other Vegetable (except Potato) and Melon Farming",
    "restaurant": "Full-Service Restaurants",
    "packaged_food": "All Other Miscellaneous Food Manufacturing",
    "beverages": "Soft Drink Manufacturing",
    "clothing": "Men's and Boys' Cut and Sew Apparel Manufacturing",
    "electronics": "Electronic Computer Manufacturing",
    "household": "Soap and Other Detergent Manufacturing",
    # leave UNKNOWN_LABEL out of MAP on purpose
}


def _detect_factor_col(df: pd.DataFrame) -> str:
    """
    EPA file column names differ across versions. Prefer the 'with margins' CO2e-per-USD column.
    """
    # Most common in older versions:
    if "Supply Chain Emission Factors with Margins" in df.columns:
        return "Supply Chain Emission Factors with Margins"

    # Newer variants often include "margins" + "CO2e" + "USD"
    candidates = [
        c for c in df.columns
        if ("margins" in c.lower()) and ("co2e" in c.lower()) and ("usd" in c.lower())
    ]
    if candidates:
        return candidates[0]

    raise ValueError(f"Could not detect EPA factor column. Columns: {list(df.columns)}")


def main():
    paths = Paths()

    # Expect EPA CSV to be present locally (ignored by git)
    # If your EPA file name differs, update this line:
    epa_csv = Path("data") / "epa_factors.csv"
    out_csv = paths.factors_csv

    df = pd.read_csv(epa_csv)
    if "2017 NAICS Title" not in df.columns:
        raise ValueError("EPA CSV must include '2017 NAICS Title' column")

    df["2017 NAICS Title"] = df["2017 NAICS Title"].astype(str).str.strip()
    factor_col = _detect_factor_col(df)
    df[factor_col] = pd.to_numeric(df[factor_col], errors="coerce")

    def lookup_exact(title_exact: str) -> float:
        title_exact = title_exact.strip()
        match = df[df["2017 NAICS Title"] == title_exact]
        if match.empty:
            raise ValueError(f"Could not find NAICS Title: {title_exact}")
        val = match.iloc[0][factor_col]
        if pd.isna(val):
            raise ValueError(f"Factor value is NaN for NAICS Title: {title_exact}")
        return float(val)

    rows = []
    for cat, title in MAP.items():
        rows.append((cat, lookup_exact(title)))

    # Fallback for unknown = median of all available sectors
    unknown_factor = float(df[factor_col].dropna().median())
    rows.append((UNKNOWN_LABEL, unknown_factor))

    out = pd.DataFrame(rows, columns=["category", "kgco2e_per_usd"]).sort_values("kgco2e_per_usd")
    out.to_csv(out_csv, index=False)

    print("Using EPA factor column:", factor_col)
    print("Wrote", out_csv)
    print(out)


if __name__ == "__main__":
    main()
