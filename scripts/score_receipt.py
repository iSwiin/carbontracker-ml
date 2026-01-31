from carbontracker.engine import score_receipt_csv
from carbontracker.config import Paths
import re


# Tailpipe CO2 (gasoline) kg CO2 per gallon burned
GASOLINE_CO2_PER_GAL = 8.887


def parse_gallons(text: str):
    """
    Extracts gallons from receipt text like:
      'UNLEADED GAS 12.6GAL' or 'FUEL 10.5 GAL'
    """
    t = (text or "").upper()
    m = re.search(r"(\d+(\.\d+)?)\s*(GAL|G)\b", t)
    return float(m.group(1)) if m else None


def main():
    paths = Paths()

    # Use engine to do: clean -> drop junk -> predict -> threshold -> spend-based kgCO2e
    result = score_receipt_csv(
        csv_path=paths.receipt_lines_csv,
        paths=paths,
        conf_threshold=0.45,   # adjust as needed
        drop_junk=True,
        score_unknown=False,   # unknown gets 0 kgCO2e
    )

    df = result["items"]

    # Optional: replace gasoline spend-based CO2e with tailpipe if gallons can be parsed
    if "category" in df.columns:
        for i, row in df.iterrows():
            if row["category"] == "gasoline":
                gal = parse_gallons(row["text"])
                if gal is not None:
                    df.at[i, "kgco2e"] = gal * GASOLINE_CO2_PER_GAL

    total = round(float(df["kgco2e"].sum()), 2)

    print(df[["text", "price", "category", "confidence", "kgco2e"]])
    print("\nTOTAL kgCO2e:", total)
    print("TOTAL $:", result["total_spend"])
    print("UNCLASSIFIED $:", result["unclassified_spend"])
    print("\nCO2e by category:", result["by_category"])


if __name__ == "__main__":
    main()
