import pandas as pd

EPA_CSV = "data/epa_factors.csv"
OUT = "data/category_factors.csv"

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
    # leave unknown out of MAP on purpose
}

df = pd.read_csv(EPA_CSV)
df["2017 NAICS Title"] = df["2017 NAICS Title"].astype(str).str.strip()
df["Supply Chain Emission Factors with Margins"] = pd.to_numeric(
    df["Supply Chain Emission Factors with Margins"], errors="coerce"
)

def lookup_exact(title_exact: str) -> float:
    title_exact = title_exact.strip()
    match = df[df["2017 NAICS Title"] == title_exact]
    if match.empty:
        raise ValueError(f"Could not find NAICS Title: {title_exact}")
    return float(match.iloc[0]["Supply Chain Emission Factors with Margins"])

rows = []
for cat, title in MAP.items():
    rows.append((cat, lookup_exact(title)))

# Fallback for unknown = median of all sectors
unknown_factor = float(df["Supply Chain Emission Factors with Margins"].median())
rows.append(("unknown", unknown_factor))

out = pd.DataFrame(rows, columns=["category", "kgco2e_per_usd"])
out.to_csv(OUT, index=False)
print("Wrote", OUT)
print(out.sort_values("kgco2e_per_usd"))
