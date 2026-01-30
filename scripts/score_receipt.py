import pandas as pd
import joblib
import numpy as np
import re

MODEL = "models/item_category_clf.joblib"
FACTORS = "data/category_factors.csv"
IN = "data/receipt_lines.csv"

GASOLINE_CO2_PER_GAL = 8.887  # kg CO2 per gallon burned (tailpipe CO2) :contentReference[oaicite:3]{index=3}

model = joblib.load(MODEL)
factors = pd.read_csv(FACTORS).set_index("category")["kgco2e_per_usd"].to_dict()

df = pd.read_csv(IN)
texts = df["text"].astype(str).fillna("").tolist()

proba = model.predict_proba(texts)
pred = model.predict(texts)
conf = proba.max(axis=1)

# don’t guess when unsure
pred = np.where(conf < 0.55, "unknown", pred)

df["category"] = pred
df["confidence"] = conf.round(3)

# Spend-based CO2e
df["kgco2e"] = df.apply(
    lambda r: float(r["price"]) * float(factors.get(r["category"], factors.get("unknown", 0.0))),
    axis=1
)

# Optional: add tailpipe for gasoline if gallons is present in text
def parse_gallons(t: str):
    m = re.search(r"(\d+(\.\d+)?)\s*(GAL|G)\b", t.upper())
    return float(m.group(1)) if m else None

for i, t in enumerate(df["text"].astype(str)):
    if df.loc[i, "category"] == "gasoline":
        gal = parse_gallons(t)
        if gal is not None:
            df.loc[i, "kgco2e"] = gal * GASOLINE_CO2_PER_GAL

print(df[["text", "price", "category", "confidence", "kgco2e"]])
print("\nTOTAL kgCO2e:", round(df["kgco2e"].sum(), 2))
