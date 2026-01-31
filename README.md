# CarbonTrackerML
Receipt line-item classification + spend-based CO₂e estimation from everyday purchases.

CarbonTrackerML includes:
- A text classifier that maps receipt line items → spend categories (e.g., dairy, beef, gasoline)
- A CO₂e scoring pipeline using category emission factors (kgCO₂e per USD)
- A FastAPI backend exposing POST /score for app integration

## Project layout
CarbonTrackerML/
  api/                    FastAPI app
  data/                   CSV inputs (large/generated files typically ignored by git)
    sample/               small demo inputs you can commit
  models/                 trained model artifacts (ignored by git)
  scripts/                CLI wrappers (train/score/predict/build factors)
  src/carbontracker/      core package code
  tests/                  optional pytest tests

## Setup (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install fastapi uvicorn

## Data files
Training data: data/items.csv
Format:
text,label
CHKN BRST 1.24LB,poultry
2% MILK 1GAL,dairy
PAPER TOWELS 6ROLL,household

Emission factors: data/category_factors.csv
Format:
category,kgco2e_per_usd
beef,3.298
dairy,2.578
household,0.412
unknown,0.000

Scoring input: data/receipt_lines.csv
Format:
text,price
CHKN BRST 1.24LB,9.87
2% MILK 1GAL,4.29
PAPER TOWELS 6ROLL,12.99

Tip: commit small examples in data/sample/ and keep large/generated files ignored by git.

## Train the model
python -m scripts.train
Saves: models/item_category_clf.joblib

## Predict one line item
python -m scripts.predict "CHKN BRST 1.24LB"
Example output:
{"text":"CHKN BRST 1.24LB","label":"poultry","confidence":0.91}

## Score a receipt CSV (end-to-end)
python -m scripts.score_receipt
Prints:
- line-by-line results (text, price, category, confidence, kgCO₂e)
- total kgCO₂e
- total spend and unclassified spend
- kgCO₂e by category

## Build category_factors.csv from EPA factors (optional)
If you have an EPA emission factor CSV in data/epa_factors.csv:
python -m scripts.build_factors
This generates data/category_factors.csv based on NAICS title mappings in the script.

## Run the API
Start server:
python -m uvicorn api.app:app --reload
Swagger docs:
http://127.0.0.1:8000/docs

POST /score request body example:
{
  "lines": [
    { "text": "CHKN BRST 1.24LB", "price": 9.87 },
    { "text": "2% MILK 1GAL", "price": 4.29 }
  ],
  "conf_threshold": 0.45,
  "drop_junk": true
}

Response includes:
- items (line-by-line)
- total_kgco2e
- total_spend
- unclassified_spend
- by_category

## Notes / limitations
- CO₂e estimates are spend-based (kgCO₂e per USD) using category factors; values vary by geography and supply chain.
- This repo assumes you already have structured line items (text + price). OCR and parsing are separate modules.

## License
MIT (add a LICENSE file if you want this explicitly licensed).
