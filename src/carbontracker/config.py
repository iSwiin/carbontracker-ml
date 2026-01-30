from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_TRAIN_CSV = DATA_DIR / "items.csv"
DEFAULT_FACTORS_CSV = DATA_DIR / "category_factors.csv"
DEFAULT_RECEIPT_LINES_CSV = DATA_DIR / "receipt_lines.csv"

DEFAULT_MODEL_PATH = MODELS_DIR / "item_category_clf.joblib"

# If confidence is below this, we mark as "unknown"
DEFAULT_CONF_THRESHOLD = 0.45

# Lines containing any of these tokens are not purchasable items
JUNK_KEYWORDS = [
    "SUBTOTAL",
    "TOTAL",
    "TAX",
    "BALANCE",
    "VISA",
    "MASTERCARD",
    "AMEX",
    "DISCOVER",
    "DEBIT",
    "CREDIT",
    "AUTH",
    "APPROVAL",
    "APPROVED",
    "CHANGE",
    "CASHIER",
    "REGISTER",
    "STORE",
    "TERMINAL",
    "TRANSACTION",
    "REF",
    "REFERENCE",
    "VOID",
    "RETURN",
]

# Default label name to use for non-classified items
UNKNOWN_LABEL = "unknown"


@dataclass(frozen=True)
class Paths:
    train_csv: Path = DEFAULT_TRAIN_CSV
    factors_csv: Path = DEFAULT_FACTORS_CSV
    receipt_lines_csv: Path = DEFAULT_RECEIPT_LINES_CSV
    model_path: Path = DEFAULT_MODEL_PATH
