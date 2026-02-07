import sys

from carbontracker.config import DEFAULT_CONF_THRESHOLD, UNKNOWN_LABEL, Paths
from carbontracker.model import load_model, predict_one
from carbontracker.receipt_cleaning import normalize_text


def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print('Usage: python -m scripts.predict "RECEIPT LINE TEXT"')
        raise SystemExit(1)

    paths = Paths()
    model = load_model(paths.model_path)

    cleaned = normalize_text(text)
    label, conf = predict_one(model, cleaned)

    # threshold to avoid guessing
    if conf < DEFAULT_CONF_THRESHOLD:
        label = UNKNOWN_LABEL

    print({"text": cleaned, "label": label, "confidence": round(conf, 3)})


if __name__ == "__main__":
    main()
