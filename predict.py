import sys
import joblib

MODEL_PATH = "models/item_category_clf.joblib"

def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print("Usage: python predict.py <receipt line item text>")
        return

    model = joblib.load(MODEL_PATH)

    # predict + confidence
    proba = model.predict_proba([text])[0]
    label = model.predict([text])[0]
    conf = float(proba.max())

    # threshold to avoid guessing
    if conf < 0.55:
        label = "unknown"

    print({"text": text, "label": label, "confidence": round(conf, 3)})

if __name__ == "__main__":
    main()
