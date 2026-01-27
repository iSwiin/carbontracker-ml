import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

DATA_PATH = "data/items.csv"
MODEL_PATH = "models/item_category_clf.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df["label"] = df["label"].astype(str).fillna("").str.strip()

    # basic sanity check
    if df.empty or df["text"].nunique() < 10:
        raise ValueError("Add more labeled examples to data/items.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
        ("clf", LogisticRegression(max_iter=3000, n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== Evaluation on held-out test set ===")
    print(classification_report(y_test, preds))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
