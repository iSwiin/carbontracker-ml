from carbontracker.model import train_from_csv, save_model
from carbontracker.config import Paths


def main():
    paths = Paths()

    res = train_from_csv(
        csv_path=paths.train_csv,
        text_col="text",
        label_col="label",
        test_size=0.2,
        random_state=42,
    )

    print("=== Evaluation on held-out test set ===")
    print(res.report)

    save_model(res.model, paths.model_path)
    print(f"Saved model to: {paths.model_path}")


if __name__ == "__main__":
    main()
