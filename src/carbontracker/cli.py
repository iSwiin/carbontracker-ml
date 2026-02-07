from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import DEFAULT_CONF_THRESHOLD, Paths
from .engine import score_dataframe, score_receipt_csv
from .model import load_model, predict_one, save_model, train_from_csv


def _cmd_train(args: argparse.Namespace) -> None:
    paths = Paths()
    data_path = Path(args.data) if args.data else paths.train_csv
    out_path = Path(args.out) if args.out else paths.model_path

    res = train_from_csv(
        csv_path=data_path,
        text_col="text",
        label_col="label",
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(res.report)

    save_model(res.model, out_path)
    print(f"Saved model to: {out_path}")


def _cmd_predict(args: argparse.Namespace) -> None:
    paths = Paths()
    model_path = Path(args.model) if args.model else paths.model_path
    model = load_model(model_path)

    label, conf = predict_one(model, args.text)
    out = {"text": args.text, "label": label, "confidence": round(conf, 3)}
    print(json.dumps(out, ensure_ascii=False))


def _cmd_score_csv(args: argparse.Namespace) -> None:
    paths = Paths()

    csv_path = Path(args.csv) if args.csv else paths.receipt_lines_csv
    factors_path = Path(args.factors) if args.factors else paths.factors_csv
    model_path = Path(args.model) if args.model else paths.model_path

    df = pd.read_csv(csv_path)
    result = score_dataframe(
        df=df,
        model_path=model_path,
        factors_path=factors_path,
        conf_threshold=args.threshold,
        drop_junk=not args.keep_junk,
        score_unknown=args.score_unknown,
    )

    items = result["items"]
    if args.out_items:
        out_items = Path(args.out_items)
        out_items.parent.mkdir(parents=True, exist_ok=True)
        items.to_csv(out_items, index=False)
        print(f"Wrote: {out_items}")

    # Print compact JSON summary
    summary: Dict[str, Any] = {
        "total_kgco2e": result["total_kgco2e"],
        "total_spend": result["total_spend"],
        "unclassified_spend": result["unclassified_spend"],
        "by_category": result["by_category"],
        "num_lines_scored": result["num_lines_scored"],
        "conf_threshold": result["conf_threshold"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="carbontracker", description="CarbonTrackerML CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train classifier from data/items.csv and save a model artifact")
    p_train.add_argument("--data", default=None, help="Path to items.csv (default: data/items.csv)")
    p_train.add_argument("--out", default=None, help="Output .joblib path (default: models/item_category_clf.joblib)")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Predict a category for a single line item")
    p_pred.add_argument("text", help="Line item text, e.g. '2% MILK 1GAL'")
    p_pred.add_argument("--model", default=None, help="Path to model .joblib")
    p_pred.set_defaults(func=_cmd_predict)

    p_score = sub.add_parser("score-csv", help="Score a receipt CSV (text,price) end-to-end")
    p_score.add_argument("--csv", default=None, help="Path to receipt CSV (default: data/receipt_lines.csv)")
    p_score.add_argument("--model", default=None, help="Path to model .joblib")
    p_score.add_argument("--factors", default=None, help="Path to category_factors.csv")
    p_score.add_argument("--threshold", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold")
    p_score.add_argument("--keep-junk", action="store_true", help="Do not drop receipt metadata lines")
    p_score.add_argument("--score-unknown", action="store_true", help="Score unknown items using factor if present")
    p_score.add_argument("--out-items", default=None, help="Optional path to write scored line items as CSV")
    p_score.set_defaults(func=_cmd_score_csv)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
