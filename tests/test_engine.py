import pandas as pd

from carbontracker.config import UNKNOWN_LABEL
from carbontracker.engine import score_dataframe
from carbontracker.model import save_model, train_from_csv


def test_score_dataframe_end_to_end(tmp_path):
    # Minimal labeled data
    items_csv = tmp_path / "items.csv"
    items_csv.write_text(
        "text,label\n"
        "2% MILK,dairy\n"
        "WHOLE MILK,dairy\n"
        "BEEF JERKY,beef\n"
        "BEEF STEAK,beef\n"
        "GASOLINE,gasoline\n"
        "UNLEADED GAS,gasoline\n",
        encoding="utf-8",
    )

    factors_csv = tmp_path / "category_factors.csv"
    factors_csv.write_text(
        "category,kgco2e_per_usd\ndairy,2.0\nbeef,3.0\ngasoline,1.0\nunknown,0.0\n",
        encoding="utf-8",
    )

    model_path = tmp_path / "model.joblib"
    res = train_from_csv(items_csv, test_size=0.33, random_state=0)
    save_model(res.model, model_path)

    df = pd.DataFrame(
        [
            {"text": "2% MILK", "price": 5.00},
            {"text": "BEEF STEAK", "price": 10.00},
            {"text": "TOTAL 15.00", "price": 15.00},
        ]
    )

    out = score_dataframe(
        df=df,
        model_path=model_path,
        factors_path=factors_csv,
        conf_threshold=0.0,  # don't force unknown
        drop_junk=True,
        score_unknown=False,
    )

    assert out["num_lines_scored"] == 2  # TOTAL line dropped
    items = out["items"]
    assert set(items.columns) >= {"text", "price", "category", "confidence", "kgco2e"}
    assert float(items["kgco2e"].sum()) >= 0.0

    # if any unknown exists, it should have 0 kgco2e with score_unknown=False
    unk = items[items["category"] == UNKNOWN_LABEL]
    if len(unk):
        assert float(unk["kgco2e"].sum()) == 0.0
