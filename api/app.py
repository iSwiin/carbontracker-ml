from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

from carbontracker.engine import score_dataframe
from carbontracker.config import Paths

app = FastAPI(title="CarbonTracker API")
paths = Paths()

class Line(BaseModel):
    text: str
    price: float

class ScoreRequest(BaseModel):
    lines: List[Line]
    conf_threshold: float = 0.45
    drop_junk: bool = True

@app.post("/score")
def score(req: ScoreRequest):
    df = pd.DataFrame([l.model_dump() for l in req.lines])
    result = score_dataframe(
        df=df,
        model_path=paths.model_path,
        factors_path=paths.factors_csv,
        conf_threshold=req.conf_threshold,
        drop_junk=req.drop_junk,
        score_unknown=False,
    )
    result["items"] = result["items"].to_dict(orient="records")
    return result
