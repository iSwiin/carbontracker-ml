from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from carbontracker.config import Paths
from carbontracker.engine import score_dataframe
from carbontracker.ocr import OCRConfig, extract_line_items, lines_to_dataframe, ocr_pdf_bytes

app = FastAPI(
    title="CarbonTracker API",
    description="Receipt line-item classification + spend-based CO2e estimation.",
    version="0.1.1",
)

paths = Paths()

# ---------- Models ----------


class Line(BaseModel):
    text: str = Field(..., min_length=1, examples=["2% MILK 1GAL"])
    price: float = Field(..., ge=0.0, examples=[4.29])


class ScoreRequest(BaseModel):
    lines: list[Line]
    conf_threshold: float = Field(0.45, ge=0.0, le=1.0)
    drop_junk: bool = True
    topk: int = Field(
        3, ge=1, le=10, description="Number of top predictions to include per line item"
    )


class TopKPred(BaseModel):
    label: str
    prob: float


class ScoredItem(BaseModel):
    text: str
    price: float
    category: str
    confidence: float
    kgco2e: float
    topk: list[TopKPred]


class ScoreResponse(BaseModel):
    items: list[ScoredItem]
    total_kgco2e: float
    total_spend: float
    unclassified_spend: float
    by_category: dict
    num_lines_scored: int
    conf_threshold: float
    drop_junk: bool
    score_unknown: bool
    topk: int


class OCRScoreResponse(ScoreResponse):
    extracted_lines: list[Line]
    num_extracted_lines: int


def _rows_to_scored_items(rows: list[dict]) -> list[ScoredItem]:
    items: list[ScoredItem] = []
    for row in rows:
        top_list = row.get("topk") or []
        topk_preds = [TopKPred(label=lbl, prob=float(prob)) for lbl, prob in top_list]
        items.append(
            ScoredItem(
                text=row["text"],
                price=float(row["price"]),
                category=row["category"],
                confidence=float(row["confidence"]),
                kgco2e=float(row["kgco2e"]),
                topk=topk_preds,
            )
        )
    return items


# ---------- JSON scoring (already-structured lines) ----------


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    df = pd.DataFrame([line.model_dump() for line in req.lines])

    try:
        result = score_dataframe(
            df=df,
            model_path=paths.model_path,
            factors_path=paths.factors_csv,
            conf_threshold=req.conf_threshold,
            drop_junk=req.drop_junk,
            score_unknown=False,
            topk=req.topk,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    rows = result["items"].to_dict(orient="records")
    items = _rows_to_scored_items(rows)

    return ScoreResponse(
        items=items,
        total_kgco2e=result["total_kgco2e"],
        total_spend=result["total_spend"],
        unclassified_spend=result["unclassified_spend"],
        by_category=result["by_category"],
        num_lines_scored=result["num_lines_scored"],
        conf_threshold=result["conf_threshold"],
        drop_junk=result["drop_junk"],
        score_unknown=result["score_unknown"],
        topk=result.get("topk", req.topk),
    )


# ---------- OCR scoring (upload receipt image/PDF) ----------


@app.post("/ocr-score", response_model=OCRScoreResponse)
async def ocr_score(
    file: UploadFile = File(...),  # noqa: B008
    conf_threshold: float = 0.45,
    drop_junk: bool = True,
    topk: int = 3,
    tesseract_cmd: str | None = None,
) -> OCRScoreResponse:
    """Upload an image/PDF receipt, run OCR, extract candidate (text, price) lines,
    then score CO2e using the existing model + factors.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    name = (file.filename or "").lower()
    cfg = OCRConfig(tesseract_cmd=tesseract_cmd)

    try:
        if name.endswith(".pdf"):
            ocr_text = ocr_pdf_bytes(data, cfg=cfg)
        else:
            import pytesseract
            from PIL import Image

            if cfg.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

            img = Image.open(io.BytesIO(data)).convert("RGB")
            ocr_text = pytesseract.image_to_string(img, config=cfg.tesseract_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}") from e

    extracted = extract_line_items(ocr_text)
    df = lines_to_dataframe(extracted)

    try:
        result = score_dataframe(
            df=df,
            model_path=paths.model_path,
            factors_path=paths.factors_csv,
            conf_threshold=conf_threshold,
            drop_junk=drop_junk,
            score_unknown=False,
            topk=topk,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    rows = result["items"].to_dict(orient="records")
    items = _rows_to_scored_items(rows)
    extracted_lines = [Line(**row) for row in df.to_dict(orient="records")]

    return OCRScoreResponse(
        items=items,
        total_kgco2e=result["total_kgco2e"],
        total_spend=result["total_spend"],
        unclassified_spend=result["unclassified_spend"],
        by_category=result["by_category"],
        num_lines_scored=result["num_lines_scored"],
        conf_threshold=result["conf_threshold"],
        drop_junk=result["drop_junk"],
        score_unknown=result["score_unknown"],
        topk=result.get("topk", topk),
        extracted_lines=extracted_lines,
        num_extracted_lines=int(len(extracted_lines)),
    )
