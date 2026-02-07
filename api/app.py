from __future__ import annotations

from typing import List, Optional

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
    lines: List[Line]
    conf_threshold: float = Field(0.45, ge=0.0, le=1.0)
    drop_junk: bool = True


class ScoredItem(BaseModel):
    text: str
    price: float
    category: str
    confidence: float
    kgco2e: float


class ScoreResponse(BaseModel):
    items: List[ScoredItem]
    total_kgco2e: float
    total_spend: float
    unclassified_spend: float
    by_category: dict
    num_lines_scored: int
    conf_threshold: float
    drop_junk: bool
    score_unknown: bool


class OCRScoreResponse(ScoreResponse):
    extracted_lines: List[Line]
    num_extracted_lines: int


# ---------- JSON scoring (already-structured lines) ----------


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    df = pd.DataFrame([l.model_dump() for l in req.lines])

    try:
        result = score_dataframe(
            df=df,
            model_path=paths.model_path,
            factors_path=paths.factors_csv,
            conf_threshold=req.conf_threshold,
            drop_junk=req.drop_junk,
            score_unknown=False,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    items = [ScoredItem(**row) for row in result["items"].to_dict(orient="records")]
    result["items"] = items
    return ScoreResponse(**result)


# ---------- OCR scoring (upload receipt image/PDF) ----------


@app.post("/ocr-score", response_model=OCRScoreResponse)
async def ocr_score(
    file: UploadFile = File(...),
    conf_threshold: float = 0.45,
    drop_junk: bool = True,
    tesseract_cmd: Optional[str] = None,
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
            # OCR image bytes without writing a temp file
            import pytesseract
            from PIL import Image
            import io

            if cfg.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

            img = Image.open(io.BytesIO(data)).convert("RGB")
            ocr_text = pytesseract.image_to_string(img, config=cfg.tesseract_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

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
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    scored_items = [ScoredItem(**row) for row in result["items"].to_dict(orient="records")]
    extracted_lines = [Line(**row) for row in df.to_dict(orient="records")]

    result["items"] = scored_items
    return OCRScoreResponse(
        **{k: v for k, v in result.items() if k != "items"},
        items=scored_items,
        extracted_lines=extracted_lines,
        num_extracted_lines=int(len(extracted_lines)),
    )
