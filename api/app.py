from __future__ import annotations

from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from carbontracker.config import Paths
from carbontracker.engine import score_dataframe
from carbontracker.ocr import (
    OCRConfig,
    ocr_pdf_bytes,
    extract_line_items,
    lines_to_dataframe,
)

app = FastAPI(title="CarbonTracker API")
paths = Paths()


# ---------- JSON scoring (already-structured lines) ----------

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


# ---------- OCR scoring (upload receipt image/PDF) ----------

@app.post("/ocr-score")
async def ocr_score(
    file: UploadFile = File(...),
    conf_threshold: float = 0.45,
    drop_junk: bool = True,
    tesseract_cmd: Optional[str] = None,
):
    """
    Upload an image/PDF receipt, run OCR, extract candidate (text, price) lines,
    then score CO2e using the existing model + factors.

    Query params:
      - conf_threshold: below this => category="unknown"
      - drop_junk: remove lines like TOTAL/TAX/AUTH
      - tesseract_cmd: optional full path to tesseract.exe (if not on PATH)
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
            import io
            from PIL import Image
            import pytesseract

            if cfg.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

            img = Image.open(io.BytesIO(data)).convert("RGB")
            ocr_text = pytesseract.image_to_string(img, config=cfg.tesseract_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    extracted = extract_line_items(ocr_text)
    df = lines_to_dataframe(extracted)

    result = score_dataframe(
        df=df,
        model_path=paths.model_path,
        factors_path=paths.factors_csv,
        conf_threshold=conf_threshold,
        drop_junk=drop_junk,
        score_unknown=False,
    )

    result["items"] = result["items"].to_dict(orient="records")
    result["extracted_lines"] = df.to_dict(orient="records")
    result["num_extracted_lines"] = int(len(df))

    return result
