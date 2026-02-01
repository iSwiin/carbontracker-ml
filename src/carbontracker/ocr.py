from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
from PIL import Image
import pytesseract

# Optional PDF support
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None


_AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})*\.\d{2})(?!\d)")
_CURRENCY_RE = re.compile(r"[$€£]\s*")


@dataclass
class OCRConfig:
    # If tesseract isn't on PATH, set this to the full path to tesseract.exe
    tesseract_cmd: Optional[str] = None
    # OCR engine mode / page segmentation mode; 6 works well for receipt blocks
    tesseract_config: str = "--oem 3 --psm 6"
    # For PDF conversion
    pdf_dpi: int = 250
    pdf_first_page_only: bool = True


def _configure_tesseract(cfg: OCRConfig) -> None:
    """
    Configure pytesseract to find the tesseract executable.
    """
    cmd = cfg.tesseract_cmd or os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def _normalize_line(s: str) -> str:
    s = s.replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def ocr_image_bytes(image_bytes: bytes, cfg: OCRConfig = OCRConfig()) -> str:
    """
    OCR raw image bytes -> text.
    """
    _configure_tesseract(cfg)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(img, config=cfg.tesseract_config)


def ocr_image_path(image_path: str, cfg: OCRConfig = OCRConfig()) -> str:
    """
    OCR image file path -> text.
    """
    _configure_tesseract(cfg)
    img = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(img, config=cfg.tesseract_config)


def ocr_pdf_bytes(pdf_bytes: bytes, cfg: OCRConfig = OCRConfig()) -> str:
    """
    OCR a PDF (bytes) by converting pages to images then OCR'ing them.
    Requires pdf2image + Poppler.
    """
    if convert_from_bytes is None:
        raise RuntimeError("pdf2image not available. Install: python -m pip install pdf2image")

    _configure_tesseract(cfg)

    pages = convert_from_bytes(
        pdf_bytes,
        dpi=cfg.pdf_dpi,
        first_page=1,
        last_page=1 if cfg.pdf_first_page_only else None,
    )

    texts = []
    for page in pages:
        texts.append(pytesseract.image_to_string(page, config=cfg.tesseract_config))
    return "\n".join(texts)


def extract_line_items(ocr_text: str) -> List[Dict[str, float | str]]:
    """
    Convert OCR text into candidate receipt line items with prices.

    Heuristic:
    - For each line, find the LAST currency amount like 12.34
    - Treat that as price
    - Remaining text (left side) is item text
    """
    lines_out: List[Dict[str, float | str]] = []
    for raw in (ocr_text or "").splitlines():
        line = _normalize_line(raw)
        if not line or len(line) < 3:
            continue

        # Remove currency symbols to simplify parsing
        cleaned = _CURRENCY_RE.sub("", line)

        amounts = list(_AMOUNT_RE.finditer(cleaned))
        if not amounts:
            continue

        last = amounts[-1].group(1).replace(",", "")
        try:
            price = float(last)
        except Exception:
            continue

        # Remove the last amount occurrence from the text to get item description
        start, end = amounts[-1].span()
        item_text = (cleaned[:start] + cleaned[end:]).strip(" -")
        item_text = _normalize_line(item_text)

        if not item_text:
            continue

        lines_out.append({"text": item_text.upper(), "price": price})

    return lines_out


def lines_to_dataframe(lines: List[Dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(lines, columns=["text", "price"])


# Needed because we used io.BytesIO above
import io
