from __future__ import annotations

import re
from typing import Iterable

from .config import JUNK_KEYWORDS


_multi_space = re.compile(r"\s+")
_nonprint = re.compile(r"[^\x20-\x7E]+")


def normalize_text(text: str) -> str:
    """
    Normalizes OCR-ish receipt text to reduce noise.
    """
    t = "" if text is None else str(text)
    t = _nonprint.sub(" ", t)
    t = t.strip()
    t = _multi_space.sub(" ", t)
    return t


def is_junk_line(text: str, junk_keywords: Iterable[str] = JUNK_KEYWORDS) -> bool:
    """
    Returns True if the line looks like receipt metadata (TOTAL, TAX, AUTH, etc.).
    """
    t = normalize_text(text).upper()
    if not t:
        return True
    return any(k in t for k in junk_keywords)
