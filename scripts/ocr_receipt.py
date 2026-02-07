import sys
from pathlib import Path

from carbontracker.config import Paths
from carbontracker.engine import score_dataframe
from carbontracker.ocr import (
    OCRConfig,
    extract_line_items,
    lines_to_dataframe,
    ocr_image_path,
    ocr_pdf_bytes,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.ocr_receipt <path_to_receipt_image_or_pdf>")
        raise SystemExit(1)

    receipt_path = Path(sys.argv[1])
    if not receipt_path.exists():
        raise FileNotFoundError(receipt_path)

    cfg = OCRConfig()  # optionally: OCRConfig(tesseract_cmd=r"...\tesseract.exe")
    paths = Paths()

    if receipt_path.suffix.lower() == ".pdf":
        pdf_bytes = receipt_path.read_bytes()
        ocr_text = ocr_pdf_bytes(pdf_bytes, cfg=cfg)
    else:
        ocr_text = ocr_image_path(str(receipt_path), cfg=cfg)

    lines = extract_line_items(ocr_text)
    df = lines_to_dataframe(lines)

    print("\n=== Extracted lines (first 30) ===")
    print(df.head(30))

    result = score_dataframe(
        df=df,
        model_path=paths.model_path,
        factors_path=paths.factors_csv,
        conf_threshold=0.45,
        drop_junk=True,
        score_unknown=False,
    )

    scored = result["items"]
    print("\n=== Scored lines (first 30) ===")
    print(scored[["text", "price", "category", "confidence", "kgco2e"]].head(30))

    print("\nTOTAL kgCO2e:", result["total_kgco2e"])
    print("TOTAL $:", result["total_spend"])
    print("UNCLASSIFIED $:", result["unclassified_spend"])
    print("CO2e by category:", result["by_category"])


if __name__ == "__main__":
    main()
