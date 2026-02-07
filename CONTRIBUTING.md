# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,api,ocr]"
```

## Quality gates

```bash
ruff format .
ruff check .
pytest
```

## Notes

- Keep large/private data out of git. Use `data/sample/` for small reproducible examples.
- Store trained model artifacts in `models/` (ignored by git).
