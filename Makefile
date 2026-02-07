# Common developer commands

.PHONY: install-dev format lint test api

install-dev:
	python -m pip install -U pip
	pip install -e ".[dev,api,ocr]"

format:
	ruff format .

lint:
	ruff check .

test:
	pytest

api:
	python -m uvicorn api.app:app --reload
