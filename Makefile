.PHONY: help install lint format check test cov build publish clean

help:
	@echo "Available targets:"
	@echo "  install  Install the project with dev extras (editable)"
	@echo "  lint     Run ruff lint checks"
	@echo "  format   Auto-format the code with ruff"
	@echo "  check    Run ruff (lint + format check) without modifying files"
	@echo "  test     Run the test suite with pytest"
	@echo "  cov      Run tests with coverage report"
	@echo "  build    Build sdist + wheel into ./dist"
	@echo "  publish  Upload ./dist artifacts to PyPI via twine"
	@echo "  clean    Remove build/test caches and artifacts"

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

check:
	ruff check .
	ruff format --check .

test:
	pytest

cov:
	pytest --cov=ftlangdetect --cov-report=term-missing

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
