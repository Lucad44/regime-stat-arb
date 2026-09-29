.PHONY: install test lint run sweep diagnose figures clean

install:
	pip install -e ".[fast,dev]"

test:
	pytest

lint:
	ruff check src tests

run:
	regimearb run --out results/run

sweep:
	regimearb sweep --out results/sweep

diagnose:
	regimearb diagnose --out results/diagnose

figures: run sweep diagnose
	cp results/run/*.png results/sweep/*.png results/diagnose/*.png docs/figures/

clean:
	rm -rf results build dist .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
