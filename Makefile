.PHONY: install train serve test lint format docker-build compose-up drift data clean

install:
	pip install -e ".[dev]"

train:
	intent-train --data data/raw/online_shoppers_intention.csv

track:
	MLFLOW_TRACKING_URI=http://localhost:5000 intent-train --track --register-model

serve:
	uvicorn intent_mlops.serve:app --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

docker-build:
	docker build --target with-model -t purchase-intent-mlops:local .

compose-up:
	docker compose up --build

drift:
	intent-monitor --reference data/raw/online_shoppers_intention.csv \
		--current data/raw/online_shoppers_intention.csv \
		--output artifacts/drift.json

data:
	python scripts/fetch_data.py --force

clean:
	rm -rf artifacts mlruns .pytest_cache .ruff_cache
