.PHONY: install data train drift serve lint test demo clean

install:
	uv sync --locked

# fraud_wave: same population, later months, prices x2 and 3x the fraud - input for the drift demo
data:
	uv run fraud-ml generate --out data/transactions.parquet
	uv run fraud-ml generate --start 2026-06-01 --amount-scale 2.0 --fraud-multiplier 3 \
		--out data/fraud_wave.parquet

train:
	uv run fraud-ml train --data data/transactions.parquet

drift:
	uv run fraud-ml drift --data data/transactions.parquet --since 2026-04-30 \
		--out reports/drift_test_period.md
	uv run fraud-ml drift --data data/fraud_wave.parquet --out reports/drift_fraud_wave.md

serve:
	uv run fraud-ml serve --model artifacts/model.joblib --history data/transactions.parquet

lint:
	uv run ruff format --check .
	uv run ruff check .

test:
	uv run pytest -q

demo: data train drift

clean:
	rm -rf data artifacts .pytest_cache .ruff_cache
