"""End-to-end run of the command line on a small dataset."""

from __future__ import annotations

import pandas as pd

from fraud_detection.cli import main

GENERATE = "generate --cards 250 --merchants 80 --days 80 --seed 3 --fraud-multiplier 3"


def run(command: str) -> int:
    return main(command.split())


def test_generate_train_drift(tmp_path, capsys):
    data, wave = tmp_path / "tx.parquet", tmp_path / "wave.parquet"
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"

    assert run(f"{GENERATE} --out {data}") == 0
    assert len(pd.read_parquet(data)) > 10_000

    assert run(f"train --data {data} --artifacts {artifacts} --reports {reports}") == 0
    assert (artifacts / "model.joblib").exists()
    assert (reports / "model_card.md").read_text().startswith("# Model card: ")

    # Same population, prices tripled: log_amount must be flagged.
    assert run(f"{GENERATE} --amount-scale 3 --out {wave}") == 0
    drift = reports / "drift.md"
    assert run(f"drift --model {artifacts / 'model.joblib'} --data {wave} --out {drift}") == 0
    row = next(line for line in drift.read_text().splitlines() if "| log_amount |" in line)
    assert "significant" in row
    assert "Drift report" in capsys.readouterr().out
