# Drift report (PSI)

- **model**: hist_gradient_boosting `hgb-511569fef2`
- **reference**: training period of the model (profiles stored in the artefact)
- **current**: `data/transactions.parquet`, 106,503 transactions from 2026-04-30 00:00

1 significant, 0 moderate, 21 stable (PSI thresholds 0.10 / 0.25).

| feature | PSI | status |
|---|---:|---|
| category_risk | 0.4185 | significant |
| amount_zscore | 0.0237 | stable |
| new_country | 0.0150 | stable |
| merchant_risk | 0.0121 | stable |
| log_merchant_volume | 0.0061 | stable |
| card_tx_count_7d | 0.0014 | stable |
| model_score | 0.0008 | stable |
| new_device | 0.0005 | stable |
| log_amount_ratio | 0.0003 | stable |
| hour_cos | 0.0003 | stable |
| log_card_amount_24h | 0.0002 | stable |
| log_amount | 0.0002 | stable |
| hour_sin | 0.0001 | stable |
| card_tx_count_24h | 0.0001 | stable |
| log_seconds_since_last | 0.0001 | stable |
| card_tx_count_10m | 0.0001 | stable |
| is_online | 0.0000 | stable |
| country_change | 0.0000 | stable |
| is_night | 0.0000 | stable |
| card_tx_count_1h | 0.0000 | stable |
| device_change | 0.0000 | stable |
| card_history_capped | 0.0000 | stable |
