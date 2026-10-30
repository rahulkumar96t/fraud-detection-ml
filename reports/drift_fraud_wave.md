# Drift report (PSI)

- **model**: hist_gradient_boosting `hgb-511569fef2`
- **reference**: training period of the model (profiles stored in the artefact)
- **current**: `data/fraud_wave.parquet`, 410,828 transactions from 2026-07-02 00:00

1 significant, 3 moderate, 18 stable (PSI thresholds 0.10 / 0.25).

| feature | PSI | status |
|---|---:|---|
| category_risk | 1.9956 | significant |
| log_amount | 0.2042 | moderate |
| merchant_risk | 0.1927 | moderate |
| log_card_amount_24h | 0.1858 | moderate |
| model_score | 0.0089 | stable |
| amount_zscore | 0.0071 | stable |
| new_country | 0.0020 | stable |
| log_merchant_volume | 0.0015 | stable |
| log_amount_ratio | 0.0007 | stable |
| card_tx_count_7d | 0.0003 | stable |
| card_tx_count_10m | 0.0001 | stable |
| hour_sin | 0.0001 | stable |
| card_tx_count_1h | 0.0001 | stable |
| new_device | 0.0000 | stable |
| log_seconds_since_last | 0.0000 | stable |
| hour_cos | 0.0000 | stable |
| is_online | 0.0000 | stable |
| card_tx_count_24h | 0.0000 | stable |
| is_night | 0.0000 | stable |
| device_change | 0.0000 | stable |
| country_change | 0.0000 | stable |
| card_history_capped | 0.0000 | stable |
