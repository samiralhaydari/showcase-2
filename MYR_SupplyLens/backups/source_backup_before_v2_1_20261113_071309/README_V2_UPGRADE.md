# MYR SupplyLens V2 — Upgrade Guide

This patch upgrades the existing Drive project in place. It does not delete or overwrite V1 model artifacts.

## What changes

1. Four chronological partitions: 65% train, 10% validation, 10% calibration, 15% test.
2. Nine features: log rate, 1/5/10/20-session momentum, 5/20/60-session volatility, and 20-session log-rate range.
3. Non-overlapping walk-forward test evaluation using common forecast origins spaced 63 observations apart.
4. Benchmarks: random walk, 252-observation historical drift, 20-observation recent drift, and ARIMA(1,1,1).
5. Empirical 80% and 95% forecast uncertainty bands calibrated on a separate calibration partition.
6. Versioned V2 model, scaler, metadata, benchmark and calibration artifacts.
7. V2 Streamlit Model Insights, forecast intervals, and improved currency display precision.

## Critical methodology notes

- The CNN-LSTM weights remain fixed during walk-forward testing; each origin receives an updated historical input window. This is a fixed-model rolling-origin test, not repeated model retraining.
- Test origins are spaced 63 observations apart. This avoids overlapping target blocks for the longest forecast horizon and makes the horizon comparisons use the same origins.
- The resulting independent test-origin count is much smaller than the old overlapping sample count. Always report this count alongside performance metrics.
- Calibration uses a separate chronological period, not the test period. Calibration forecast windows overlap, so the bands are descriptive empirical uncertainty estimates, not a formal conformal coverage guarantee.
- ARIMA is fitted only on observations available at each forecast origin. If an ARIMA fit fails at an origin, that origin is omitted from the ARIMA metric and the count will be lower.
- A positive `skill_vs_random_walk_pct` means lower MAE than random walk. Negative means random walk performed better.
- Do not claim CNN-LSTM is superior unless the held-out results support that claim.

## Apply to existing Google Drive project

Expected project directory:

`/content/drive/MyDrive/MYR_SupplyLens`

1. Upload `MYR_SupplyLens_V2_Patch.zip` and `MYR_SupplyLens_V2_Colab.ipynb` to the root of My Drive.
2. Open the V2 notebook in Colab.
3. Run the setup cell. It extracts only source/requirements/docs into the existing project directory. Existing `artifacts/models`, scalers, metadata, data and SQLite records are not removed.
4. Install the V2 requirements. TensorFlow is not force-reinstalled by the notebook; the Colab runtime's TensorFlow is used.
5. Run the smoke tests and inspect the data.
6. Train USD, CNY and IDR V2 models. Start with USD if you want to test the pipeline before spending time on all three.
7. Launch Streamlit from the notebook.

## Artifact naming

V2 outputs use names such as:

- `artifacts/models/cnn_lstm_v2_USD.keras`
- `artifacts/scalers/scaler_X_v2_USD.joblib`
- `artifacts/scalers/scaler_y_v2_USD.joblib`
- `artifacts/metadata/metadata_v2_USD.json`
- `artifacts/metadata/metrics_v2_USD.csv`
- `artifacts/metadata/walk_forward_details_v2_USD.csv`
- `artifacts/metadata/calibration_v2_USD.csv`

V1 files remain intact. The forecast loader prefers V2 and falls back to V1 if no V2 bundle exists.

## Suggested Colab training commands

```bash
python scripts/train_model.py --currency USD --epochs 40
python scripts/train_model.py --currency CNY --epochs 40
python scripts/train_model.py --currency IDR --epochs 40
```

If ARIMA is taking too long, temporarily skip it:

```bash
python scripts/train_model.py --currency USD --epochs 40 --skip-arima
```

For the final competition screenshots, train all three currencies with ARIMA enabled if time allows.

## If package installation is already healthy

Do not reinstall TensorFlow or force-downgrade NumPy. The notebook installs the requirements list without explicitly reinstalling TensorFlow. If Colab reports a dependency conflict, capture the exact `pip` conflict message and package versions before changing the environment.
