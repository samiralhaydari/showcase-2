# MYR SupplyLens V2.1 — Upgrade Guide

This patch upgrades the existing Google Drive project in place. It creates a new V2.1 model-artifact family and retains V1/V2 model files, scalers, data and SQLite records.

## Main changes

1. Five chronological partitions: 60% train, 10% early-stopping validation, 10% method selection, 5% calibration, 15% final test.
2. Validation-based selection of one method per currency and displayed horizon.
3. Five comparable forecasting methods: CNN-LSTM, random walk, historical drift (252), recent drift (20), and ARIMA(1,1,1).
4. Separate MAE skill and RMSE skill metrics, each with paired-bootstrap 95% confidence intervals.
5. Random-walk directional accuracy is correctly represented as N/A because the method predicts no directional change.
6. Method-specific empirical calibration bands.
7. Forecast and business simulators use the validation-selected method at 5, 21, 42 and 63 sessions.
8. V2.1 source and model artifacts are versioned separately from V2/V1.

## Selection and test protocol

- The selection partition is chronologically separate from training and early stopping.
- Within each selection horizon, forecast origins are spaced by that horizon to avoid overlapping target blocks for that horizon.
- Lowest selection-partition MAE is the primary criterion. RMSE and absolute-error standard deviation are tie-breakers.
- A minimum of eight selection origins is required. Otherwise, random walk is used as the conservative fallback.
- The selection policy is saved before the final test evaluation. Test results do not select the model.
- Final test origins use common 63-observation spacing for all displayed horizons.
- Paired bootstrap confidence intervals use identical test origins for each candidate and random walk. Small sample counts still require cautious interpretation.

## Apply to Google Drive

Expected project directory: `/content/drive/MyDrive/MYR_SupplyLens`.

1. Upload `MYR_SupplyLens_V2_1_Patch.zip` and `MYR_SupplyLens_V2_1_Colab.ipynb` to the root of My Drive.
2. Open the notebook in Colab.
3. Run the setup cell. It creates a source backup and overlays only allow-listed source, test, script and documentation files.
4. It does not extract or delete `data/` or `artifacts/`.
5. Install the Colab requirements. TensorFlow is not force-reinstalled.
6. Run `python -m pytest -q` and inspect the source data.
7. Train the desired currencies. The new `_v2_1_` files do not overwrite V2 or V1 bundles.
8. Launch Streamlit.

## Suggested training commands

```bash
python scripts/train_model.py --currency USD --epochs 40
python scripts/train_model.py --currency CNY --epochs 40
python scripts/train_model.py --currency IDR --epochs 40
```

If ARIMA is too slow, temporarily skip it:

```bash
python scripts/train_model.py --currency USD --epochs 40 --skip-arima
```

To reduce calibration-time ARIMA fitting, increase the calibration stride:

```bash
python scripts/train_model.py --currency USD --epochs 40 --calibration-stride 10
```

A larger stride reduces calibration samples. If fewer than 30 samples remain at a method/horizon, the corresponding interval is marked unavailable.

## Important interpretation

Positive `mae_skill_vs_random_walk_pct` means lower MAE than random walk. Positive `rmse_skill_vs_random_walk_pct` means lower RMSE. Their confidence intervals are separate. An interval crossing zero is inconclusive. Random-walk directional accuracy is N/A. Calibration bands are empirical scenario ranges, not guaranteed coverage intervals.
