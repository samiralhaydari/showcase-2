# MYR SupplyLens V2.1

**Predict. Plan. Procure. Profit.**

V2.1 upgrades the existing Streamlit manufacturing decision-support prototype with a benchmark-aware forecasting engine. CNN-LSTM remains available, but the platform selects a forecasting method separately for each currency and supported horizon using a dedicated chronological selection partition.

## Quick start with Google Colab

1. Download `MYR_SupplyLens_V2_1_Patch.zip` and `MYR_SupplyLens_V2_1_Colab.ipynb`.
2. Upload both files to the root of Google Drive (`My Drive`).
3. Open the notebook in Colab and run its cells in order.
4. The notebook overlays source files into `/content/drive/MyDrive/MYR_SupplyLens` after backing up existing source files. It does not extract into or delete `data/` or `artifacts/`.
5. Run the smoke tests and data inspection.
6. Train USD, CNY and IDR V2.1 bundles. Existing V2 and V1 model artifacts are retained.
7. Launch Streamlit from the notebook.

## V2.1 evaluation design

- Chronological split: 60% train, 10% early-stopping validation, 10% method selection, 5% calibration, 15% untouched final test.
- Nine features: log rate, 1/5/10/20-session momentum, 5/20/60-session volatility, and 20-session log-rate range.
- Five forecasting methods: CNN-LSTM, random walk, historical drift (252), recent drift (20), and ARIMA(1,1,1).
- Selection is per currency and displayed horizon. The selection partition is not used for gradient updates or early stopping. Origins are spaced by the target horizon within each selection comparison.
- Selection rule: lowest selection-partition MAE; RMSE and absolute-error standard deviation are tie-breakers. If fewer than eight usable origins are available, random walk is selected as a conservative fallback.
- Final test: selection policy is frozen first; test origins use common 63-observation spacing so target blocks do not overlap across displayed horizons.
- Metrics: MAE and RMSE are reported separately, with separate paired-bootstrap skill scores and confidence intervals against random walk.
- Random walk predicts no directional change. Its directional accuracy is N/A, not 0%.
- Calibration: method-specific empirical 80% and 95% absolute log-error quantiles from a separate chronological calibration partition. Calibration windows may overlap, so these are descriptive scenario bands, not formal conformal guarantees.

## Versioned outputs and preservation

V2.1 outputs use `_v2_1_` filenames, for example:

- `artifacts/models/cnn_lstm_v2_1_USD.keras`
- `artifacts/scalers/scaler_X_v2_1_USD.joblib`
- `artifacts/scalers/scaler_y_v2_1_USD.joblib`
- `artifacts/metadata/metadata_v2_1_USD.json`
- `artifacts/metadata/metrics_v2_1_USD.csv`
- `artifacts/metadata/selection_metrics_v2_1_USD.csv`
- `artifacts/metadata/selection_policy_v2_1_USD.csv`
- `artifacts/metadata/calibration_v2_1_USD.csv`

V2.1 is loaded first, then V2, then V1. V2/V1 models, scalers, raw data, SQLite records, saved supplier details and invoice exposures are not removed or overwritten by the patch. A fresh V2.1 training run creates a new model bundle.

## Colab package note

The 2.1.1 dependency hotfix pins a mutually compatible NumPy/SciPy/scikit-learn/pandas stack for Python 3.13 Colab runtimes. The notebook automatically restarts the runtime if package versions changed, preventing stale modules from remaining loaded in memory. TensorFlow is not reinstalled.

`requirements-colab.txt` intentionally does not reinstall TensorFlow. Colab's installed TensorFlow is used. The NumPy/SciPy ranges are constrained to avoid the incompatible NumPy/SciPy combination that can break TensorFlow imports. Check the runtime package versions before manually reinstalling TensorFlow.

## Data and limitations

Bank Negara Malaysia daily middle reference rates via data.gov.my, CC BY 4.0. Quotation is MYR per one unit of foreign currency. Central-bank reference rates are not necessarily transaction rates offered to SMEs. Small independent test-origin counts mean that results must be interpreted cautiously. Forecasts are estimates, not financial advice or guaranteed outcomes.
