# MYR SupplyLens V2

**Predict. Plan. Procure. Profit.**

V2 upgrades the existing Streamlit and CNN-LSTM prototype with a stricter model-evaluation protocol and business-facing uncertainty information.

## Quick start

1. Upload `MYR_SupplyLens_V2_Patch.zip` and `MYR_SupplyLens_V2_Colab.ipynb` to the root of Google Drive.
2. Open the V2 notebook in Colab.
3. Run the cells in order. The notebook patches `/content/drive/MyDrive/MYR_SupplyLens` in place.
4. Train USD, CNY and IDR.
5. Inspect benchmark metrics, then launch Streamlit.

## Evaluation design

- Chronological split: 65% train, 10% validation, 10% calibration, 15% test.
- Nine features: log rate, 1/5/10/20-session momentum, 5/20/60-session volatility, 20-session log-rate range.
- Walk-forward test: fixed trained model, advancing historical input window, common origins spaced 63 observations apart.
- Benchmarks: random walk, 252-observation historical drift, 20-observation recent drift and ARIMA(1,1,1).
- Forecast uncertainty: empirical 80% and 95% absolute log-error quantiles from a separate calibration split.

## Important interpretation

Positive `skill_vs_random_walk_pct` means the model has lower MAE than random walk. Negative means random walk performs better. The independent origin count is deliberately smaller than the old overlapping sample count; always report it. Calibration windows overlap, so the intervals are descriptive empirical bands, not guaranteed coverage intervals. ARIMA may fail at individual origins; its `n_origins` count is displayed.

## Versioned outputs

V2 artifacts use `_v2_` names. V1 models and scalers are retained. Forecast loading prefers V2 and falls back to V1 until a V2 currency model is trained.

## Data

Bank Negara Malaysia daily middle reference rates via data.gov.my, CC BY 4.0. The quotation is MYR per one unit of foreign currency. Central-bank reference rates are not necessarily transaction rates offered by banks.
