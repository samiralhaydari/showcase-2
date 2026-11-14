import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from .config import (
    LOOKBACK, FORECAST_HORIZON, FEATURE_COLUMNS, MODEL_VERSION,
    MODEL_DIR, SCALER_DIR, METADATA_DIR, FORECAST_DIR,
)
from .features import make_feature_frame


def load_model_bundle(currency):
    currency = currency.upper()
    # Prefer V2, but retain backward compatibility with existing V1 model artifacts.
    candidates = [
        {
            "version": MODEL_VERSION,
            "model": MODEL_DIR / f"cnn_lstm_{MODEL_VERSION}_{currency}.keras",
            "scaler_X": SCALER_DIR / f"scaler_X_{MODEL_VERSION}_{currency}.joblib",
            "scaler_y": SCALER_DIR / f"scaler_y_{MODEL_VERSION}_{currency}.joblib",
            "metadata": METADATA_DIR / f"metadata_{MODEL_VERSION}_{currency}.json",
            "calibration": METADATA_DIR / f"calibration_{MODEL_VERSION}_{currency}.csv",
        },
        {
            "version": "v1",
            "model": MODEL_DIR / f"cnn_lstm_{currency}.keras",
            "scaler_X": SCALER_DIR / f"scaler_X_{currency}.joblib",
            "scaler_y": SCALER_DIR / f"scaler_y_{currency}.joblib",
            "metadata": METADATA_DIR / f"metadata_{currency}.json",
            "calibration": None,
        },
    ]
    for item in candidates:
        required = [item["model"], item["scaler_X"], item["scaler_y"], item["metadata"]]
        if all(path.exists() for path in required):
            model = keras.models.load_model(item["model"])
            scaler_X = joblib.load(item["scaler_X"])
            scaler_y = joblib.load(item["scaler_y"])
            metadata = json.loads(item["metadata"].read_text(encoding="utf-8"))
            calibration = None
            if item["calibration"] is not None and item["calibration"].exists():
                calibration = pd.read_csv(item["calibration"])
            return model, scaler_X, scaler_y, metadata, calibration

    raise FileNotFoundError(
        f"No trained V2 or V1 model bundle found for {currency}. "
        "Train the currency model in the Colab notebook."
    )


def forecast_currency(rate_series, currency):
    currency = currency.upper()
    model, scaler_X, scaler_y, metadata, calibration = load_model_bundle(currency)
    features = make_feature_frame(rate_series).dropna()
    lookback = int(metadata.get("lookback", LOOKBACK))
    horizon = int(metadata.get("forecast_horizon", FORECAST_HORIZON))
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)

    if len(features) < lookback:
        raise ValueError(f"Need at least {lookback} usable feature observations.")
    missing_features = [c for c in feature_columns if c not in features.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    window = features.iloc[-lookback:][feature_columns].to_numpy(dtype=np.float32)
    X = scaler_X.transform(window.reshape(-1, window.shape[-1])).reshape(
        1, lookback, window.shape[-1]
    )
    pred_scaled = model.predict(X, verbose=0)
    pred_returns = scaler_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).reshape(-1)

    latest = rate_series.dropna()
    latest_rate = float(latest.iloc[-1])
    latest_date = pd.Timestamp(latest.index[-1])
    forecast_rates = latest_rate * np.exp(np.cumsum(pred_returns))
    forecast_dates = pd.bdate_range(
        start=latest_date + pd.Timedelta(days=1), periods=horizon
    )

    forecast = pd.DataFrame({
        "date": forecast_dates,
        "forecast_rate": forecast_rates,
        "predicted_log_return": pred_returns,
        "cumulative_log_return": np.cumsum(pred_returns),
        "currency": currency,
    })

    # Attach empirical calibrated uncertainty bands when V2 calibration is present.
    if calibration is not None and not calibration.empty:
        cal = calibration.set_index("horizon")
        for level, column in [("80", "q80_abs_log_error"), ("95", "q95_abs_log_error")]:
            quantiles = np.array([
                float(cal.loc[i, column]) if i in cal.index and pd.notna(cal.loc[i, column]) else np.nan
                for i in range(1, horizon + 1)
            ])
            forecast[f"lower_{level}"] = forecast_rates * np.exp(-quantiles)
            forecast[f"upper_{level}"] = forecast_rates * np.exp(quantiles)

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(FORECAST_DIR / f"{currency}_forecast.csv", index=False)
    return forecast, metadata
