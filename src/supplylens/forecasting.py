import json
import os

import joblib
import numpy as np
import pandas as pd

from .config import (
    LOOKBACK, FORECAST_HORIZON, FEATURE_COLUMNS, MODEL_VERSION,
    LEGACY_MODEL_VERSIONS, MODEL_DIR, SCALER_DIR, METADATA_DIR, FORECAST_DIR,
    DISPLAY_HORIZONS,
)
from .features import make_feature_frame
from .evaluation import benchmark_forecast_paths


def _bundle_paths(currency, version):
    return {
        "version": version,
        "model": MODEL_DIR / f"cnn_lstm_{version}_{currency}.keras" if version != "v1" else MODEL_DIR / f"cnn_lstm_{currency}.keras",
        "scaler_X": SCALER_DIR / f"scaler_X_{version}_{currency}.joblib" if version != "v1" else SCALER_DIR / f"scaler_X_{currency}.joblib",
        "scaler_y": SCALER_DIR / f"scaler_y_{version}_{currency}.joblib" if version != "v1" else SCALER_DIR / f"scaler_y_{currency}.joblib",
        "metadata": METADATA_DIR / f"metadata_{version}_{currency}.json" if version != "v1" else METADATA_DIR / f"metadata_{currency}.json",
        "calibration": METADATA_DIR / f"calibration_{version}_{currency}.csv" if version != "v1" else None,
        "selection": METADATA_DIR / f"selection_policy_{version}_{currency}.csv" if version == MODEL_VERSION else None,
    }


def _load_network(path):
    """Load the trained network for inference.

    The NumPy evaluator is tried first because it needs no TensorFlow (about
    700 MB less memory, which matters on free hosting). Anything it does not
    recognise falls back to Keras, as does SUPPLYLENS_USE_KERAS=1.
    """
    if os.environ.get("SUPPLYLENS_USE_KERAS", "") != "1":
        try:
            from .lite_inference import LiteModel
            return LiteModel(path)
        except Exception:
            pass
    from tensorflow import keras
    return keras.models.load_model(path)


def load_model_bundle(currency):
    currency = currency.upper()
    versions = [MODEL_VERSION] + [v for v in LEGACY_MODEL_VERSIONS if v != MODEL_VERSION]
    for version in versions:
        item = _bundle_paths(currency, version)
        required = [item["model"], item["scaler_X"], item["scaler_y"], item["metadata"]]
        if not all(path.exists() for path in required):
            continue

        model = _load_network(item["model"])
        scaler_X = joblib.load(item["scaler_X"])
        scaler_y = joblib.load(item["scaler_y"])
        metadata = json.loads(item["metadata"].read_text(encoding="utf-8"))
        calibration = None
        if item["calibration"] is not None and item["calibration"].exists():
            calibration = pd.read_csv(item["calibration"])
        selection = None
        if item["selection"] is not None and item["selection"].exists():
            selection = pd.read_csv(item["selection"])
        return model, scaler_X, scaler_y, metadata, calibration, selection

    raise FileNotFoundError(
        f"No trained V2.1, V2 or V1 model bundle found for {currency}. "
        "Train the currency model in the Colab notebook."
    )


def _calibration_quantile(calibration, model_name, horizon, level):
    if calibration is None or calibration.empty:
        return np.nan
    table = calibration
    if "model" in table.columns:
        table = table[table["model"] == model_name]
    else:
        # Legacy V2 calibration tables contain CNN-LSTM only.
        if model_name != "CNN-LSTM":
            return np.nan
    table = table[table["horizon"] == int(horizon)]
    if table.empty:
        return np.nan
    column = f"q{level}_abs_log_error"
    value = table.iloc[0].get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def bundle_signature(currency):
    """Identify the bundle that would be loaded, without loading it.

    Returns (version, ((filename, mtime_ns), ...)). The UI uses this as a cache
    key, so a retrained model is picked up without restarting the app. Raises
    FileNotFoundError when no complete bundle exists.
    """
    currency = currency.upper()
    versions = [MODEL_VERSION] + [v for v in LEGACY_MODEL_VERSIONS if v != MODEL_VERSION]
    for version in versions:
        item = _bundle_paths(currency, version)
        required = [item["model"], item["scaler_X"], item["scaler_y"], item["metadata"]]
        if not all(path.exists() for path in required):
            continue
        files = required + [
            path for path in (item["calibration"], item["selection"])
            if path is not None and path.exists()
        ]
        return version, tuple((path.name, path.stat().st_mtime_ns) for path in files)
    raise FileNotFoundError(
        f"No trained V2.1, V2 or V1 model bundle found for {currency}. "
        "Train the currency model in the Colab notebook."
    )


def forecast_currency(rate_series, currency, bundle=None):
    """Forecast one currency.

    ``bundle`` is an already-loaded result of ``load_model_bundle``. The Streamlit
    app passes a cached one so the Keras model is not re-read from disk on every
    interaction; scripts and tests can omit it.
    """
    currency = currency.upper()
    if bundle is None:
        bundle = load_model_bundle(currency)
    model, scaler_X, scaler_y, metadata, calibration, selection = bundle
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
    cnn_rates = latest_rate * np.exp(np.cumsum(pred_returns))
    log_history = np.log(latest.to_numpy(dtype=float)[-(756 + 1):])
    paths = benchmark_forecast_paths(
        log_history, latest_rate, horizon, cnn_path=cnn_rates,
        include_arima=bool(metadata.get("arima_included", True)),
    )
    if "CNN-LSTM" not in paths:
        paths["CNN-LSTM"] = cnn_rates

    forecast_dates = pd.bdate_range(
        start=latest_date + pd.Timedelta(days=1), periods=horizon
    )
    forecast = pd.DataFrame({
        "date": forecast_dates,
        "forecast_rate": cnn_rates,
        "predicted_log_return": pred_returns,
        "cumulative_log_return": np.cumsum(pred_returns),
        "currency": currency,
    })

    path_columns = {
        "CNN-LSTM": "forecast_cnn_lstm",
        "Random walk": "forecast_random_walk",
        "Historical drift (252)": "forecast_historical_drift_252",
        "Recent drift (20)": "forecast_recent_drift_20",
        "ARIMA(1,1,1)": "forecast_arima_111",
    }
    for method, column in path_columns.items():
        forecast[column] = paths.get(method, np.full(horizon, np.nan, dtype=float))

    # Preserve legacy CNN-LSTM intervals for visual reference.
    for level in ("80", "95"):
        q = np.asarray([
            _calibration_quantile(calibration, "CNN-LSTM", step, level)
            for step in range(1, horizon + 1)
        ], dtype=float)
        forecast[f"lower_{level}"] = cnn_rates * np.exp(-q)
        forecast[f"upper_{level}"] = cnn_rates * np.exp(q)

    forecast["selected_model"] = None
    forecast["selected_forecast_rate"] = np.nan
    forecast["selection_reason"] = None
    for level in ("80", "95"):
        forecast[f"selected_lower_{level}"] = np.nan
        forecast[f"selected_upper_{level}"] = np.nan

    policy_lookup = {}
    if selection is not None and not selection.empty:
        policy_lookup = {
            int(row["horizon"]): row.to_dict()
            for _, row in selection.iterrows()
        }

    for h in DISPLAY_HORIZONS:
        if h > horizon:
            continue
        policy = policy_lookup.get(int(h), {})
        selected_method = str(policy.get("selected_model", "CNN-LSTM"))
        reason = str(policy.get("selection_reason", "Legacy bundle: CNN-LSTM default."))
        if selected_method not in paths:
            selected_method = "CNN-LSTM"
            reason = "Selected method unavailable at inference; CNN-LSTM fallback used."

        selected_rate = float(paths[selected_method][h - 1])
        forecast.loc[h - 1, "selected_model"] = selected_method
        forecast.loc[h - 1, "selected_forecast_rate"] = selected_rate
        forecast.loc[h - 1, "selection_reason"] = reason

        for level in ("80", "95"):
            q = _calibration_quantile(calibration, selected_method, h, level)
            if np.isfinite(q):
                forecast.loc[h - 1, f"selected_lower_{level}"] = selected_rate * np.exp(-q)
                forecast.loc[h - 1, f"selected_upper_{level}"] = selected_rate * np.exp(q)

    metadata = dict(metadata)
    metadata["selection_mode"] = (
        "validation-based per-currency/per-horizon selection"
        if policy_lookup else "CNN-LSTM legacy fallback; no selection policy artifact"
    )

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    version = str(metadata.get("model_version", "legacy"))
    forecast.to_csv(FORECAST_DIR / f"{currency}_forecast_{version}.csv", index=False)
    # Keep the previous latest-forecast filename for integrations built against V2.
    forecast.to_csv(FORECAST_DIR / f"{currency}_forecast.csv", index=False)
    return forecast, metadata
