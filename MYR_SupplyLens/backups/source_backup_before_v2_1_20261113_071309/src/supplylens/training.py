import json
from datetime import datetime, timezone
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .config import (
    LOOKBACK, FORECAST_HORIZON, FEATURE_COLUMNS, MODEL_VERSION,
    MODEL_DIR, SCALER_DIR, METADATA_DIR,
)
from .data import load_exchange_rates, get_currency_series
from .features import make_supervised_samples
from .model import build_cnn_lstm
from .evaluation import evaluate_walk_forward, calibrate_forecast_intervals


def scale_X(X, scaler):
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)


def _versioned_paths(currency):
    return {
        "model": MODEL_DIR / f"cnn_lstm_{MODEL_VERSION}_{currency}.keras",
        "scaler_X": SCALER_DIR / f"scaler_X_{MODEL_VERSION}_{currency}.joblib",
        "scaler_y": SCALER_DIR / f"scaler_y_{MODEL_VERSION}_{currency}.joblib",
        "metadata": METADATA_DIR / f"metadata_{MODEL_VERSION}_{currency}.json",
        "metrics": METADATA_DIR / f"metrics_{MODEL_VERSION}_{currency}.csv",
        "details": METADATA_DIR / f"walk_forward_details_{MODEL_VERSION}_{currency}.csv",
        "calibration": METADATA_DIR / f"calibration_{MODEL_VERSION}_{currency}.csv",
    }


def train_currency_model(currency="USD", epochs=40, batch_size=32,
                         lookback=LOOKBACK, horizon=FORECAST_HORIZON,
                         force_download=False, include_arima=True):
    currency = currency.upper()
    rates = load_exchange_rates(force_download=force_download)
    series = get_currency_series(rates, currency)
    data = make_supervised_samples(series, lookback=lookback, horizon=horizon)

    train = data["train"]
    val = data["validation"]
    calibration = data["calibration"]
    test = data["test"]
    for label, part in [("train", train), ("validation", val),
                        ("calibration", calibration), ("test", test)]:
        if len(part["X"]) == 0:
            raise ValueError(f"{label} partition is empty. More historical observations are required.")

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    scaler_X.fit(train["X"].reshape(-1, train["X"].shape[-1]))
    scaler_y.fit(train["y"].reshape(-1, 1))

    X_train = scale_X(train["X"], scaler_X)
    X_val = scale_X(val["X"], scaler_X)
    X_cal = scale_X(calibration["X"], scaler_X)
    X_test = scale_X(test["X"], scaler_X)
    y_train = scaler_y.transform(train["y"].reshape(-1, 1)).reshape(train["y"].shape)
    y_val = scaler_y.transform(val["y"].reshape(-1, 1)).reshape(val["y"].shape)

    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = build_cnn_lstm(lookback, len(FEATURE_COLUMNS), horizon)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    calibration_table = calibrate_forecast_intervals(
        model,
        X_cal,
        scaler_y,
        calibration["origins"],
        data["aligned_rates"],
        horizon=horizon,
    )
    metrics, details = evaluate_walk_forward(
        model,
        X_test,
        scaler_y,
        test["origins"],
        data["aligned_rates"],
        data["aligned_dates"],
        currency=currency,
        include_arima=include_arima,
    )

    for folder in [MODEL_DIR, SCALER_DIR, METADATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    paths = _versioned_paths(currency)

    model.save(paths["model"])
    joblib.dump(scaler_X, paths["scaler_X"])
    joblib.dump(scaler_y, paths["scaler_y"])
    metrics.to_csv(paths["metrics"], index=False)
    details.to_csv(paths["details"], index=False)
    calibration_table.to_csv(paths["calibration"], index=False)

    cnn_rows = metrics[metrics["model"] == "CNN-LSTM"]
    beats = int((cnn_rows["skill_vs_random_walk_pct"] > 0).sum())
    metadata = {
        "model_version": MODEL_VERSION,
        "currency": currency,
        "rate_convention": f"MYR per 1 {currency}",
        "source": "Bank Negara Malaysia via data.gov.my",
        "source_url": "https://data.gov.my/data-catalogue/exchangerates_daily_1200",
        "rate_type": "middle",
        "lookback": lookback,
        "forecast_horizon": horizon,
        "feature_columns": FEATURE_COLUMNS,
        "architecture": (
            "Conv1D(32, kernel=3, causal) -> MaxPooling1D(2) -> LSTM(32) "
            "-> Dropout(0.20) -> Dense(32, ReLU) -> Dense(horizon)"
        ),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_observation_date": str(series.index[-1].date()),
        "latest_rate": float(series.iloc[-1]),
        "train_samples": int(len(train["X"])),
        "validation_samples": int(len(val["X"])),
        "calibration_samples": int(len(calibration["X"])),
        "test_candidate_samples": int(len(test["X"])),
        "walk_forward_origins": int(details["origin_index"].nunique()),
        "walk_forward_spacing": int(63),
        "walk_forward_method": (
            "Fixed trained model; rolling forecast origins; common 63-observation "
            "spacing so target blocks do not overlap across reported horizons."
        ),
        "benchmarks": [
            "Random walk", "Historical drift (252)", "Recent drift (20)",
            "ARIMA(1,1,1)" if include_arima else "ARIMA disabled",
        ],
        "cnn_lstm_horizons_beating_random_walk": beats,
        "cnn_lstm_horizons_evaluated": int(len(cnn_rows)),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "epochs_completed": int(len(history.history["loss"])),
        "calibration_method": (
            "Empirical absolute log-error quantiles from a separate calibration split. "
            "Calibration origins overlap; bands are descriptive and are not a formal "
            "independent-sample conformal coverage guarantee."
        ),
        "calibration_min_samples": 30,
        "arima_included": bool(include_arima),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(paths["model"]),
        "metadata": metadata,
        "metrics": metrics,
        "calibration": calibration_table,
        "history": history.history,
    }
