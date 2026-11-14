import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    DISPLAY_HORIZONS,
    WALK_FORWARD_SPACING,
    ARIMA_MAX_HISTORY,
    CALIBRATION_MIN_SAMPLES,
)


def _inverse_return_predictions(model, X_scaled, scaler_y):
    pred_scaled = model.predict(X_scaled, verbose=0)
    pred_returns = scaler_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).reshape(pred_scaled.shape)
    return pred_returns


def calibrate_forecast_intervals(model, X_cal_scaled, scaler_y, calibration_origins,
                                 aligned_rates, horizon=63,
                                 min_samples=CALIBRATION_MIN_SAMPLES):
    """Calibrate descriptive empirical log-error bands on a held-out calibration split.

    All calibration origins are used to improve quantile stability. Their forecast
    windows overlap, so the resulting bands are empirical uncertainty estimates,
    not a formal independent-sample conformal coverage guarantee.
    """
    if len(calibration_origins) == 0:
        raise ValueError("No calibration samples are available.")

    pred_returns = _inverse_return_predictions(model, X_cal_scaled, scaler_y)
    origins = np.asarray(calibration_origins, dtype=int)
    base_rates = aligned_rates[origins]
    predicted_paths = base_rates[:, None] * np.exp(np.cumsum(pred_returns, axis=1))

    rows = []
    for step in range(1, horizon + 1):
        valid = origins + step < len(aligned_rates)
        if not np.any(valid):
            continue
        actual = aligned_rates[origins[valid] + step]
        predicted = predicted_paths[valid, step - 1]
        abs_log_error = np.abs(np.log(actual) - np.log(predicted))
        count = int(len(abs_log_error))
        if count >= min_samples:
            q80 = float(np.quantile(abs_log_error, 0.80, method="higher"))
            q95 = float(np.quantile(abs_log_error, 0.95, method="higher"))
        else:
            q80, q95 = np.nan, np.nan
        rows.append({
            "horizon": step,
            "q80_abs_log_error": q80,
            "q95_abs_log_error": q95,
            "calibration_samples": count,
            "interval_available": bool(count >= min_samples),
        })
    return pd.DataFrame(rows)


def _fit_arima_forecast(log_history, horizon):
    """Fit ARIMA(1,1,1) using history available at the forecast origin only."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(
                log_history,
                order=(1, 1, 1),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            forecast_log = np.asarray(fitted.forecast(steps=horizon), dtype=float)
        if len(forecast_log) != horizon or not np.all(np.isfinite(forecast_log)):
            return None
        return np.exp(forecast_log)
    except Exception:
        return None


def _metric_row(currency, horizon, model_name, origin_dates, actual, predicted, base):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    base = np.asarray(base, dtype=float)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)
    direction = float(
        np.mean(np.sign(actual - base) == np.sign(predicted - base)) * 100.0
    )
    return {
        "currency": currency,
        "horizon": int(horizon),
        "model": model_name,
        "n_origins": int(len(actual)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "directional_accuracy_pct": direction,
    }


def evaluate_walk_forward(model, X_test_scaled, scaler_y, test_origins,
                          aligned_rates, aligned_dates, currency,
                          horizons=DISPLAY_HORIZONS,
                          spacing=WALK_FORWARD_SPACING,
                          include_arima=True):
    """Evaluate a fixed trained model at rolling origins with non-overlapping blocks.

    The model weights remain fixed. At each later origin, the input window advances
    using only observations available by that origin. Origins are spaced by 63
    observations so target blocks do not overlap even at the longest horizon.
    """
    if len(test_origins) == 0:
        raise ValueError("No test samples were created.")

    origins = np.asarray(test_origins, dtype=int)
    X_scaled = np.asarray(X_test_scaled)
    pred_returns = _inverse_return_predictions(model, X_scaled, scaler_y)
    origin_to_row = {int(origin): i for i, origin in enumerate(origins)}

    # Select one common set of forecast origins for every horizon.
    selected_origins = origins[::spacing]
    if len(selected_origins) == 0:
        raise ValueError("No walk-forward origins remain after applying spacing.")

    # Precompute ARIMA paths once per origin; this is the most expensive benchmark.
    arima_paths = {}
    if include_arima:
        for origin in selected_origins:
            start = max(0, int(origin) - ARIMA_MAX_HISTORY)
            history = np.log(aligned_rates[start:int(origin) + 1])
            arima_paths[int(origin)] = _fit_arima_forecast(history, pred_returns.shape[1])

    detail_rows = []
    for origin in selected_origins:
        row_idx = origin_to_row[int(origin)]
        base_rate = float(aligned_rates[origin])
        actual_path = aligned_rates[origin + 1:origin + pred_returns.shape[1] + 1]
        if len(actual_path) < max(horizons):
            continue

        cnn_path = base_rate * np.exp(np.cumsum(pred_returns[row_idx]))
        history_start = max(0, int(origin) - 252)
        history_log = np.log(aligned_rates[history_start:int(origin) + 1])
        historical_returns = np.diff(history_log)
        drift_252 = float(np.mean(historical_returns[-252:])) if len(historical_returns) else 0.0
        recent_returns = np.diff(np.log(aligned_rates[max(0, int(origin) - 20):int(origin) + 1]))
        drift_20 = float(np.mean(recent_returns[-20:])) if len(recent_returns) else 0.0
        steps = np.arange(1, pred_returns.shape[1] + 1)
        drift_252_path = base_rate * np.exp(drift_252 * steps)
        drift_20_path = base_rate * np.exp(drift_20 * steps)
        random_walk_path = np.full(pred_returns.shape[1], base_rate, dtype=float)
        arima_path = arima_paths.get(int(origin)) if include_arima else None

        paths = {
            "CNN-LSTM": cnn_path,
            "Random walk": random_walk_path,
            "Historical drift (252)": drift_252_path,
            "Recent drift (20)": drift_20_path,
        }
        if arima_path is not None:
            paths["ARIMA(1,1,1)"] = arima_path

        origin_date = pd.Timestamp(aligned_dates[origin]).date().isoformat()
        for h in horizons:
            actual = float(actual_path[h - 1])
            for model_name, path in paths.items():
                predicted = float(path[h - 1])
                detail_rows.append({
                    "currency": currency,
                    "origin_index": int(origin),
                    "origin_date": origin_date,
                    "horizon": int(h),
                    "model": model_name,
                    "base_rate": base_rate,
                    "actual_rate": actual,
                    "forecast_rate": predicted,
                    "absolute_error": abs(actual - predicted),
                    "absolute_percentage_error": abs((actual - predicted) / actual) * 100.0,
                    "direction_correct": bool(np.sign(actual - base_rate) == np.sign(predicted - base_rate)),
                })

    details = pd.DataFrame(detail_rows)
    if details.empty:
        raise ValueError("Walk-forward evaluation produced no result rows.")

    metric_rows = []
    for (horizon, model_name), group in details.groupby(["horizon", "model"], sort=True):
        metric_rows.append(_metric_row(
            currency,
            horizon,
            model_name,
            group["origin_date"].tolist(),
            group["actual_rate"].to_numpy(),
            group["forecast_rate"].to_numpy(),
            group["base_rate"].to_numpy(),
        ))
    metrics = pd.DataFrame(metric_rows)

    # Paired skill scores: compare each model with random walk on identical origins.
    rw = details[details["model"] == "Random walk"][[
        "horizon", "origin_index", "absolute_error"
    ]].rename(columns={"absolute_error": "random_walk_absolute_error"})
    paired = details.merge(rw, on=["horizon", "origin_index"], how="left")
    skill_rows = []
    rng = np.random.default_rng(42)
    for (horizon, model_name), group in paired.groupby(["horizon", "model"]):
        rw_errors = group["random_walk_absolute_error"].to_numpy(dtype=float)
        model_errors = group["absolute_error"].to_numpy(dtype=float)
        random_walk_mae = float(np.mean(rw_errors))
        model_mae = float(np.mean(model_errors))
        skill_value = (
            (random_walk_mae - model_mae) / random_walk_mae * 100.0
            if random_walk_mae else np.nan
        )

        # Paired bootstrap resamples the same independent forecast origins for
        # the model and random-walk benchmark.
        if model_name == "Random walk":
            ci_low, ci_high = 0.0, 0.0
        elif len(group) >= 2 and random_walk_mae > 0:
            sample_indices = rng.integers(0, len(group), size=(2000, len(group)))
            boot_rw = rw_errors[sample_indices].mean(axis=1)
            boot_model = model_errors[sample_indices].mean(axis=1)
            boot_skill = (boot_rw - boot_model) / np.maximum(boot_rw, 1e-15) * 100.0
            ci_low, ci_high = np.quantile(boot_skill, [0.025, 0.975]).tolist()
        else:
            ci_low, ci_high = np.nan, np.nan

        skill_rows.append({
            "horizon": int(horizon),
            "model": model_name,
            "skill_vs_random_walk_pct": float(skill_value),
            "skill_ci_lower_pct": float(ci_low),
            "skill_ci_upper_pct": float(ci_high),
        })
    skill = pd.DataFrame(skill_rows)
    metrics = metrics.merge(skill, on=["horizon", "model"], how="left")
    metrics["n_origins"] = metrics["n_origins"].astype(int)
    metrics = metrics.sort_values(["horizon", "model"]).reset_index(drop=True)
    return metrics, details
