import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    DISPLAY_HORIZONS,
    WALK_FORWARD_SPACING,
    ARIMA_MAX_HISTORY,
    CALIBRATION_MIN_SAMPLES,
    CALIBRATION_ORIGIN_STRIDE,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    MIN_SELECTION_ORIGINS,
)


def _inverse_return_predictions(model, X_scaled, scaler_y):
    pred_scaled = model.predict(X_scaled, verbose=0)
    pred_returns = scaler_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).reshape(pred_scaled.shape)
    return pred_returns


def _fit_arima_forecast(log_history, horizon):
    """Fit ARIMA(1,1,1) using only observations available at the forecast origin."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(
                np.asarray(log_history, dtype=float),
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


def benchmark_forecast_paths(log_history, base_rate, horizon, cnn_path=None, include_arima=True):
    """Return comparable forecast paths for CNN-LSTM and the four V2.1 benchmarks."""
    log_history = np.asarray(log_history, dtype=float)
    if len(log_history) < 2:
        raise ValueError("At least two historical observations are required for benchmarks.")

    returns = np.diff(log_history)
    drift_252 = float(np.mean(returns[-252:])) if len(returns) else 0.0
    drift_20 = float(np.mean(returns[-20:])) if len(returns) else 0.0
    steps = np.arange(1, horizon + 1, dtype=float)

    paths = {
        "Random walk": np.full(horizon, float(base_rate), dtype=float),
        "Historical drift (252)": float(base_rate) * np.exp(drift_252 * steps),
        "Recent drift (20)": float(base_rate) * np.exp(drift_20 * steps),
    }
    if cnn_path is not None:
        paths["CNN-LSTM"] = np.asarray(cnn_path, dtype=float)
    if include_arima:
        arima_path = _fit_arima_forecast(log_history[-(ARIMA_MAX_HISTORY + 1):], horizon)
        if arima_path is not None:
            paths["ARIMA(1,1,1)"] = arima_path
    return paths


def _metric_row(currency, horizon, model_name, actual, predicted, base):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    base = np.asarray(base, dtype=float)
    absolute_errors = np.abs(actual - predicted)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)

    # A random-walk forecast is exactly the latest observed value. It makes no
    # directional prediction, so reporting 0% or 100% directional accuracy is wrong.
    if model_name == "Random walk":
        direction = np.nan
        direction_status = "N/A — predicts no directional change"
    else:
        direction = float(
            np.mean(np.sign(actual - base) == np.sign(predicted - base)) * 100.0
        )
        direction_status = "Calculated"

    return {
        "currency": currency,
        "horizon": int(horizon),
        "model": model_name,
        "n_origins": int(len(actual)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "absolute_error_std": float(np.std(absolute_errors, ddof=1)) if len(absolute_errors) > 1 else np.nan,
        "directional_accuracy_pct": direction,
        "directional_status": direction_status,
    }


def _paired_skill_and_ci(rw_errors, model_errors, model_name, horizon,
                         bootstrap_samples=BOOTSTRAP_SAMPLES):
    rw_errors = np.asarray(rw_errors, dtype=float)
    model_errors = np.asarray(model_errors, dtype=float)
    valid = np.isfinite(rw_errors) & np.isfinite(model_errors)
    rw_errors, model_errors = rw_errors[valid], model_errors[valid]

    if model_name == "Random walk":
        return {
            "mae_skill_vs_random_walk_pct": 0.0,
            "mae_skill_ci_lower_pct": 0.0,
            "mae_skill_ci_upper_pct": 0.0,
            "rmse_skill_vs_random_walk_pct": 0.0,
            "rmse_skill_ci_lower_pct": 0.0,
            "rmse_skill_ci_upper_pct": 0.0,
        }

    rw_mae = float(np.mean(rw_errors)) if len(rw_errors) else np.nan
    model_mae = float(np.mean(model_errors)) if len(model_errors) else np.nan
    rw_rmse = float(np.sqrt(np.mean(np.square(rw_errors)))) if len(rw_errors) else np.nan
    model_rmse = float(np.sqrt(np.mean(np.square(model_errors)))) if len(model_errors) else np.nan

    mae_skill = 100.0 * (1.0 - model_mae / rw_mae) if rw_mae > 0 else np.nan
    rmse_skill = 100.0 * (1.0 - model_rmse / rw_rmse) if rw_rmse > 0 else np.nan

    if len(rw_errors) < 2 or not np.isfinite(rw_mae) or rw_mae <= 0 or not np.isfinite(rw_rmse) or rw_rmse <= 0:
        mae_low = mae_high = rmse_low = rmse_high = np.nan
    else:
        # Deterministic seed per horizon/model makes the evaluation reproducible.
        model_seed = sum(ord(ch) for ch in model_name)
        rng = np.random.default_rng(BOOTSTRAP_SEED + int(horizon) * 100 + model_seed)
        sample_indices = rng.integers(0, len(rw_errors), size=(bootstrap_samples, len(rw_errors)))
        boot_rw = rw_errors[sample_indices]
        boot_model = model_errors[sample_indices]

        boot_rw_mae = np.mean(boot_rw, axis=1)
        boot_model_mae = np.mean(boot_model, axis=1)
        boot_mae_skill = (1.0 - boot_model_mae / np.maximum(boot_rw_mae, 1e-15)) * 100.0

        boot_rw_rmse = np.sqrt(np.mean(np.square(boot_rw), axis=1))
        boot_model_rmse = np.sqrt(np.mean(np.square(boot_model), axis=1))
        boot_rmse_skill = (1.0 - boot_model_rmse / np.maximum(boot_rw_rmse, 1e-15)) * 100.0

        mae_low, mae_high = np.quantile(boot_mae_skill, [0.025, 0.975]).tolist()
        rmse_low, rmse_high = np.quantile(boot_rmse_skill, [0.025, 0.975]).tolist()

    return {
        "mae_skill_vs_random_walk_pct": float(mae_skill),
        "mae_skill_ci_lower_pct": float(mae_low),
        "mae_skill_ci_upper_pct": float(mae_high),
        "rmse_skill_vs_random_walk_pct": float(rmse_skill),
        "rmse_skill_ci_lower_pct": float(rmse_low),
        "rmse_skill_ci_upper_pct": float(rmse_high),
    }


def evaluate_walk_forward(model, X_scaled, scaler_y, origins,
                          aligned_rates, aligned_dates, currency,
                          horizons=DISPLAY_HORIZONS,
                          spacing=WALK_FORWARD_SPACING,
                          spacing_by_horizon=None,
                          include_arima=True):
    """Evaluate fixed weights at rolling origins.

    `spacing_by_horizon` permits selection-set origins to be spaced by the target
    horizon (e.g. 5 for a 5-session forecast), while final test evaluation uses
    the common 63-observation spacing for every reported horizon.
    """
    if len(origins) == 0:
        raise ValueError("No evaluation samples were created.")

    origins = np.asarray(origins, dtype=int)
    X_scaled = np.asarray(X_scaled)
    if len(X_scaled) != len(origins):
        raise ValueError("X_scaled and origins must contain the same number of samples.")

    pred_returns = _inverse_return_predictions(model, X_scaled, scaler_y)
    if max(horizons) > pred_returns.shape[1]:
        raise ValueError("A requested evaluation horizon exceeds the model output horizon.")
    origin_to_row = {int(origin): i for i, origin in enumerate(origins)}

    selected_by_horizon = {}
    all_selected = set()
    for h in horizons:
        h_spacing = int((spacing_by_horizon or {}).get(int(h), spacing))
        if h_spacing < 1:
            raise ValueError("Origin spacing must be at least one observation.")
        selected = origins[::h_spacing]
        selected_by_horizon[int(h)] = selected
        all_selected.update(int(x) for x in selected)

    if not all_selected:
        raise ValueError("No walk-forward origins remain after applying spacing.")

    arima_paths = {}
    if include_arima:
        for origin in sorted(all_selected):
            start = max(0, int(origin) - ARIMA_MAX_HISTORY)
            history = np.log(aligned_rates[start:int(origin) + 1])
            arima_paths[int(origin)] = _fit_arima_forecast(history, pred_returns.shape[1])

    detail_rows = []
    for h in horizons:
        for origin in selected_by_horizon[int(h)]:
            origin = int(origin)
            row_idx = origin_to_row[origin]
            base_rate = float(aligned_rates[origin])
            actual_path = aligned_rates[origin + 1:origin + pred_returns.shape[1] + 1]
            if len(actual_path) < int(h):
                continue

            cnn_path = base_rate * np.exp(np.cumsum(pred_returns[row_idx]))
            history_start = max(0, origin - ARIMA_MAX_HISTORY)
            log_history = np.log(aligned_rates[history_start:origin + 1])
            paths = benchmark_forecast_paths(
                log_history, base_rate, pred_returns.shape[1],
                cnn_path=cnn_path, include_arima=False,
            )
            arima_path = arima_paths.get(origin) if include_arima else None
            if arima_path is not None:
                paths["ARIMA(1,1,1)"] = arima_path

            origin_date = pd.Timestamp(aligned_dates[origin]).date().isoformat()
            actual = float(actual_path[int(h) - 1])
            for model_name, path in paths.items():
                predicted = float(path[int(h) - 1])
                if model_name == "Random walk":
                    direction_correct = np.nan
                else:
                    direction_correct = bool(
                        np.sign(actual - base_rate) == np.sign(predicted - base_rate)
                    )
                error = abs(actual - predicted)
                detail_rows.append({
                    "currency": currency,
                    "origin_index": origin,
                    "origin_date": origin_date,
                    "horizon": int(h),
                    "model": model_name,
                    "base_rate": base_rate,
                    "actual_rate": actual,
                    "forecast_rate": predicted,
                    "absolute_error": error,
                    "squared_error": float((actual - predicted) ** 2),
                    "absolute_percentage_error": abs((actual - predicted) / actual) * 100.0,
                    "direction_correct": direction_correct,
                })

    details = pd.DataFrame(detail_rows)
    if details.empty:
        raise ValueError("Walk-forward evaluation produced no result rows.")

    metric_rows = []
    for (horizon, model_name), group in details.groupby(["horizon", "model"], sort=True):
        metric_rows.append(_metric_row(
            currency, horizon, model_name,
            group["actual_rate"].to_numpy(),
            group["forecast_rate"].to_numpy(),
            group["base_rate"].to_numpy(),
        ))
    metrics = pd.DataFrame(metric_rows)

    rw = details[details["model"] == "Random walk"][[
        "horizon", "origin_index", "absolute_error"
    ]].rename(columns={"absolute_error": "random_walk_absolute_error"})

    skill_rows = []
    for (horizon, model_name), group in details.groupby(["horizon", "model"], sort=True):
        paired = group[["origin_index", "absolute_error"]].merge(
            rw[rw["horizon"] == horizon][["origin_index", "random_walk_absolute_error"]],
            on="origin_index", how="inner"
        )
        skill_rows.append({
            "horizon": int(horizon),
            "model": model_name,
            **_paired_skill_and_ci(
                paired["random_walk_absolute_error"].to_numpy(),
                paired["absolute_error"].to_numpy(),
                model_name,
                horizon,
            ),
        })

    metrics = metrics.merge(pd.DataFrame(skill_rows), on=["horizon", "model"], how="left")
    # Preserve the V2 column names for older dashboard code and CSV consumers.
    metrics["skill_vs_random_walk_pct"] = metrics["mae_skill_vs_random_walk_pct"]
    metrics["skill_ci_lower_pct"] = metrics["mae_skill_ci_lower_pct"]
    metrics["skill_ci_upper_pct"] = metrics["mae_skill_ci_upper_pct"]
    metrics["n_origins"] = metrics["n_origins"].astype(int)
    metrics = metrics.sort_values(["horizon", "model"]).reset_index(drop=True)
    return metrics, details


def select_benchmark_methods(selection_metrics, min_origins=MIN_SELECTION_ORIGINS):
    """Select one method per currency/horizon using selection-partition MAE.

    MAE is the primary criterion. RMSE and absolute-error standard deviation are
    deterministic tie-breakers. The final test metrics are never an input.
    Random walk is the conservative fallback when the selection sample is too small.
    """
    if selection_metrics is None or selection_metrics.empty:
        raise ValueError("Selection metrics are empty; cannot build a forecast policy.")

    rows = []
    for (currency, horizon), group in selection_metrics.groupby(["currency", "horizon"], sort=True):
        eligible = group[
            (group["n_origins"] >= int(min_origins))
            & np.isfinite(group["mae"])
            & np.isfinite(group["rmse"])
        ].copy()

        fallback = False
        if eligible.empty:
            random_walk = group[group["model"] == "Random walk"]
            if random_walk.empty:
                raise ValueError(f"No usable benchmark result for {currency} horizon {horizon}.")
            winner = random_walk.iloc[0]
            fallback = True
            reason = f"Random walk fallback: fewer than {min_origins} usable selection origins."
        else:
            eligible = eligible.sort_values(
                ["mae", "rmse", "absolute_error_std", "model"],
                ascending=[True, True, True, True],
                na_position="last",
            )
            winner = eligible.iloc[0]
            reason = "Lowest selection-partition MAE; RMSE and error variability break ties."

        rows.append({
            "currency": currency,
            "horizon": int(horizon),
            "selected_model": str(winner["model"]),
            "selection_metric": "selection-partition MAE",
            "selection_mae": float(winner["mae"]),
            "selection_rmse": float(winner["rmse"]),
            "n_selection_origins": int(winner["n_origins"]),
            "eligible_candidate_count": int(len(eligible)),
            "selection_fallback": bool(fallback),
            "selection_reason": reason,
        })
    return pd.DataFrame(rows).sort_values(["currency", "horizon"]).reset_index(drop=True)


def calibrate_method_forecast_intervals(
    model, X_cal_scaled, scaler_y, calibration_origins, aligned_rates,
    horizon=63, include_arima=True,
    origin_stride=CALIBRATION_ORIGIN_STRIDE,
    min_samples=CALIBRATION_MIN_SAMPLES,
):
    """Calibrate descriptive empirical error bands for every candidate method.

    Calibration origins are thinned by `origin_stride` to limit repeated ARIMA
    fitting. The resulting forecast windows can still overlap; these are empirical
    scenario bands, not a formal independent-sample conformal guarantee.
    """
    if len(calibration_origins) == 0:
        raise ValueError("No calibration samples are available.")
    if len(X_cal_scaled) != len(calibration_origins):
        raise ValueError("X_cal_scaled and calibration_origins must have equal lengths.")

    pred_returns = _inverse_return_predictions(model, X_cal_scaled, scaler_y)
    origins_all = np.asarray(calibration_origins, dtype=int)
    origin_to_row = {int(origin): i for i, origin in enumerate(origins_all)}
    origins = origins_all[::max(1, int(origin_stride))]
    error_store = {}

    for origin in origins:
        origin = int(origin)
        actual_path = np.asarray(aligned_rates[origin + 1:origin + horizon + 1], dtype=float)
        if len(actual_path) < horizon:
            continue
        base_rate = float(aligned_rates[origin])
        cnn_path = base_rate * np.exp(np.cumsum(pred_returns[origin_to_row[origin]]))
        history_start = max(0, origin - ARIMA_MAX_HISTORY)
        log_history = np.log(aligned_rates[history_start:origin + 1])
        model_horizon = int(pred_returns.shape[1])
        paths = benchmark_forecast_paths(
            log_history, base_rate, model_horizon, cnn_path=cnn_path, include_arima=include_arima
        )
        actual_log = np.log(actual_path)
        for method, path in paths.items():
            errors = np.abs(actual_log - np.log(np.asarray(path, dtype=float)[:horizon]))
            for step, error in enumerate(errors, start=1):
                error_store.setdefault((method, step), []).append(float(error))

    rows = []
    methods = sorted({method for method, _ in error_store})
    for method in methods:
        for step in range(1, horizon + 1):
            errors = np.asarray(error_store.get((method, step), []), dtype=float)
            count = int(len(errors))
            available = count >= int(min_samples)
            if available:
                q80 = float(np.quantile(errors, 0.80, method="higher"))
                q95 = float(np.quantile(errors, 0.95, method="higher"))
            else:
                q80, q95 = np.nan, np.nan
            rows.append({
                "model": method,
                "horizon": int(step),
                "q80_abs_log_error": q80,
                "q95_abs_log_error": q95,
                "calibration_samples": count,
                "interval_available": bool(available),
                "origin_stride": int(max(1, origin_stride)),
            })
    return pd.DataFrame(rows)


def calibrate_forecast_intervals(model, X_cal_scaled, scaler_y, calibration_origins,
                                 aligned_rates, horizon=63,
                                 min_samples=CALIBRATION_MIN_SAMPLES):
    """Backward-compatible CNN-LSTM-only calibration wrapper."""
    table = calibrate_method_forecast_intervals(
        model, X_cal_scaled, scaler_y, calibration_origins, aligned_rates,
        horizon=horizon, include_arima=False, min_samples=min_samples,
    )
    return table[table["model"] == "CNN-LSTM"].drop(columns=["model"]).reset_index(drop=True)
