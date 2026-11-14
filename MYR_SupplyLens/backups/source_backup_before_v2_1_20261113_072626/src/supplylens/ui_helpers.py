import numpy as np
import streamlit as st
from .data import load_exchange_rates, get_currency_series
from .config import CURRENCY_NAMES, MODEL_DIR, MODEL_VERSION, LEGACY_MODEL_VERSIONS


@st.cache_data(ttl=3600, show_spinner=False)
def cached_rates():
    return load_exchange_rates()


def get_current_rate(rates, currency):
    series = get_currency_series(rates, currency)
    return float(series.iloc[-1]), series.index[-1]


def selected_forecast_value(forecast, horizon):
    """Return the validation-selected forecast at a supported checkpoint.

    Legacy V2/V1 bundles have no selection policy, so their CNN-LSTM forecast is
    returned as a clearly labelled compatibility fallback.
    """
    row = forecast.iloc[min(int(horizon), len(forecast)) - 1]
    selected = row.get("selected_forecast_rate", np.nan)
    if selected is not None and np.isfinite(float(selected)):
        model_name = row.get("selected_model") or "CNN-LSTM"
        return float(selected), str(model_name)
    return float(row["forecast_rate"]), "CNN-LSTM (legacy/default fallback)"


def get_forecast_rate(rates, currency, horizon):
    from .forecasting import forecast_currency
    series = get_currency_series(rates, currency)
    forecast, metadata = forecast_currency(series, currency)
    value, _ = selected_forecast_value(forecast, horizon)
    return value, forecast, metadata


def currency_label(code):
    return f"{code} — {CURRENCY_NAMES.get(code, code)}"


def currency_rate_precision(code):
    return {
        "USD": 4, "CNY": 6, "IDR": 8, "SGD": 4,
        "THB": 5, "JPY": 6, "KRW": 7, "PHP": 6,
        "VND": 7, "BND": 4, "EUR": 4, "GBP": 4,
    }.get(str(code).upper(), 6)


def trained_currencies():
    found = set()
    for version in [MODEL_VERSION] + list(LEGACY_MODEL_VERSIONS):
        if version == "v1":
            pattern = "cnn_lstm_*.keras"
            prefix = "cnn_lstm_"
        else:
            pattern = f"cnn_lstm_{version}_*.keras"
            prefix = f"cnn_lstm_{version}_"
        for path in MODEL_DIR.glob(pattern):
            if version == "v1" and (path.stem.startswith("cnn_lstm_v2_") or path.stem.startswith("cnn_lstm_v2_1_")):
                continue
            name = path.stem.replace(prefix, "", 1)
            if name and name not in {"v2", "v2_1"}:
                found.add(name)
    return sorted(found)
