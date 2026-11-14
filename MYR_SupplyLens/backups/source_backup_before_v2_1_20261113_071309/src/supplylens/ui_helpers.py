import streamlit as st
from .data import load_exchange_rates, get_currency_series
from .config import CURRENCY_NAMES, MODEL_DIR, MODEL_VERSION


@st.cache_data(ttl=3600, show_spinner=False)
def cached_rates():
    return load_exchange_rates()


def get_current_rate(rates, currency):
    series = get_currency_series(rates, currency)
    return float(series.iloc[-1]), series.index[-1]


def get_forecast_rate(rates, currency, horizon):
    from .forecasting import forecast_currency
    series = get_currency_series(rates, currency)
    forecast, metadata = forecast_currency(series, currency)
    row = forecast.iloc[min(int(horizon), len(forecast)) - 1]
    return float(row["forecast_rate"]), forecast, metadata


def currency_label(code):
    return f"{code} — {CURRENCY_NAMES.get(code, code)}"


def currency_rate_precision(code):
    return {
        "USD": 4, "CNY": 6, "IDR": 8, "SGD": 4,
        "THB": 5, "JPY": 6, "KRW": 7, "PHP": 6,
        "VND": 7, "BND": 4, "EUR": 4, "GBP": 4,
    }.get(str(code).upper(), 6)


def trained_currencies():
    v2 = sorted(
        p.stem.replace(f"cnn_lstm_{MODEL_VERSION}_", "")
        for p in MODEL_DIR.glob(f"cnn_lstm_{MODEL_VERSION}_*.keras")
    )
    if v2:
        return v2
    return sorted(p.stem.replace("cnn_lstm_", "") for p in MODEL_DIR.glob("cnn_lstm_*.keras"))
