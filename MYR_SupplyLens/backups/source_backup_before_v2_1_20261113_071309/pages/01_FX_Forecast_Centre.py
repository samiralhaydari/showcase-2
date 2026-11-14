import streamlit as st
import plotly.graph_objects as go
from src.supplylens.ui_helpers import cached_rates, currency_label, currency_rate_precision
from src.supplylens.data import get_currency_series
from src.supplylens.forecasting import forecast_currency
from src.supplylens.config import PRIORITY_CURRENCIES, DISPLAY_HORIZONS

st.set_page_config(page_title="FX Forecast Centre", page_icon="📈", layout="wide")
st.title("FX Forecast Centre")
st.caption("Historical reference rates, CNN-LSTM forecasts and calibrated uncertainty estimates.")

try:
    rates = cached_rates()
except Exception as exc:
    st.error(f"Unable to load data: {exc}")
    st.stop()

currencies = [c for c in PRIORITY_CURRENCIES if c in rates.columns] or list(rates.columns)
currency = st.selectbox("Select currency", currencies, format_func=currency_label)
series = get_currency_series(rates, currency)
current_rate, latest_date = float(series.iloc[-1]), series.index[-1]
precision = currency_rate_precision(currency)

a, b, c = st.columns(3)
a.metric("Latest reference rate", f"RM {current_rate:.{precision}f}")
b.metric("Latest data date", str(latest_date.date()))
c.metric("Quotation", f"MYR per 1 {currency}")

window = st.slider("Historical display window", 90, 750, 365, 30)
fig = go.Figure()
hist = series.tail(window)
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical middle rate",
                         line=dict(color="#2563EB", width=2)))
try:
    forecast, metadata = forecast_currency(series, currency)
    if {"lower_95", "upper_95"}.issubset(forecast.columns):
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["lower_95"],
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper_95"],
                                 line=dict(width=0), fill="tonexty", fillcolor="rgba(245,158,11,0.10)",
                                 name="Empirical 95% uncertainty band", hoverinfo="skip"))
    if {"lower_80", "upper_80"}.issubset(forecast.columns):
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["lower_80"],
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper_80"],
                                 line=dict(width=0), fill="tonexty", fillcolor="rgba(245,158,11,0.20)",
                                 name="Empirical 80% uncertainty band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast_rate"],
                             name="CNN-LSTM forecast",
                             line=dict(color="#D97706", width=2, dash="dash")))
    st.success(f"Loaded {metadata.get('model_version', 'V1')} model trained at {metadata.get('trained_at_utc', 'unknown time')}.")
    checkpoints = []
    for h in DISPLAY_HORIZONS:
        row = forecast.iloc[h - 1]
        item = {
            "Trading-session horizon": h,
            "Approximate weekday date": row["date"].date(),
            f"Forecast MYR per 1 {currency}": f"{row['forecast_rate']:.{precision}f}",
        }
        if "lower_80" in forecast.columns and "upper_80" in forecast.columns:
            item["Empirical 80% range"] = (
                f"{row['lower_80']:.{precision}f} – {row['upper_80']:.{precision}f}"
                if row["lower_80"] == row["lower_80"] else "Not calibrated"
            )
        checkpoints.append(item)
    st.subheader("Forecast checkpoints")
    st.dataframe(checkpoints, use_container_width=True, hide_index=True)
except FileNotFoundError:
    st.warning(f"No trained model for {currency}. Train it in the Colab notebook.")
except Exception as exc:
    st.error(f"Forecast could not be generated: {exc}")

fig.update_layout(
    title=f"{currency} reference rate and forecast",
    xaxis_title="Date", yaxis_title=f"MYR per 1 {currency}",
    hovermode="x unified", legend=dict(orientation="h", y=1.12)
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Uncertainty bands are empirical estimates calibrated on a separate historical period. "
    "They are not guaranteed coverage intervals. The middle rate is a central-bank reference "
    "rate, not necessarily the transaction rate available to an SME. Forecast dates skip weekends "
    "but do not yet apply a full Malaysian market-holiday calendar."
)
