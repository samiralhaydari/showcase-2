import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.supplylens.ui_helpers import cached_rates, currency_label, currency_rate_precision
from src.supplylens.data import get_currency_series
from src.supplylens.forecasting import forecast_currency
from src.supplylens.config import PRIORITY_CURRENCIES, DISPLAY_HORIZONS

st.set_page_config(page_title="FX Forecast Centre", page_icon="📈", layout="wide")
st.title("FX Forecast Centre")
st.caption("Historical reference rates, benchmark-aware forecasts and empirical uncertainty estimates.")

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
fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values, name="Historical middle rate",
    line=dict(color="#2563EB", width=2)
))

try:
    forecast, metadata = forecast_currency(series, currency)
    if {"lower_95", "upper_95"}.issubset(forecast.columns):
        if forecast["lower_95"].notna().any() and forecast["upper_95"].notna().any():
            fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["lower_95"],
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper_95"],
                                     line=dict(width=0), fill="tonexty", fillcolor="rgba(245,158,11,0.10)",
                                     name="CNN-LSTM empirical 95% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["forecast_rate"],
        name="CNN-LSTM reference path", line=dict(color="#D97706", width=2, dash="dash")
    ))

    checkpoints = []
    marker_x, marker_y, marker_text = [], [], []
    for h in DISPLAY_HORIZONS:
        row = forecast.iloc[h - 1]
        selected_value = row.get("selected_forecast_rate", float("nan"))
        if pd.isna(selected_value):
            selected_value = float(row["forecast_rate"])
            selected_model = "CNN-LSTM (fallback)"
        else:
            selected_model = str(row.get("selected_model") or "CNN-LSTM")
        marker_x.append(row["date"])
        marker_y.append(float(selected_value))
        marker_text.append(f"{h} sessions · {selected_model}")
        item = {
            "Horizon (sessions)": h,
            "Approximate weekday date": row["date"].date(),
            "Selected method": selected_model,
            f"Selected forecast (MYR per 1 {currency})": f"{float(selected_value):.{precision}f}",
        }
        low, high = row.get("selected_lower_80", float("nan")), row.get("selected_upper_80", float("nan"))
        if pd.notna(low) and pd.notna(high):
            item["Empirical 80% scenario range"] = f"{float(low):.{precision}f} – {float(high):.{precision}f}"
        else:
            item["Empirical 80% scenario range"] = "Not available"
        checkpoints.append(item)

    fig.add_trace(go.Scatter(
        x=marker_x, y=marker_y, mode="markers+text", text=marker_text,
        textposition="top center", name="Validation-selected checkpoints",
        marker=dict(color="#15803D", size=10, symbol="diamond"),
        hovertemplate="%{text}<br>MYR rate: %{y:.8f}<extra></extra>",
    ))
    st.success(
        f"Loaded {metadata.get('model_version', 'V1')} model. "
        f"Selection mode: {metadata.get('selection_mode', 'CNN-LSTM default')}."
    )
    st.subheader("Forecast checkpoints")
    st.dataframe(pd.DataFrame(checkpoints), use_container_width=True, hide_index=True)
except FileNotFoundError:
    st.warning(f"No trained model for {currency}. Train it in the Colab notebook.")
except Exception as exc:
    st.error(f"Forecast could not be generated: {exc}")

fig.update_layout(
    title=f"{currency} reference rate and forecast", xaxis_title="Date",
    yaxis_title=f"MYR per 1 {currency}", hovermode="x unified",
    legend=dict(orientation="h", y=1.12)
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The dashed path is the CNN-LSTM reference forecast. Green diamonds show the method "
    "selected using a separate historical selection partition at the four displayed horizons. "
    "Empirical uncertainty ranges are descriptive historical scenarios, not guaranteed coverage. "
    "The middle rate is a central-bank reference rate, not necessarily an SME transaction rate. "
    "Forecast dates skip weekends but do not apply a full Malaysian market-holiday calendar."
)
