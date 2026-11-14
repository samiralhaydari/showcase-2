import streamlit as st
import pandas as pd
import plotly.express as px
from src.supplylens.ui_helpers import cached_rates, currency_label, currency_rate_precision
from src.supplylens.data import get_currency_series
from src.supplylens.forecasting import forecast_currency
from src.supplylens.config import PRIORITY_CURRENCIES
from src.supplylens.business import gross_margin, invoice_cost

st.set_page_config(page_title="Profitability Simulator", page_icon="📊", layout="wide")
st.title("Profitability Simulator")
st.caption("Estimate how foreign-currency input costs can affect manufacturing gross margins.")

try:
    rates = cached_rates()
except Exception as exc:
    st.error(f"Unable to load rates: {exc}")
    st.stop()

currencies = [c for c in PRIORITY_CURRENCIES if c in rates.columns]
currency = st.selectbox("Imported material currency", currencies, format_func=currency_label)
horizon = st.select_slider("Forecast horizon (trading sessions)", [5, 21, 42, 63], value=21)

c1, c2 = st.columns(2)
with c1:
    foreign_material = st.number_input(f"Imported material cost per unit ({currency})", min_value=0.0, value=8.0, step=0.5)
    labour = st.number_input("Local labour cost per unit (MYR)", min_value=0.0, value=15.0, step=1.0)
    overhead = st.number_input("Manufacturing overhead per unit (MYR)", min_value=0.0, value=20.0, step=1.0)
with c2:
    packaging = st.number_input("Packaging and other local costs per unit (MYR)", min_value=0.0, value=5.0, step=1.0)
    selling_price = st.number_input("Selling price per finished unit (MYR)", min_value=0.01, value=100.0, step=1.0)

series = get_currency_series(rates, currency)
current_rate = float(series.iloc[-1])
precision = currency_rate_precision(currency)
try:
    forecast_df, metadata = forecast_currency(series, currency)
    forecast_rate = float(forecast_df.iloc[horizon-1]["forecast_rate"])
    basis = f"{metadata.get('model_version', 'V1')} CNN-LSTM forecast"
except Exception:
    forecast_df = None
    forecast_rate, basis = current_rate, "Current reference rate (no trained model)"

current_imported = invoice_cost(foreign_material, current_rate)
projected_imported = invoice_cost(foreign_material, forecast_rate)
local_costs = labour + overhead + packaging
current = gross_margin(selling_price, current_imported, local_costs)
projected = gross_margin(selling_price, projected_imported, local_costs)

st.caption(
    f"Latest reference rate: RM {current_rate:.{precision}f} per 1 {currency}. "
    f"Projected scenario basis: {basis}."
)
a, b, c = st.columns(3)
a.metric("Current gross margin", f"{current['gross_margin_pct']:.2f}%")
b.metric("Projected gross margin", f"{projected['gross_margin_pct']:.2f}%",
         f"{projected['gross_margin_pct']-current['gross_margin_pct']:+.2f} pp")
c.metric("Projected imported cost / unit", f"RM {projected_imported:.2f}")

comparison = pd.DataFrame([
    {"Scenario": "Current reference rate", "Imported material (MYR)": current_imported,
     "Total unit cost (MYR)": current["total_cost"], "Gross margin (%)": current["gross_margin_pct"]},
    {"Scenario": "Projected rate", "Imported material (MYR)": projected_imported,
     "Total unit cost (MYR)": projected["total_cost"], "Gross margin (%)": projected["gross_margin_pct"]},
])
st.subheader("Unit economics comparison")
formatted = comparison.copy()
for col in ["Imported material (MYR)", "Total unit cost (MYR)"]:
    formatted[col] = formatted[col].map(lambda x: f"RM {x:,.2f}")
formatted["Gross margin (%)"] = formatted["Gross margin (%)"].map(lambda x: f"{x:.2f}%")
st.dataframe(formatted, use_container_width=True, hide_index=True)

if forecast_df is not None and {"lower_80", "upper_80"}.issubset(forecast_df.columns):
    interval = forecast_df.iloc[horizon-1]
    if pd.notna(interval["lower_80"]) and pd.notna(interval["upper_80"]):
        low_cost = invoice_cost(foreign_material, float(interval["lower_80"]))
        high_cost = invoice_cost(foreign_material, float(interval["upper_80"]))
        low_margin = gross_margin(selling_price, low_cost, local_costs)["gross_margin_pct"]
        high_margin = gross_margin(selling_price, high_cost, local_costs)["gross_margin_pct"]
        st.info(
            f"Indicative empirical 80% scenario range: gross margin "
            f"{min(low_margin, high_margin):.2f}% to {max(low_margin, high_margin):.2f}%. "
            "This range is calibrated from historical errors and is not a guarantee."
        )

fig = px.bar(comparison, x="Scenario", y="Gross margin (%)",
             title="Gross margin under current and projected currency rates",
             text_auto=".2f")
st.plotly_chart(fig, use_container_width=True)
st.info(
    "Simplified gross-margin model. Financing costs, taxes, discounts, wastage "
    "and other operating expenses are excluded unless entered."
)
