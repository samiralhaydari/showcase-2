import streamlit as st
from src.supplylens.ui_helpers import cached_rates, trained_currencies, currency_rate_precision
from src.supplylens.database import init_db, dashboard_counts
from src.supplylens.config import DATASET_PAGE, PRIORITY_CURRENCIES

st.set_page_config(page_title="MYR SupplyLens", page_icon="📊", layout="wide")
init_db()

st.title("MYR SupplyLens")
st.caption("Predict. Plan. Procure. Profit. | Multi-Industry Currency & Business Cost Intelligence")
st.markdown(
    "MYR SupplyLens converts regional exchange-rate forecasts into procurement, "
    "supplier-cost and product-profitability insights. The initial demonstration "
    "targets Malaysian manufacturing SMEs."
)

try:
    rates = cached_rates()
    latest_date = rates.index.max()
    available = list(rates.columns)
    trained = trained_currencies()
    counts = dashboard_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available currency series", len(available))
    c2.metric("Trained CNN-LSTM models", len(trained))
    c3.metric("Saved suppliers", counts["suppliers"])
    c4.metric("Purchase exposures", counts["exposures"])

    st.subheader("Regional currency coverage")
    show_codes = [c for c in PRIORITY_CURRENCIES if c in available]
    st.write(" · ".join(show_codes) if show_codes else "No priority currency columns found.")
    st.info(f"Latest observation: {latest_date.date()}. Rates are MYR per one unit of foreign currency.")
    st.subheader("Latest reference rates")
    if show_codes:
        display = rates.loc[latest_date, show_codes].rename("MYR per 1 foreign currency unit").to_frame()
        display["Formatted reference rate"] = [
            f"{float(display.loc[code, 'MYR per 1 foreign currency unit']):.{currency_rate_precision(code)}f}"
            for code in display.index
        ]
        st.dataframe(display[["Formatted reference rate"]], use_container_width=True)
except Exception as exc:
    st.error(f"Could not load exchange-rate data: {exc}")
    st.caption("Check Colab internet access and rerun the data-loading cell.")

st.subheader("Platform modules")
cols = st.columns(3)
modules = [
    ("FX Forecast Centre", "Historical rates, CNN-LSTM forecasts and uncertainty estimates."),
    ("Procurement Simulator", "Estimate MYR costs of supplier invoices."),
    ("Supplier Comparison", "Compare equivalent projected landed costs."),
    ("Profitability Simulator", "Assess manufacturing gross margins."),
    ("Model Insights", "Inspect model design, walk-forward benchmarks and calibration."),
]
for i, (name, description) in enumerate(modules):
    with cols[i % 3]:
        st.markdown(f"**{name}**")
        st.caption(description)

st.divider()
st.caption(
    f"Data: Bank Negara Malaysia via data.gov.my · {DATASET_PAGE} · CC BY 4.0. "
    "Reference rates are not guaranteed transaction rates. Forecasts are estimates, not financial advice."
)
