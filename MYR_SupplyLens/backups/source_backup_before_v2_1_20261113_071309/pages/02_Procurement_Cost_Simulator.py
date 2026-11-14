import streamlit as st
from src.supplylens.ui_helpers import cached_rates, get_current_rate, get_forecast_rate, currency_label, currency_rate_precision
from src.supplylens.config import PRIORITY_CURRENCIES
from src.supplylens.business import invoice_cost, apply_user_scenario_shock
from src.supplylens.database import save_exposure

st.set_page_config(page_title="Procurement Cost Simulator", page_icon="🧾", layout="wide")
st.title("Procurement Cost Simulator")
st.caption("Translate a foreign-currency supplier invoice into an estimated MYR procurement cost.")

try:
    rates = cached_rates()
except Exception as exc:
    st.error(f"Unable to load rates: {exc}")
    st.stop()

currencies = [c for c in PRIORITY_CURRENCIES if c in rates.columns]
currency = st.selectbox("Invoice currency", currencies, format_func=currency_label)
amount = st.number_input("Supplier invoice amount", min_value=0.0, value=100000.0, step=1000.0)
horizon = st.select_slider("Payment horizon (trading sessions)", [5, 21, 42, 63], value=21)
supplier = st.text_input("Supplier name", value="China Raw Materials Supplier")
category = st.selectbox("Purchase category", ["Raw materials", "Machinery", "Components", "Packaging", "Other"])
due_date = st.date_input("Expected payment date")

current_rate, latest_date = get_current_rate(rates, currency)
precision = currency_rate_precision(currency)
st.info(f"Latest reference rate: RM {current_rate:.{precision}f} per 1 {currency} ({latest_date.date()}).")

try:
    forecast_rate, forecast, metadata = get_forecast_rate(rates, currency, horizon)
    current_cost = invoice_cost(amount, current_rate)
    expected_cost = invoice_cost(amount, forecast_rate)
    shock = st.slider("User-defined adverse scenario shock (%)", 0.0, 10.0, 2.0, 0.5)
    adverse_rate = apply_user_scenario_shock(forecast_rate, shock)
    adverse_cost = invoice_cost(amount, adverse_rate)

    st.subheader("Estimated procurement cost")
    a, b, c = st.columns(3)
    a.metric("At latest reference rate", f"RM {current_cost:,.2f}")
    b.metric("At CNN-LSTM forecast", f"RM {expected_cost:,.2f}", f"{expected_cost-current_cost:+,.2f}")
    c.metric("User-defined adverse scenario", f"RM {adverse_cost:,.2f}", f"{adverse_cost-current_cost:+,.2f}")

    if "lower_80" in forecast.columns and "upper_80" in forecast.columns:
        row = forecast.iloc[horizon - 1]
        if row["lower_80"] == row["lower_80"]:
            st.info(
                f"Empirical 80% forecast range at this horizon: "
                f"RM {invoice_cost(amount, row['lower_80']):,.2f} – "
                f"RM {invoice_cost(amount, row['upper_80']):,.2f}. "
                "This is an indicative historical calibration range, not a guarantee."
            )
    st.caption(
        f"Forecast checkpoint: {horizon} trading sessions. The adverse scenario adds "
        f"{shock:.1f}% to the model forecast; this is a user assumption, not a prediction."
    )
    if st.button("Save invoice exposure", type="primary"):
        save_exposure(supplier, currency, amount, due_date.isoformat(), category)
        st.success("Invoice exposure saved to SQLite.")
except FileNotFoundError:
    st.warning(f"No trained model is available for {currency}. Showing current-rate conversion only.")
    st.metric("Current-rate invoice conversion", f"RM {invoice_cost(amount, current_rate):,.2f}")
