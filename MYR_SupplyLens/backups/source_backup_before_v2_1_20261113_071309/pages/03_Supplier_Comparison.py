import streamlit as st
import pandas as pd
from src.supplylens.ui_helpers import cached_rates, currency_label, currency_rate_precision
from src.supplylens.data import get_currency_series
from src.supplylens.forecasting import forecast_currency
from src.supplylens.config import PRIORITY_CURRENCIES
from src.supplylens.business import invoice_cost, landed_cost

st.set_page_config(page_title="Supplier Comparison", page_icon="⚖️", layout="wide")
st.title("Supplier Comparison")
st.caption("Compare equivalent supplier quotations using projected MYR invoice conversion and landed costs.")
st.info("The default supplier names, quantities and quotations are illustrative examples. Compare identical product specifications and quantities for a valid sourcing decision.")

try:
    rates = cached_rates()
except Exception as exc:
    st.error(f"Unable to load rates: {exc}")
    st.stop()

currencies = ["MYR"] + [c for c in PRIORITY_CURRENCIES if c in rates.columns]
default_codes = ["MYR", "IDR", "CNY"]
horizon = st.select_slider("Forecast horizon (trading sessions)", [5, 21, 42, 63], value=21)
rows = []
cols = st.columns(3)

for i in range(3):
    with cols[i]:
        st.subheader(f"Supplier {chr(65+i)}")
        name = st.text_input("Supplier name", value=f"Supplier {chr(65+i)}", key=f"name{i}")
        country = st.text_input("Country", value=["Malaysia", "Indonesia", "China"][i], key=f"country{i}")
        default_index = currencies.index(default_codes[i]) if default_codes[i] in currencies else 0
        currency = st.selectbox(
            "Quotation currency", currencies, index=default_index,
            format_func=lambda code: "MYR — Malaysian Ringgit" if code == "MYR" else currency_label(code),
            key=f"currency{i}"
        )
        amount = st.number_input("Quoted invoice amount", min_value=0.0,
                                 value=[60000.0, 450000000.0, 100000.0][i],
                                 step=1000.0, key=f"amount{i}")
        freight = st.number_input("Freight (MYR)", min_value=0.0, value=1000.0, step=100.0, key=f"freight{i}")
        insurance = st.number_input("Insurance (MYR)", min_value=0.0, value=200.0, step=50.0, key=f"insurance{i}")
        duties = st.number_input("Duties / other taxes (MYR)", min_value=0.0, value=500.0, step=50.0, key=f"duties{i}")

        if currency == "MYR":
            rate, basis = 1.0, "Local MYR quotation"
        else:
            series = get_currency_series(rates, currency)
            current_rate = float(series.iloc[-1])
            try:
                forecast_df, _ = forecast_currency(series, currency)
                rate = float(forecast_df.iloc[horizon-1]["forecast_rate"])
                basis = "CNN-LSTM forecast"
            except Exception:
                rate, basis = current_rate, "Current reference rate (no model)"

        invoice_myr = invoice_cost(amount, rate)
        total = landed_cost(invoice_myr, freight, insurance, duties)
        rows.append({
            "Supplier": name,
            "Country": country,
            "Currency": currency,
            "Rate used (MYR/unit)": rate,
            "Rate basis": basis,
            "Quoted invoice amount": amount,
            "Invoice (MYR)": invoice_myr,
            "Freight (MYR)": freight,
            "Insurance (MYR)": insurance,
            "Duties / taxes (MYR)": duties,
            "Total landed cost (MYR)": total,
        })

result = pd.DataFrame(rows).sort_values("Total landed cost (MYR)").reset_index(drop=True)
st.subheader("Projected landed-cost comparison")
formatted = result.copy()
formatted["Rate used (MYR/unit)"] = [
    f"{r:.{currency_rate_precision(c)}f}" for r, c in zip(result["Rate used (MYR/unit)"], result["Currency"])
]
formatted["Quoted invoice amount"] = result.apply(
    lambda row: f"{row['Currency']} {row['Quoted invoice amount']:,.2f}", axis=1
)
for col in ["Invoice (MYR)", "Freight (MYR)", "Insurance (MYR)", "Duties / taxes (MYR)", "Total landed cost (MYR)"]:
    formatted[col] = result[col].map(lambda x: f"RM {x:,.2f}")
st.dataframe(formatted, use_container_width=True, hide_index=True)

winner = result.iloc[0]
st.success(
    f"Lowest calculated landed cost: {winner['Supplier']} ({winner['Country']}) at "
    f"RM {winner['Total landed cost (MYR)']:,.2f}. Cost alone does not assess quality, "
    "lead time, reliability, product specifications or sustainability."
)
