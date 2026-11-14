import json
import streamlit as st
import pandas as pd
import plotly.express as px

from src.supplylens.config import METADATA_DIR, PRIORITY_CURRENCIES, MODEL_VERSION

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
st.title("Model Insights & Methodology")
st.caption("Transparent model design, out-of-sample evaluation and forecast uncertainty.")

st.subheader("CNN-LSTM architecture")
st.code(
    "Input: 90 observations × 9 features\n"
    "Conv1D: 32 filters, kernel size 3, causal padding, ReLU\n"
    "MaxPooling1D: pool size 2\n"
    "LSTM: 32 units\n"
    "Dropout: 0.20\n"
    "Dense: 32 units, ReLU\n"
    "Output: 63 future daily log returns",
    language="text",
)

feature_rows = [
    ["log_rate", "Natural logarithm of the exchange-rate level"],
    ["return_1d", "One-observation log return"],
    ["momentum_5", "Five-observation log momentum"],
    ["momentum_10", "Ten-observation log momentum"],
    ["momentum_20", "Twenty-observation log momentum"],
    ["volatility_5", "Five-observation rolling return volatility"],
    ["volatility_20", "Twenty-observation rolling return volatility"],
    ["volatility_60", "Sixty-observation rolling return volatility"],
    ["range_20", "Twenty-observation high-low range of log rates"],
]
st.subheader("Input features")
st.dataframe(pd.DataFrame(feature_rows, columns=["Feature", "Description"]),
             use_container_width=True, hide_index=True)

st.subheader("Evaluation methodology")
st.markdown(
    "- Chronological partitions: 65% training, 10% validation, 10% calibration, 15% test.\n"
    "- Model weights are fixed after training; forecast origins roll forward through the held-out test period.\n"
    "- Common forecast origins are spaced 63 observations apart, preventing overlapping target blocks at the longest horizon.\n"
    "- Benchmarks: random walk, 252-observation historical drift, 20-observation recent drift and ARIMA(1,1,1).\n"
    "- Skill score: positive values mean lower MAE than random walk; negative values mean random walk is better.\n"
    "- Skill-score uncertainty: paired bootstrap 95% intervals resample the same non-overlapping forecast origins for each model and random-walk comparison. If the interval crosses zero, evidence of an advantage is inconclusive."
)
st.caption(
    "The walk-forward origins are deliberately non-overlapping. This is more conservative "
    "than reporting hundreds of overlapping windows, but the number of independent origins "
    "is smaller and must be shown alongside the metrics."
)

st.subheader("V2 trained model evaluation")
all_metrics = []
all_metadata = []
for currency in PRIORITY_CURRENCIES:
    meta_path = METADATA_DIR / f"metadata_{MODEL_VERSION}_{currency}.json"
    metric_path = METADATA_DIR / f"metrics_{MODEL_VERSION}_{currency}.csv"
    calibration_path = METADATA_DIR / f"calibration_{MODEL_VERSION}_{currency}.csv"
    if not meta_path.exists() or not metric_path.exists():
        continue
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metric_path)
    all_metadata.append((currency, metadata, calibration_path))
    all_metrics.append(metrics)

if not all_metadata:
    st.warning("No V2 models found. Run the V2 training cells in Google Colab first.")
else:
    combined = pd.concat(all_metrics, ignore_index=True)
    cnn = combined[combined["model"] == "CNN-LSTM"].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("V2 currencies trained", len(all_metadata))
    c2.metric("Independent test origins / currency", 
              int(round(sum(m.get("walk_forward_origins", 0) for _, m, _ in all_metadata) / len(all_metadata))))
    c3.metric("Benchmarks", combined["model"].nunique())

    st.markdown("**CNN-LSTM skill against random walk**")
    if not cnn.empty:
        chart = px.bar(
            cnn, x="horizon", y="skill_vs_random_walk_pct", color="currency",
            barmode="group", text_auto=".2f",
            labels={"horizon": "Forecast horizon (observations)",
                    "skill_vs_random_walk_pct": "MAE improvement vs random walk (%)"},
            title="Positive values favour CNN-LSTM; negative values favour random walk",
        )
        chart.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(chart, use_container_width=True)

    st.markdown("**Benchmark comparison**")
    display_columns = [
        "currency", "horizon", "model", "n_origins", "mae", "rmse",
        "mape_pct", "directional_accuracy_pct", "skill_vs_random_walk_pct",
        "skill_ci_lower_pct", "skill_ci_upper_pct"
    ]
    st.dataframe(
        combined[display_columns].sort_values(["currency", "horizon", "model"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "mae": st.column_config.NumberColumn(format="%.8f"),
            "rmse": st.column_config.NumberColumn(format="%.8f"),
            "mape_pct": st.column_config.NumberColumn(format="%.4f%%"),
            "directional_accuracy_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "skill_vs_random_walk_pct": st.column_config.NumberColumn(format="%.3f%%"),
            "skill_ci_lower_pct": st.column_config.NumberColumn(format="%.3f%%"),
            "skill_ci_upper_pct": st.column_config.NumberColumn(format="%.3f%%"),
        },
    )

    for currency, metadata, calibration_path in all_metadata:
        with st.expander(f"{currency} — {metadata.get('trained_at_utc', 'training date unavailable')}"):
            a, b, c = st.columns(3)
            a.metric("Latest observation", metadata.get("latest_observation_date", "—"))
            b.metric("Latest reference rate", f"{metadata.get('latest_rate', float('nan')):.8f}")
            b.caption(f"MYR per 1 {currency}")
            c.metric("Non-overlapping test origins", metadata.get("walk_forward_origins", 0))
            st.write("Train / validation / calibration samples:",
                     metadata.get("train_samples"), "/",
                     metadata.get("validation_samples"), "/",
                     metadata.get("calibration_samples"))
            st.write("Architecture:", metadata.get("architecture", ""))
            if calibration_path.exists():
                calibration = pd.read_csv(calibration_path)
                available = int(calibration["interval_available"].sum())
                st.write(f"Calibrated forecast steps available: {available} / {len(calibration)}")
                st.caption(metadata.get("calibration_method", ""))
            currency_cnn = cnn[cnn["currency"] == currency]
            beats = int((currency_cnn["skill_vs_random_walk_pct"] > 0).sum())
            st.write(f"CNN-LSTM beats random walk at {beats} of {len(currency_cnn)} reported horizons.")

st.subheader("Data source and limitations")
st.markdown(
    "- Source: Bank Negara Malaysia via data.gov.my, Daily Exchange Rates (1200).\n"
    "- Selected rate type: middle, where available.\n"
    "- Quotation: MYR per one unit of foreign currency.\n"
    "- Calibration bands use a separate historical calibration split; overlapping calibration origins mean they are descriptive estimates, not guaranteed coverage intervals.\n"
    "- ARIMA failures at individual origins are omitted from that benchmark's metric and reflected in its origin count.\n"
    "- Central-bank reference rates are not necessarily the transaction rates offered by banks.\n"
    "- Forecasts are estimates, not financial advice or guaranteed outcomes."
)
st.markdown("[Official dataset documentation](https://data.gov.my/data-catalogue/exchangerates_daily_1200)")
