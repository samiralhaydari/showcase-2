import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.supplylens.config import (
    METADATA_DIR, PRIORITY_CURRENCIES, MODEL_VERSION, LEGACY_MODEL_VERSIONS,
)

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
st.title("Model Insights & Methodology")
st.caption("Transparent model design, validation-based selection, out-of-sample evaluation and forecast uncertainty.")

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

st.subheader("V2.1 evaluation methodology")
st.markdown(
    "- Chronological partitions: 60% training, 10% early-stopping validation, "
    "10% method selection, 5% calibration and 15% untouched final test.\n"
    "- The method-selection partition is not used for gradient updates or early stopping.\n"
    "- Selection is performed separately for each currency and displayed horizon. "
    "Origins are spaced by the target horizon within that selection partition.\n"
    "- Selection rule: lowest selection-partition MAE; RMSE and absolute-error variability "
    "are deterministic tie-breakers. If fewer than eight usable origins exist, random walk is used.\n"
    "- The final test policy is frozen before the untouched test partition is evaluated. "
    "Test origins use common 63-observation spacing to avoid overlapping target blocks.\n"
    "- Benchmarks: random walk, 252-observation historical drift, 20-observation recent drift "
    "and ARIMA(1,1,1).\n"
    "- MAE skill and RMSE skill are calculated separately: positive values indicate lower "
    "error than random walk; negative values indicate higher error.\n"
    "- Paired bootstrap 95% intervals resample identical independent test origins for the "
    "model and random-walk benchmark. Intervals crossing zero are inconclusive.\n"
    "- Random walk has no directional prediction; directional accuracy is therefore N/A."
)
st.caption(
    "The calibration partition is separate from selection and test. Calibration origins are "
    "thinned to limit repeated ARIMA fitting, but forecast windows can overlap; intervals are "
    "descriptive empirical scenario bands, not formal conformal coverage guarantees."
)


def load_currency_version(currency):
    versions = [MODEL_VERSION] + [v for v in LEGACY_MODEL_VERSIONS if v != MODEL_VERSION]
    for version in versions:
        meta_path = METADATA_DIR / f"metadata_{version}_{currency}.json" if version != "v1" else METADATA_DIR / f"metadata_{currency}.json"
        metric_path = METADATA_DIR / f"metrics_{version}_{currency}.csv" if version != "v1" else METADATA_DIR / f"metrics_{currency}.csv"
        if meta_path.exists() and metric_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = pd.read_csv(metric_path)
            selection_path = METADATA_DIR / f"selection_policy_{version}_{currency}.csv" if version == MODEL_VERSION else None
            selection = pd.read_csv(selection_path) if selection_path and selection_path.exists() else None
            selection_metrics_path = METADATA_DIR / f"selection_metrics_{version}_{currency}.csv" if version == MODEL_VERSION else None
            selection_metrics = pd.read_csv(selection_metrics_path) if selection_metrics_path and selection_metrics_path.exists() else None
            return version, metadata, metrics, selection, selection_metrics
    return None


loaded = []
for currency in PRIORITY_CURRENCIES:
    item = load_currency_version(currency)
    if item is not None:
        version, metadata, metrics, selection, selection_metrics = item
        metrics["bundle_version"] = version
        loaded.append((currency, version, metadata, metrics, selection, selection_metrics))

if not loaded:
    st.warning("No trained V2.1, V2 or V1 evaluation artifacts found. Run the training cells in Google Colab first.")
else:
    combined = pd.concat([item[3] for item in loaded], ignore_index=True)
    cnn = combined[combined["model"] == "CNN-LSTM"].copy()
    if "mae_skill_vs_random_walk_pct" not in cnn.columns and "skill_vs_random_walk_pct" in cnn.columns:
        cnn["mae_skill_vs_random_walk_pct"] = cnn["skill_vs_random_walk_pct"]
    if "mae_skill_ci_lower_pct" not in cnn.columns and "skill_ci_lower_pct" in cnn.columns:
        cnn["mae_skill_ci_lower_pct"] = cnn["skill_ci_lower_pct"]
        cnn["mae_skill_ci_upper_pct"] = cnn["skill_ci_upper_pct"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Currencies with evaluation", len(loaded))
    c2.metric("V2.1 bundles", sum(1 for item in loaded if item[1] == MODEL_VERSION))
    c3.metric("Forecasting methods", int(combined["model"].nunique()))

    st.subheader("CNN-LSTM skill against random walk")
    if not cnn.empty and "mae_skill_vs_random_walk_pct" in cnn.columns:
        chart = px.bar(
            cnn, x="horizon", y="mae_skill_vs_random_walk_pct", color="currency",
            barmode="group", text_auto=".2f",
            labels={"horizon": "Forecast horizon (observations)",
                    "mae_skill_vs_random_walk_pct": "MAE improvement vs random walk (%)"},
            title="Positive values favour CNN-LSTM; negative values favour random walk",
        )
        chart.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Final held-out test benchmark comparison")
    display = combined.copy()
    if "directional_accuracy_pct" in display.columns:
        display.loc[display["model"] == "Random walk", "directional_accuracy_pct"] = np.nan
    wanted = [
        "bundle_version", "currency", "horizon", "model", "n_origins", "mae", "rmse",
        "mape_pct", "directional_accuracy_pct", "mae_skill_vs_random_walk_pct",
        "mae_skill_ci_lower_pct", "mae_skill_ci_upper_pct",
        "rmse_skill_vs_random_walk_pct", "rmse_skill_ci_lower_pct", "rmse_skill_ci_upper_pct",
    ]
    # Legacy V2 artifacts have the original MAE-only skill columns.
    for col in wanted:
        if col not in display.columns:
            if col == "mae_skill_vs_random_walk_pct" and "skill_vs_random_walk_pct" in display.columns:
                display[col] = display["skill_vs_random_walk_pct"]
            elif col == "mae_skill_ci_lower_pct" and "skill_ci_lower_pct" in display.columns:
                display[col] = display["skill_ci_lower_pct"]
            elif col == "mae_skill_ci_upper_pct" and "skill_ci_upper_pct" in display.columns:
                display[col] = display["skill_ci_upper_pct"]
            else:
                display[col] = np.nan
    st.dataframe(
        display[wanted].sort_values(["bundle_version", "currency", "horizon", "model"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "mae": st.column_config.NumberColumn(format="%.8f"),
            "rmse": st.column_config.NumberColumn(format="%.8f"),
            "mape_pct": st.column_config.NumberColumn(format="%.4f%%"),
            "directional_accuracy_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "mae_skill_vs_random_walk_pct": st.column_config.NumberColumn(format="%.3f%%"),
            "mae_skill_ci_lower_pct": st.column_config.NumberColumn(format="%.3f%%"),
            "mae_skill_ci_upper_pct": st.column_config.NumberColumn(format="%.3f%%"),
            "rmse_skill_vs_random_walk_pct": st.column_config.NumberColumn(format="%.3f%%"),
            "rmse_skill_ci_lower_pct": st.column_config.NumberColumn(format="%.3f%%"),
            "rmse_skill_ci_upper_pct": st.column_config.NumberColumn(format="%.3f%%"),
        },
    )

    st.subheader("Validation-selected forecasting methods")
    policy_frames = [selection.assign(bundle_version=version) for _, version, _, _, selection, _ in loaded if selection is not None and not selection.empty]
    if policy_frames:
        policy_display = pd.concat(policy_frames, ignore_index=True)
        st.dataframe(
            policy_display[[
                "bundle_version", "currency", "horizon", "selected_model",
                "selection_mae", "selection_rmse", "n_selection_origins",
                "selection_fallback", "selection_reason",
            ]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No V2.1 selection policies are available yet. Legacy V2/V1 bundles use CNN-LSTM as a compatibility fallback.")

    for currency, version, metadata, metrics, selection, selection_metrics in loaded:
        with st.expander(f"{currency} — {version} — {metadata.get('trained_at_utc', 'training date unavailable')}"):
            a, b, c = st.columns(3)
            a.metric("Latest observation", metadata.get("latest_observation_date", "—"))
            b.metric("Latest reference rate", f"{metadata.get('latest_rate', float('nan')):.8f}")
            b.caption(f"MYR per 1 {currency}")
            c.metric("Independent test origins", metadata.get("walk_forward_origins", 0))
            st.write(
                "Train / early-stop validation / selection / calibration samples:",
                metadata.get("train_samples"), "/", metadata.get("validation_samples"), "/",
                metadata.get("selection_samples", "not available"), "/", metadata.get("calibration_samples"),
            )
            st.write("Architecture:", metadata.get("architecture", ""))
            st.write("Selection methodology:", metadata.get("selection_method", "Legacy V2: no separate selection policy."))
            st.write("Calibration:", metadata.get("calibration_method", "Not available."))
            if selection is not None and not selection.empty:
                st.write("Selected methods:")
                st.dataframe(selection[["horizon", "selected_model", "n_selection_origins", "selection_reason"]],
                             use_container_width=True, hide_index=True)
            currency_cnn = metrics[metrics["model"] == "CNN-LSTM"]
            skill_col = "mae_skill_vs_random_walk_pct" if "mae_skill_vs_random_walk_pct" in currency_cnn.columns else "skill_vs_random_walk_pct"
            beats = int((currency_cnn[skill_col] > 0).sum()) if skill_col in currency_cnn.columns else 0
            st.write(f"CNN-LSTM has lower MAE than random walk at {beats} of {len(currency_cnn)} reported test horizons.")

st.subheader("Data source and limitations")
st.markdown(
    "- Source: Bank Negara Malaysia via data.gov.my, Daily Exchange Rates (1200).\n"
    "- Selected rate type: middle, where available.\n"
    "- Quotation: MYR per one unit of foreign currency.\n"
    "- Validation-selected methods are chosen before final test evaluation; final test metrics are not used to select a method.\n"
    "- Test-origin counts are small and must be considered when interpreting performance and confidence intervals.\n"
    "- Random walk predicts no directional change; directional accuracy is not applicable.\n"
    "- Calibration bands are empirical estimates, not guaranteed coverage intervals.\n"
    "- Central-bank reference rates are not necessarily the transaction rates offered by banks.\n"
    "- Forecasts are estimates, not financial advice or guaranteed outcomes."
)
st.markdown("[Official dataset documentation](https://data.gov.my/data-catalogue/exchangerates_daily_1200)")
