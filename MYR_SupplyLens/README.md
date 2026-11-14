# MYR SupplyLens V2.1.1 — Runtime-Safe Colab Rewrite

**Predict. Plan. Procure. Profit.**

V2.1.1 preserves the V2.1 benchmark-aware forecasting implementation and rewrites its Colab setup so it does not attempt to restart or kill the kernel programmatically.

## Critical runtime-safety changes

- No `google.colab.runtime.restart_runtime()` call.
- No `IPython.kernel.do_shutdown()` call.
- No `os.kill`, `SIGKILL`, or process termination fallback.
- No `--force-reinstall` and no `--no-deps` replacement of the scientific stack.
- TensorFlow is never installed or upgraded by the Colab notebook.
- The notebook reads the installed Colab versions of NumPy, pandas, SciPy, scikit-learn, statsmodels, TensorFlow, Keras, h5py, protobuf and ml-dtypes, then writes them into a temporary pip constraints file.
- Project requirements are installed under those constraints, preventing pip from silently replacing the runtime's core scientific packages.
- The notebook verifies that those core versions did not change after installation.
- If a core dependency is missing or a resolver conflict occurs, the notebook stops with a readable error. It does not attempt a restart.
- If a manual restart is ever necessary, the user must use Colab's UI: **Runtime → Restart session**. The notebook never performs this action itself.

## Apply to Google Drive

1. Download `MYR_SupplyLens_V2_1_1_Patch.zip` and `MYR_SupplyLens_V2_1_1_Colab.ipynb`.
2. Upload both files to the root of Google Drive (`My Drive`).
3. Open the new V2.1.1 notebook in Colab. If the old runtime disconnected, connect a fresh standard CPU or GPU runtime first. Do not use a destructive factory reset.
4. Run the notebook from top to bottom. The source overlay backs up only files that differ before replacing them.
5. The patch overlays source, scripts, tests and documentation only. It does not extract into, delete, or overwrite `data/` or `artifacts/`.
6. Run the dependency preflight and import smoke test. The notebook does not automatically restart the runtime.
7. Run unit tests and inspect the data.
8. Set `RUN_TRAINING = True` when ready. Start with USD, then CNY and IDR.
9. Set `LAUNCH_STREAMLIT = True` to launch the application.

## Colab dependency strategy

The Colab notebook does not blindly pin the runtime to older package versions. It locks the versions already installed in that specific runtime and installs only requirements that are missing or outside the project's broad compatibility ranges. This is intentional: Colab updates its preinstalled packages over time, and forcing a NumPy/SciPy downgrade while the kernel is alive can leave stale compiled modules in memory.

The notebook does not install TensorFlow. Standard Colab CPU/GPU runtimes provide TensorFlow. If TensorFlow is absent, choose a standard CPU/GPU runtime rather than a TPU runtime; the notebook will not replace TensorFlow automatically.

If the scientific import smoke test fails after package installation, do not rerun pip or use a process-kill command. Use **Runtime → Restart session** manually, then rerun the notebook from the first cell. This is a manual fallback only, not an automatic notebook action.

## V2.1 model/evaluation design (unchanged)

- Chronological split: 60% train, 10% early-stopping validation, 10% method selection, 5% calibration, 15% untouched final test.
- Nine features: log rate, 1/5/10/20-session momentum, 5/20/60-session volatility, and 20-session log-rate range.
- Five forecasting methods: CNN-LSTM, random walk, historical drift (252), recent drift (20), and ARIMA(1,1,1).
- Selection is per currency and displayed horizon using the separate selection partition, never final test metrics.
- Selection rule: lowest selection-partition MAE; RMSE and absolute-error standard deviation are tie-breakers. If fewer than eight usable origins are available, random walk is the conservative fallback.
- Final test origins use common 63-observation spacing.
- MAE and RMSE skill are reported separately, with separate paired-bootstrap confidence intervals.
- Random-walk directional accuracy is N/A because random walk predicts no directional change.
- Calibration bands are empirical scenario ranges, not guaranteed coverage intervals.

## Artifact preservation

V2.1 outputs continue to use `_v2_1_` filenames. V2 and V1 model bundles remain as fallbacks. This V2.1.1 release is a Colab runtime/dependency repair; it does not change the model architecture, feature schema, evaluation protocol or artifact naming. You do not need to retrain solely because of this dependency hotfix if your V2.1 model bundles already exist.

The patch does not contain or overwrite trained model weights, scalers, raw/processed data, SQLite records, saved supplier details or invoice exposures.

## Training commands

```bash
python scripts/train_model.py --currency USD --epochs 40
python scripts/train_model.py --currency CNY --epochs 40
python scripts/train_model.py --currency IDR --epochs 40
```

If ARIMA is too slow, temporarily skip it:

```bash
python scripts/train_model.py --currency USD --epochs 40 --skip-arima
```

## Data and limitations

Bank Negara Malaysia daily middle reference rates via data.gov.my, CC BY 4.0. Quotation is MYR per one unit of foreign currency. Central-bank reference rates are not necessarily transaction rates offered to SMEs. Small independent test-origin counts require cautious interpretation. Forecasts are estimates, not financial advice or guaranteed outcomes.
