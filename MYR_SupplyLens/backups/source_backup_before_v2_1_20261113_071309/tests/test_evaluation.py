import numpy as np
import pandas as pd
from src.supplylens.evaluation import evaluate_walk_forward


class DummyScaler:
    def inverse_transform(self, values):
        return values


class DummyModel:
    def predict(self, X, verbose=0):
        return np.zeros((len(X), 63), dtype=float)


def test_walk_forward_uses_non_overlapping_origins_and_random_walk():
    n = 1500
    rates = 4.0 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.001, n)))
    dates = pd.bdate_range("2015-01-01", periods=n)
    origins = np.arange(1000, 1400)
    X = np.zeros((len(origins), 90, 9), dtype=float)
    metrics, details = evaluate_walk_forward(
        DummyModel(), X, DummyScaler(), origins, rates, dates,
        currency="USD", include_arima=False
    )
    assert set(metrics["model"]) >= {"CNN-LSTM", "Random walk", "Historical drift (252)", "Recent drift (20)"}
    assert details["origin_index"].nunique() <= 7
    assert metrics["n_origins"].nunique() == 1
    rw = metrics[metrics["model"] == "Random walk"]
    assert np.allclose(rw["skill_vs_random_walk_pct"], 0.0)
