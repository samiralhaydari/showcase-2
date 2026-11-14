import numpy as np
import pandas as pd
from src.supplylens.features import make_feature_frame, make_supervised_samples
from src.supplylens.config import FEATURE_COLUMNS


def test_feature_frame_v2_columns():
    dates = pd.bdate_range("2020-01-01", periods=300)
    values = np.linspace(4.0, 4.5, len(dates))
    frame = make_feature_frame(pd.Series(values, index=dates)).dropna()
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) > 0


def test_four_way_split_and_target_boundaries():
    dates = pd.bdate_range("2010-01-01", periods=2500)
    values = 4.0 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.001, len(dates))))
    data = make_supervised_samples(pd.Series(values, index=dates))
    assert all(len(data[name]["X"]) > 0 for name in ["train", "validation", "calibration", "test"])
    assert data["train"]["origins"].max() < data["validation"]["origins"].min()
    assert data["validation"]["origins"].max() < data["calibration"]["origins"].min()
    assert data["calibration"]["origins"].max() < data["test"]["origins"].min()
