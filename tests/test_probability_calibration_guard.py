import numpy as np
import pandas as pd

from src.pipeline import _calibrate_up_probability


def test_calibration_guard_blends_when_isotonic_collapses_unique_values():
    oof = pd.DataFrame(
        {
            "up_probability": [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 0.90],
            "target_log_return": [-0.02, -0.01, -0.01, -0.005, 0.005, 0.01, 0.02],
        }
    )
    raw = pd.Series([0.11, 0.14, 0.19, 0.21, 0.88], dtype=float)

    calibrated = _calibrate_up_probability(oof, raw)

    assert calibrated.nunique() >= 3
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_calibration_returns_raw_when_oof_not_usable():
    raw = pd.Series([0.2, 0.4, 0.6], dtype=float)
    out = _calibrate_up_probability(pd.DataFrame(), raw)
    pd.testing.assert_series_equal(out, raw)
