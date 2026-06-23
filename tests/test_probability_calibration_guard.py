import numpy as np
import pandas as pd

from src.validation.support import (
    calibrate_up_probability,
    calibration_split_metrics,
    fit_up_probability_calibrator,
)


def test_calibration_guard_blends_when_isotonic_collapses_unique_values():
    oof = pd.DataFrame(
        {
            "up_probability": [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 0.90],
            "target_log_return": [-0.02, -0.01, -0.01, -0.005, 0.005, 0.01, 0.02],
        }
    )
    raw = pd.Series([0.11, 0.14, 0.19, 0.21, 0.88], dtype=float)

    calibrated = calibrate_up_probability(oof, raw)

    assert calibrated.nunique() >= 3
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_calibration_returns_raw_when_oof_not_usable():
    raw = pd.Series([0.2, 0.4, 0.6], dtype=float)
    out = calibrate_up_probability(pd.DataFrame(), raw)
    pd.testing.assert_series_equal(out, raw)


def test_fitted_calibrator_does_not_depend_on_eval_targets():
    tune = pd.DataFrame(
        {
            "up_probability": [0.1, 0.3, 0.5, 0.7, 0.9],
            "target_log_return": [-0.02, -0.01, 0.01, 0.02, 0.03],
        }
    )
    eval_a = pd.DataFrame({"up_probability": [0.2, 0.8], "target_log_return": [-0.1, 0.1]})
    eval_b = eval_a.assign(target_log_return=-eval_a["target_log_return"])

    calibrator = fit_up_probability_calibrator(tune)

    pd.testing.assert_series_equal(
        calibrator.transform(eval_a["up_probability"]),
        calibrator.transform(eval_b["up_probability"]),
    )


def test_calibration_report_separates_tune_and_eval_metrics():
    tune = pd.DataFrame({"up_probability": [0.1, 0.9] * 10, "target_log_return": [-0.01, 0.01] * 10})
    eval_df = pd.DataFrame({"up_probability": [0.2, 0.8] * 10, "target_log_return": [-0.01, 0.01] * 10})

    calibrator = fit_up_probability_calibrator(tune)
    report = calibration_split_metrics(tune, eval_df, calibrator)

    assert set(report) == {"fit", "tune", "eval"}
    assert report["tune"]["sample_count"] == 20
    assert report["eval"]["sample_count"] == 20
