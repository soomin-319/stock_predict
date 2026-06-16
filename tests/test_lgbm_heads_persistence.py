from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.lgbm_heads import MODEL_ARTIFACT_VERSION, MultiHeadStockModel


def _make_training_frame(n: int = 120, feature_cols: list[str] | None = None) -> pd.DataFrame:
    feature_cols = feature_cols or ["f1", "f2", "f3"]
    rng = np.random.default_rng(0)
    data = {c: rng.normal(size=n) for c in feature_cols}
    df = pd.DataFrame(data)
    df["target_log_return"] = df["f1"] * 0.1 + rng.normal(scale=0.01, size=n)
    df["target_up"] = (df["target_log_return"] > 0).astype(int)
    return df


def _fit_model() -> tuple[MultiHeadStockModel, pd.DataFrame, list[str]]:
    feature_cols = ["f1", "f2", "f3"]
    df = _make_training_frame(feature_cols=feature_cols)
    model = MultiHeadStockModel(random_state=7, n_jobs=1, head_n_jobs=2)
    model.fit(df, feature_cols, quantiles=[0.1, 0.5, 0.9])
    return model, df, feature_cols


def test_save_and_load_roundtrip_preserves_predictions(tmp_path: Path):
    model, df, _ = _fit_model()
    before = model.predict(df)

    artifact = tmp_path / "model.joblib"
    saved = model.save(artifact)
    assert saved == artifact
    assert artifact.exists()

    reloaded = MultiHeadStockModel.load(artifact)
    assert reloaded.head_n_jobs == model.head_n_jobs
    after = reloaded.predict(df)

    np.testing.assert_allclose(before.predicted_return, after.predicted_return)
    np.testing.assert_allclose(before.up_probability, after.up_probability)
    np.testing.assert_allclose(before.quantile_low, after.quantile_low)
    np.testing.assert_allclose(before.quantile_high, after.quantile_high)


def test_save_writes_sidecar_metadata(tmp_path: Path):
    model, *_ = _fit_model()
    artifact = tmp_path / "model.joblib"
    model.save(artifact)

    meta_path = artifact.with_suffix(artifact.suffix + ".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["artifact_version"] == MODEL_ARTIFACT_VERSION
    assert meta["random_state"] == 7
    assert meta["head_n_jobs"] == 2
    assert meta["feature_count"] == 3
    assert meta["feature_hash"]
    assert 0.1 in meta["quantiles"] and 0.9 in meta["quantiles"]


def test_save_refuses_unfitted_model(tmp_path: Path):
    model = MultiHeadStockModel(random_state=7)
    with pytest.raises(RuntimeError, match="unfitted"):
        model.save(tmp_path / "ghost.joblib")


def test_load_detects_version_mismatch(tmp_path: Path, monkeypatch):
    model, *_ = _fit_model()
    artifact = tmp_path / "model.joblib"
    model.save(artifact)

    # Simulate a newer / incompatible saved artifact.
    import joblib
    payload = joblib.load(artifact)
    payload["artifact_version"] = 999
    joblib.dump(payload, artifact)

    with pytest.raises(ValueError, match="Unsupported model artifact version"):
        MultiHeadStockModel.load(artifact)


def test_predict_rejects_missing_feature_columns():
    model, df, _ = _fit_model()
    partial = df.drop(columns=["f2"])
    with pytest.raises(ValueError, match="missing .* feature columns"):
        model.predict(partial)


def test_model_ignores_legacy_multi_horizon_targets():
    feature_cols = ["f1", "f2", "f3"]
    df = _make_training_frame(feature_cols=feature_cols)
    df["target_log_return_5d"] = df["target_log_return"] * 5
    df["target_up_5d"] = (df["target_log_return_5d"] > 0).astype(int)
    df["target_log_return_20d"] = df["target_log_return"] * 20
    df["target_up_20d"] = (df["target_log_return_20d"] > 0).astype(int)

    model = MultiHeadStockModel(random_state=7, n_jobs=1)
    model.fit(df, feature_cols, quantiles=[0.1, 0.5, 0.9])
    prediction = model.predict(df)

    assert not hasattr(model, "horizon_reg_models")
    assert not hasattr(model, "horizon_cls_models")
    assert not hasattr(prediction, "horizon_predicted_return")
    assert not hasattr(prediction, "horizon_up_probability")


class _ConstantRegressor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, x):
        return np.resize(self.values, len(x))


class _ConstantClassifier:
    def predict_proba(self, x):
        probs = np.resize(np.asarray([0.6], dtype=float), len(x))
        return np.column_stack([1.0 - probs, probs])


def test_predict_sorts_crossed_quantile_outputs_row_wise():
    model = MultiHeadStockModel(random_state=7)
    model._feature_columns = ["f1"]
    model.reg_model = _ConstantRegressor([0.01, 0.02])
    model.cls_model = _ConstantClassifier()
    model.quantile_models = {
        0.1: _ConstantRegressor([0.30, 0.50]),
        0.5: _ConstantRegressor([0.20, 0.40]),
        0.9: _ConstantRegressor([0.10, 0.60]),
    }

    pred = model.predict(pd.DataFrame({"f1": [1.0, 2.0]}))

    np.testing.assert_allclose(pred.quantile_low, [0.10, 0.40])
    np.testing.assert_allclose(pred.quantile_mid, [0.20, 0.50])
    np.testing.assert_allclose(pred.quantile_high, [0.30, 0.60])
    assert (pred.quantile_low <= pred.quantile_mid).all()
    assert (pred.quantile_mid <= pred.quantile_high).all()


def test_fit_stores_imputer_values_and_predict_uses_them_for_missing_features():
    df = pd.DataFrame(
        {
            "rsi_14": [40.0, np.nan, 60.0, 80.0],
            "f1": [1.0, 2.0, 100.0, np.nan],
            "target_log_return": [-0.1, -0.05, 0.05, 0.1],
            "target_up": [0, 0, 1, 1],
        }
    )
    model = MultiHeadStockModel(random_state=7, n_jobs=1)
    model.fit(df, ["rsi_14", "f1"], quantiles=[0.1, 0.5, 0.9])

    assert model._feature_imputer_values["rsi_14"] == 50.0
    assert model._feature_imputer_values["f1"] == pytest.approx(2.0)

    captured = {}

    class _CaptureRegressor:
        def predict(self, x):
            captured["x"] = x.copy()
            return np.zeros(len(x))

    model.reg_model = _CaptureRegressor()
    model.cls_model = _ConstantClassifier()
    model.quantile_models = {
        0.1: _ConstantRegressor([0.0]),
        0.5: _ConstantRegressor([0.0]),
        0.9: _ConstantRegressor([0.0]),
    }

    model.predict(pd.DataFrame({"rsi_14": [np.nan], "f1": [np.nan]}))

    assert captured["x"].loc[0, "rsi_14"] == 50.0
    assert captured["x"].loc[0, "f1"] == pytest.approx(2.0)


def test_save_and_load_preserves_feature_imputer_values(tmp_path: Path):
    model, *_ = _fit_model()
    artifact = tmp_path / "model.joblib"
    model.save(artifact)

    reloaded = MultiHeadStockModel.load(artifact)

    assert reloaded._feature_imputer_values == model._feature_imputer_values
    meta = json.loads(artifact.with_suffix(artifact.suffix + ".meta.json").read_text(encoding="utf-8"))
    assert meta["imputer_feature_count"] == len(model._feature_columns)


@pytest.mark.parametrize(
    ("frame", "features", "match"),
    [
        (pd.DataFrame({"target_log_return": [0.1], "target_up": [1]}), [], "feature_columns"),
        (
            pd.DataFrame({"f1": [np.nan], "target_log_return": [0.1], "target_up": [1]}),
            ["f1"],
            "No usable training rows",
        ),
        (
            pd.DataFrame({"f1": [1.0, 2.0], "target_log_return": [0.1, 0.2], "target_up": [1, 1]}),
            ["f1"],
            "at least two classes",
        ),
    ],
)
def test_fit_raises_clear_validation_errors(frame, features, match):
    model = MultiHeadStockModel(random_state=7, n_jobs=1)

    with pytest.raises(ValueError, match=match):
        model.fit(frame, features, quantiles=[0.1, 0.5, 0.9])


def test_feature_importance_frame_exports_available_heads():
    model, *_ = _fit_model()

    frame = model.feature_importance_frame()

    assert {"head", "feature", "importance"}.issubset(frame.columns)
    assert {"regression", "classification", "quantile_0.1", "quantile_0.5", "quantile_0.9"}.issubset(
        set(frame["head"])
    )
    assert frame["importance"].notna().all()
