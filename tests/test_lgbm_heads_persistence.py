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
    model = MultiHeadStockModel(random_state=7, n_jobs=1)
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
