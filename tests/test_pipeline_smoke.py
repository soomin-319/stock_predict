import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import AppConfig
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.models.lgbm_heads import MultiHeadStockModel
from src.pipeline import resolve_output_path, run_pipeline


def make_sample_df(days: int = 320):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    rows = []
    for symbol in ["AAA", "BBB"]:
        price = 100.0
        for d in dates:
            ret = rng.normal(0.0005, 0.02)
            open_p = price
            close_p = price * (1 + ret)
            high = max(open_p, close_p) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_p, close_p) * (1 - abs(rng.normal(0, 0.005)))
            vol = int(rng.integers(100000, 500000))
            rows.append([d, symbol, open_p, high, low, close_p, vol])
            price = close_p
    return pd.DataFrame(rows, columns=["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"])


def test_multihead_prediction_shapes():
    cfg = AppConfig()
    raw = make_sample_df()
    feat = annotate_market_regime(build_features(raw, cfg.feature))
    feature_columns = [
        c
        for c in feat.columns
        if c.startswith(("ret_", "ma_", "close_to_ma_", "vol_"))
        or c
        in {
            "daily_return",
            "gap_return",
            "intraday_return",
            "range_pct",
            "vol_ratio_20",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_14",
            "stoch_k",
            "stoch_d",
            "cci_20",
            "obv",
            "obv_change_5d",
        }
    ]

    train = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(random_state=cfg.training.random_state)
    model.fit(train, feature_columns, cfg.training.quantiles)

    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(3).fillna(0)
    pred = model.predict(latest)

    assert len(pred.predicted_return) == len(latest)
    assert len(pred.up_probability) == len(latest)
    assert (pred.quantile_high >= pred.quantile_low).all()


def test_resolve_output_path_creates_parent(tmp_path):
    out = resolve_output_path(str(tmp_path / "nested" / "predictions.csv"), is_windows=False)
    assert out.parent.exists()


def test_resolve_output_path_windows_tmp_mapping():
    out = resolve_output_path("/tmp/predictions.csv", is_windows=True)
    assert str(out).startswith(tempfile.gettempdir())
    assert out.name == "predictions.csv"


def test_run_pipeline_generates_report_and_figures(tmp_path):
    inp = Path("data/sample_ohlcv.csv")
    out = tmp_path / "predictions.csv"
    rep = tmp_path / "report.json"
    fig = tmp_path / "figures"
    run_pipeline(str(inp), str(out), universe_csv=None, report_json=str(rep), figure_dir=str(fig), use_external=False)

    assert out.exists()
    assert rep.exists()
    payload = json.loads(rep.read_text())
    assert "walk_forward" in payload
    assert "baselines" in payload
    assert "tuned_signal" in payload
    assert "backtest" in payload
    assert "artifacts" in payload
    assert Path(payload["artifacts"]["oof_predictions_csv"]).exists()


def test_external_features_fail_gracefully_without_noise(monkeypatch):
    from src.features.external_features import add_external_market_features

    def _fail(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("yfinance.download", _fail)

    raw = make_sample_df(days=30)
    out = add_external_market_features(raw, ["^GSPC", "^IXIC", "^SOX", "^VIX"])

    assert len(out) == len(raw)
    assert set(raw.columns).issubset(set(out.columns))
    assert not any(c.startswith(("gspc_", "ixic_", "sox_", "vix_")) for c in out.columns)
