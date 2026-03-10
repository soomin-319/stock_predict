import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import AppConfig
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.models.lgbm_heads import MultiHeadStockModel
from src.pipeline import _ensure_universe_size, _split_oof_for_tuning_and_eval, resolve_output_path, run_pipeline


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

    assert model.backend in {"lightgbm", "sklearn"}
    assert len(pred.predicted_return) == len(latest)
    assert len(pred.up_probability) == len(latest)
    assert (pred.quantile_high >= pred.quantile_low).all()


def test_resolve_output_path_creates_parent(tmp_path):
    out = resolve_output_path(str(tmp_path / "nested" / "predictions.csv"), is_windows=False)
    assert out.parent.exists()


def test_resolve_output_path_windows_tmp_mapping():
    out = resolve_output_path("/tmp/predictions.csv", is_windows=True)
    assert out.parent.name == "result"
    assert out.name == "predictions.csv"


def test_run_pipeline_generates_report_and_figures(tmp_path):
    inp = Path("data/sample_ohlcv.csv")
    out = tmp_path / "predictions.csv"
    rep = tmp_path / "report.json"
    fig = tmp_path / "figures"
    run_pipeline(str(inp), str(out), universe_csv=None, report_json=str(rep), figure_dir=str(fig), use_external=False)

    result_dir = Path("result")
    assert result_dir.exists()
    pred_path = result_dir / out.name
    report_path = result_dir / rep.name
    assert pred_path.exists()
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert "walk_forward" in payload
    assert "baselines" in payload
    assert "tuned_signal" in payload
    assert "backtest" in payload
    assert "artifacts" in payload
    assert "external_feature_coverage" in payload
    assert "tuning_samples" in payload
    assert "backtest_samples" in payload
    assert "avg_turnover" in payload["backtest"]
    assert "avg_selected_count" in payload["backtest"]
    assert Path(payload["artifacts"]["oof_predictions_csv"]).exists()
    assert Path(payload["artifacts"]["actual_vs_predicted"]).exists()
    assert Path(payload["artifacts"]["actual_vs_predicted_price"]).exists()
    assert Path(payload["artifacts"]["symbol_summary_csv"]).exists()
    assert Path(payload["artifacts"]["symbol_summary_png"]).exists()
    assert Path(payload["artifacts"]["symbol_level_figure_dir"]).exists()
    assert payload["artifacts"]["symbol_level_figure_count"] > 0

    pred_df = pd.read_csv(pred_path)
    assert "signal_label" in pred_df.columns
    assert pred_df["signal_label"].astype(str).str.contains("신뢰도").all()


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


def test_split_oof_for_tuning_and_eval():
    df = make_sample_df(days=40)
    df = df[["Date", "Symbol"]].copy()
    df["signal_score"] = 0.1
    df["target_log_return"] = 0.0

    tune_df, eval_df = _split_oof_for_tuning_and_eval(df, tune_ratio=0.7)

    assert not tune_df.empty
    assert not eval_df.empty
    assert tune_df["Date"].max() < eval_df["Date"].min()


def test_external_feature_coverage_fields(monkeypatch):
    from src.features.external_features import add_external_market_features_with_coverage

    def _fail(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("yfinance.download", _fail)

    raw = make_sample_df(days=20)
    out, coverage = add_external_market_features_with_coverage(raw, ["^GSPC", "^IXIC"]) 

    assert len(out) == len(raw)
    assert coverage["requested"] == 2
    assert coverage["successful"] == 0
    assert coverage["failed"] == 2


def test_uncertainty_score_uses_percentile_scale():
    from src.inference.predict import build_prediction_frame
    from src.models.lgbm_heads import MultiHeadPrediction

    cfg = AppConfig()
    latest = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "Symbol": ["A", "B", "C"],
            "Close": [100.0, 101.0, 99.0],
            "market_regime": ["neutral", "neutral", "neutral"],
        }
    )
    pred = MultiHeadPrediction(
        predicted_return=np.array([0.01, 0.0, -0.01]),
        up_probability=np.array([0.6, 0.5, 0.4]),
        quantile_low=np.array([-0.02, -0.01, -0.03]),
        quantile_mid=np.array([0.0, 0.0, 0.0]),
        quantile_high=np.array([0.03, 0.01, 0.02]),
    )
    out = build_prediction_frame(latest, pred, cfg.signal)

    assert (out["uncertainty_score"] > 0).all()
    assert (out["uncertainty_score"] <= 1).all()




def test_normalize_user_symbols_parses_codes():
    from src.data.fetch_real_data import normalize_user_symbols

    out = normalize_user_symbols(["005930", "000660.KS", "035420, 207940"])
    assert "000660.KS" in out
    assert "005930.KS" in out or "005930.KQ" in out
    assert any(x.startswith("035420.") for x in out)


def test_append_real_ohlcv_csv_merges_without_duplicates(tmp_path, monkeypatch):
    from src.data.fetch_real_data import append_real_ohlcv_csv

    target = tmp_path / "real_ohlcv.csv"
    base = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Symbol": ["AAA", "AAA"],
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [100, 100],
        }
    )
    base.to_csv(target, index=False)

    import src.data.fetch_real_data as fr

    def _mock_fetch(symbols, start="2020-01-01", end=None):
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "Symbol": ["AAA", "BBB"],
                "Open": [2, 3],
                "High": [2, 3],
                "Low": [2, 3],
                "Close": [2, 3],
                "Volume": [200, 300],
            }
        )

    monkeypatch.setattr(fr, "fetch_real_ohlcv", _mock_fetch)
    append_real_ohlcv_csv(target, ["AAA", "BBB"])

    out = pd.read_csv(target)
    assert len(out) == 3
    assert set(out["Symbol"]) == {"AAA", "BBB"}


def test_append_real_ohlcv_csv_no_data_does_not_crash(tmp_path, monkeypatch):
    from src.data.fetch_real_data import append_real_ohlcv_csv
    import src.data.fetch_real_data as fr

    target = tmp_path / "real_ohlcv.csv"
    base = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"]),
            "Symbol": ["AAA"],
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [100],
        }
    )
    base.to_csv(target, index=False)

    def _raise(*args, **kwargs):
        raise RuntimeError("No data fetched from yfinance")

    monkeypatch.setattr(fr, "fetch_real_ohlcv", _raise)
    out_path = append_real_ohlcv_csv(target, ["ZZZ"])

    assert out_path == target
    out = pd.read_csv(target)
    assert len(out) == 1
    assert out.loc[0, "Symbol"] == "AAA"


def test_fetch_real_ohlcv_falls_back_to_pykrx(monkeypatch):
    import src.data.fetch_real_data as fr

    def _empty_download(*args, **kwargs):
        return pd.DataFrame()

    class _MockStock:
        @staticmethod
        def get_market_ohlcv_by_date(start, end, ticker):
            idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
            return pd.DataFrame(
                {
                    "시가": [1, 2],
                    "고가": [1, 2],
                    "저가": [1, 2],
                    "종가": [1, 2],
                    "거래량": [100, 200],
                },
                index=idx,
            )

    monkeypatch.setattr(fr, "_safe_download_ohlcv", _empty_download)
    monkeypatch.setattr(fr, "_import_pykrx_stock", lambda: _MockStock)

    out = fr.fetch_real_ohlcv(["005930.KS"], start="2024-01-01")

    assert not out.empty
    assert set(["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]).issubset(out.columns)
    assert out["Symbol"].iloc[0] == "005930.KS"
