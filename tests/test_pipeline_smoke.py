import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import AppConfig
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.models.lgbm_heads import MultiHeadStockModel
from src.pipeline import _drop_empty_detail_columns, _split_oof_for_tuning_and_eval, build_cli_parser, resolve_output_path, run_pipeline
from src.pipeline_support import PredictionFrameContext, build_scored_prediction_frame


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


def test_drop_empty_detail_columns_removes_only_empty_optional_fields():
    detail_df = pd.DataFrame(
        [
            {
                "Date": "2026-03-28",
                "Symbol": "005930.KS",
                "foreign_net_buy": np.nan,
                "news_sentiment": np.nan,
                "target_up_20d": np.nan,
                "predicted_return": 0.012,
            }
        ]
    )

    cleaned = _drop_empty_detail_columns(detail_df)

    assert "foreign_net_buy" not in cleaned.columns
    assert "news_sentiment" not in cleaned.columns
    assert "target_up_20d" not in cleaned.columns
    assert "predicted_return" in cleaned.columns


def test_run_pipeline_generates_report_and_figures(tmp_path):
    inp = Path("data/sample_ohlcv.csv")
    out = tmp_path / "predictions.csv"
    rep = tmp_path / "report.json"
    fig = tmp_path / "figures"
    run_pipeline(str(inp), str(out), universe_csv=None, report_json=str(rep), figure_dir=str(fig), use_external=False)

    result_dir = Path("result")
    assert result_dir.exists()
    detail_path = result_dir / "result_detail.csv"
    simple_path = result_dir / "result_simple.csv"
    report_path = result_dir / rep.name
    assert detail_path.exists()
    assert simple_path.exists()
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert "walk_forward" in payload
    assert "baselines" in payload
    assert "tuned_signal" in payload
    assert "backtest" in payload
    assert "artifacts" in payload
    assert "external_feature_coverage" in payload
    assert "probability_calibration" in payload
    assert "ece" in payload["probability_calibration"]
    assert "brier" in payload["probability_calibration"]
    assert "tuning_samples" in payload
    assert "backtest_samples" in payload
    assert "avg_turnover" in payload["backtest"]
    assert "avg_selected_count" in payload["backtest"]
    assert Path(payload["artifacts"]["result_detail_csv"]).exists()
    assert Path(payload["artifacts"]["result_simple_csv"]).exists()
    assert Path(payload["artifacts"]["result_disclosure_csv"]).exists()
    assert Path(payload["artifacts"]["actual_vs_predicted"]).exists()
    assert Path(payload["artifacts"]["actual_vs_predicted_price"]).exists()
    assert Path(payload["artifacts"]["pm_report_json"]).exists()
    assert Path(payload["artifacts"]["symbol_summary_png"]).exists()

    news_df = pd.read_csv(payload["artifacts"]["result_news_csv"])
    disclosure_df = pd.read_csv(payload["artifacts"]["result_disclosure_csv"])
    assert "뉴스 요약" in news_df.columns
    assert "공시 요약" in disclosure_df.columns
    assert Path(payload["artifacts"]["symbol_level_figure_dir"]).exists()
    assert payload["artifacts"]["symbol_level_figure_count"] > 0
    assert Path(payload["artifacts"]["symbol_level_recent_month_dir"]).exists()
    assert payload["artifacts"]["symbol_level_recent_month_figure_count"] > 0
    detail_df = pd.read_csv(detail_path)
    simple_df = pd.read_csv(simple_path)
    assert "signal_label" in detail_df.columns
    assert detail_df["signal_label"].astype(str).isin(
        ["strong_negative", "weak_negative", "neutral", "weak_positive", "strong_positive"]
    ).all()
    assert "confidence_label" in detail_df.columns
    assert detail_df["confidence_label"].astype(str).str.contains("신뢰도").all()
    assert "history_direction_accuracy" in detail_df.columns
    assert "risk_flag" in detail_df.columns
    assert "position_size_hint" in detail_df.columns
    assert "portfolio_action" in detail_df.columns
    assert "trading_gate" in detail_df.columns
    assert "backtest_cum_return" in detail_df.columns
    assert "backtest_sharpe" in detail_df.columns
    assert "backtest_benchmark_cum_return" in detail_df.columns
    assert "backtest_excess_cum_return" in detail_df.columns
    assert "predicted_return_5d" in detail_df.columns
    assert "predicted_return_20d" in detail_df.columns
    assert "up_probability_5d" in detail_df.columns
    assert "up_probability_20d" in detail_df.columns
    assert "coverage_gate_status" in detail_df.columns
    assert "foreign_net_buy" in detail_df.columns
    assert "institution_net_buy" in detail_df.columns
    assert "내일 예상 종가" in detail_df.columns
    assert "상승확률(%)" in detail_df.columns
    assert "5일 예상 수익률(%)" in detail_df.columns
    assert "20일 예상 수익률(%)" in detail_df.columns
    assert "disclosure_score" in detail_df.columns
    assert "news_sentiment" in detail_df.columns
    assert "news_relevance_score" in detail_df.columns
    assert "news_impact_score" in detail_df.columns
    assert "news_article_count" in detail_df.columns
    assert "is_top_turnover_10" in detail_df.columns
    assert "smart_money_buy_signal" in detail_df.columns
    assert "close_to_52w_high" in detail_df.columns
    assert "near_52w_high_flag" in detail_df.columns
    assert "breakout_52w_flag" in detail_df.columns
    assert "program_trading_flow" not in detail_df.columns
    assert "warning_level" not in detail_df.columns
    assert "short_sell_ratio" not in detail_df.columns
    assert "buyback_flag" not in detail_df.columns
    assert "종목코드" in simple_df.columns
    assert "종목명" in simple_df.columns
    assert "권고" in simple_df.columns
    assert "포트폴리오 액션" not in simple_df.columns
    assert "거래 게이트" not in simple_df.columns
    assert "내일 예상 종가" in simple_df.columns
    assert "내일 예상 수익률(%)" in simple_df.columns
    assert "5일 예상 수익률(%)" in simple_df.columns
    assert "20일 예상 수익률(%)" in simple_df.columns
    assert "상승확률(%)" in simple_df.columns
    assert "5일 상승확률(%)" in simple_df.columns
    assert "20일 상승확률(%)" in simple_df.columns
    assert "예측 신뢰도" in simple_df.columns
    assert "예측 이유" in simple_df.columns


def test_build_cli_parser_uses_project_relative_defaults_and_exposes_coverage_override():
    parser = build_cli_parser()
    args = parser.parse_args([])

    assert args.output == "result_detail.csv"
    assert args.report_json == "pipeline_report.json"
    assert args.figure_dir == "figures"
    assert hasattr(args, "min_external_coverage_ratio")
    assert hasattr(args, "min_investor_coverage_ratio")
    assert hasattr(args, "portfolio_value")
    assert hasattr(args, "max_daily_participation")


def test_build_scored_prediction_frame_keeps_signal_label_separate_from_confidence_context():
    from src.models.lgbm_heads import MultiHeadPrediction

    cfg = AppConfig()
    latest = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Symbol": ["A", "B"],
            "Close": [100.0, 101.0],
            "market_regime": ["neutral", "neutral"],
            "target_log_return": [0.01, -0.02],
            "value_traded": [5_000_000_000.0, 6_000_000_000.0],
            "turnover_rank_daily": [1, 20],
            "foreign_net_buy": [120_000_000_000.0, -10.0],
            "institution_net_buy": [120_000_000_000.0, -20.0],
            "nq_f_ret_1d": [0.02, -0.02],
            "rsi_14": [45.0, 72.0],
            "near_52w_high_flag": [1, 0],
            "breakout_52w_flag": [1, 0],
            "leader_confirmation_flag": [1, 0],
        }
    )
    pred = MultiHeadPrediction(
        predicted_return=np.array([0.02, -0.01]),
        up_probability=np.array([0.7, 0.4]),
        quantile_low=np.array([-0.01, -0.03]),
        quantile_mid=np.array([0.01, -0.01]),
        quantile_high=np.array([0.04, 0.01]),
        horizon_predicted_return={5: np.array([0.05, -0.02]), 20: np.array([0.1, -0.03])},
        horizon_up_probability={5: np.array([0.72, 0.38]), 20: np.array([0.75, 0.35])},
    )

    scored = build_scored_prediction_frame(
        latest,
        pred,
        cfg.signal,
        PredictionFrameContext(external_coverage_ratio=0.8, investor_coverage_ratio=0.7, min_liquidity_threshold=3_000_000_000.0),
    )

    assert "signal_label" in scored.columns
    assert scored["signal_label"].astype(str).isin(
        ["strong_negative", "weak_negative", "neutral", "weak_positive", "strong_positive"]
    ).all()
    assert "confidence_label" not in scored.columns
    assert "predicted_return_5d" in scored.columns
    assert "predicted_return_20d" in scored.columns
    assert "up_probability_5d" in scored.columns
    assert "up_probability_20d" in scored.columns
    assert scored["external_coverage_ratio"].eq(0.8).all()
    assert scored["investor_coverage_ratio"].eq(0.7).all()


def test_run_pipeline_promotes_investor_context_to_separate_progress_step(tmp_path, monkeypatch, capsys):
    inp = Path("data/sample_ohlcv.csv")
    out = tmp_path / "predictions.csv"
    rep = tmp_path / "report.json"
    fig = tmp_path / "figures"

    def _fake_add_context(data, config):
        coverage = {
            "enabled": True,
            "flow": {"requested": 1, "successful": 1, "failed": 0},
            "disclosure": {"requested": 1, "successful": 1, "failed": 0},
            "news": {"requested": 1, "successful": 1, "failed": 0},
        }
        return data.copy(), coverage

    monkeypatch.setattr("src.pipeline.add_investor_context_with_coverage", _fake_add_context)

    run_pipeline(
        str(inp),
        str(out),
        universe_csv=None,
        report_json=str(rep),
        figure_dir=str(fig),
        use_external=False,
        use_investor_context=True,
    )

    out_text = capsys.readouterr().out
    assert "[4/13] Adding investor context" in out_text
    assert "[5/13] Building price features" in out_text
    assert "[6/13] Adding external market features" in out_text


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


def test_rel_strength_is_not_duplicate_of_norm_return():
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
        predicted_return=np.array([0.02, 0.0, -0.01]),
        up_probability=np.array([0.6, 0.5, 0.4]),
        quantile_low=np.array([-0.02, -0.01, -0.03]),
        quantile_mid=np.array([0.0, 0.0, 0.0]),
        quantile_high=np.array([0.03, 0.01, 0.02]),
    )

    out = build_prediction_frame(latest, pred, cfg.signal)

    assert not out["rel_strength"].equals(out["norm_return"])




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
