import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config.settings import AppConfig, SignalConfig
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.models.lgbm_heads import MODEL_ARTIFACT_VERSION, MultiHeadStockModel
from src.pipeline import (
    PIPELINE_STAGE_KEYS,
    PipelineDiagnostics,
    _drop_empty_detail_columns,
    _build_pipeline_feature_matrix,
    _load_pipeline_config_and_data,
    _run_pipeline_validation,
    _predict_pipeline_latest,
    _prepare_pipeline_context,
    _split_oof_for_tuning_and_eval,
    build_cli_parser,
    resolve_output_path,
    run_pipeline,
)
from src.pipeline_support import PredictionFrameContext, build_scored_prediction_frame


def test_graph_module_and_dependency_are_removed():
    assert not Path("src/reports/visualize.py").exists()
    assert not Path("tests/test_visualize_recent_month.py").exists()
    assert "matplotlib" not in Path("requirements.txt").read_text().lower()
    assert "matplotlib" not in Path("pyproject.toml").read_text().lower()


def test_load_pipeline_config_and_data_uses_universe_csv(tmp_path):
    universe_csv = tmp_path / "universe.csv"
    universe_csv.write_text("Symbol\nAAA\n", encoding="utf-8-sig")

    _, _, _, data, requested_symbols = _load_pipeline_config_and_data(
        "data/sample_ohlcv.csv",
        str(universe_csv),
        None,
        {},
    )

    assert requested_symbols == ["AAA"]
    assert sorted(data["Symbol"].unique()) == ["AAA"]


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
    assert not hasattr(pred, "horizon_predicted_return")
    assert not hasattr(pred, "horizon_up_probability")


def test_build_features_creates_only_next_day_targets():
    cfg = AppConfig()
    feat = build_features(make_sample_df(), cfg.feature)

    assert {"target_log_return", "target_up", "target_close"}.issubset(feat.columns)
    assert not any(
        column in feat.columns
        for column in [
            "target_log_return_5d",
            "target_up_5d",
            "target_close_5d",
            "target_log_return_20d",
            "target_up_20d",
            "target_close_20d",
        ]
    )


def test_resolve_output_path_creates_parent(tmp_path):
    out = resolve_output_path(str(tmp_path / "nested" / "predictions.csv"), is_windows=False)
    assert out.parent.exists()


def test_resolve_output_path_windows_tmp_mapping():
    out = resolve_output_path("/tmp/predictions.csv", is_windows=True)
    assert out.parent.name == "result"
    assert out.name == "predictions.csv"


def test_drop_empty_detail_columns_preserves_empty_optional_fields_by_default():
    detail_df = pd.DataFrame(
        [
            {
                "Date": "2026-03-28",
                "Symbol": "005930.KS",
                "foreign_net_buy": np.nan,
                "news_sentiment": np.nan,
                "target_up": np.nan,
                "predicted_return": 0.012,
            }
        ]
    )

    cleaned = _drop_empty_detail_columns(detail_df)

    assert "foreign_net_buy" in cleaned.columns
    assert "news_sentiment" in cleaned.columns
    assert "target_up" in cleaned.columns
    assert "predicted_return" in cleaned.columns


def test_pipeline_diagnostics_records_stage_status_and_warnings():
    diagnostics = PipelineDiagnostics()

    diagnostics.mark_stage("load_config_and_inputs", "ok")
    diagnostics.mark_stage("prepare_context", "skipped", "disabled")
    diagnostics.warn("optional context unavailable")
    diagnostics.validate_stage_coverage()

    report = diagnostics.to_report({"coverage_gate_status": "ok"})

    assert "load_config_and_inputs" in PIPELINE_STAGE_KEYS
    assert report["stage_status"]["load_config_and_inputs"] == {"status": "ok", "reason": ""}
    assert report["stage_status"]["prepare_context"] == {"status": "skipped", "reason": "disabled"}
    assert "optional context unavailable" in report["warnings"]
    assert any("missing stage status" in warning for warning in report["warnings"])


def test_diagnostics_report_includes_resolved_parallelism():
    from src.config.settings import TrainingConfig

    diagnostics = PipelineDiagnostics()
    diagnostics.set_parallelism(
        TrainingConfig(walk_forward_n_jobs=-1, model_n_jobs=2, model_head_n_jobs=1),
        cpu_count=8,
    )

    parallelism = diagnostics.to_report({})["parallelism"]

    assert parallelism["cpu_count"] == 8
    assert parallelism["configured"]["walk_forward_n_jobs"] == -1
    assert parallelism["resolved"]["walk_forward_n_jobs"] == 8
    assert parallelism["resolved"]["model_n_jobs"] == 2
    assert parallelism["resolved"]["model_head_n_jobs"] == 1


def test_training_config_accepts_lightgbm_regularization_overrides(tmp_path):
    from src.config.settings import load_app_config

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "training": {
                    "early_stopping_rounds": 12,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.2,
                    "min_child_samples": 9,
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)

    assert cfg.training.early_stopping_rounds == 12
    assert cfg.training.reg_alpha == pytest.approx(0.1)
    assert cfg.training.reg_lambda == pytest.approx(0.2)
    assert cfg.training.min_child_samples == 9


def test_record_model_metadata_warnings_adds_sklearn_warning():
    from src.pipeline import _record_model_metadata_warnings

    diagnostics = PipelineDiagnostics()

    _record_model_metadata_warnings(diagnostics, {"warnings": ["Model backend is sklearn fallback; install LightGBM"]})

    assert any("sklearn fallback" in warning for warning in diagnostics.to_report({})["warnings"])


def test_run_pipeline_generates_report_without_graph_artifacts(tmp_path):
    inp = Path("data/sample_ohlcv.csv")
    out = tmp_path / "predictions.csv"
    rep = tmp_path / "report.json"
    fig = tmp_path / "figures"
    payload = run_pipeline(
        str(inp),
        str(out),
        universe_csv=None,
        report_json=str(rep),
        figure_dir=str(fig),
        use_external=False,
        data_fetch_coverage={"enabled": True, "requested": 2, "successful": 1},
    )

    result_dir = Path("result")
    assert result_dir.exists()
    detail_path = Path(payload["artifacts"]["result_detail_csv"])
    simple_path = Path(payload["artifacts"]["result_simple_csv"])
    report_path = Path(payload["artifacts"]["pipeline_report_json"])
    assert detail_path.exists()
    assert simple_path.exists()
    assert report_path.exists()
    for frame in (
        pd.read_csv(detail_path, encoding="utf-8-sig"),
        pd.read_csv(simple_path, encoding="utf-8-sig"),
    ):
        assert {
            "environment",
            "data_mode",
            "input_as_of_date",
            "prediction_for_date",
            "context_as_of_date",
        }.issubset(frame.columns)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["run_id"]
    assert payload["environment"] == "smoke"
    assert payload["data_mode"] == "sample"
    assert payload["input_as_of_date"]
    assert payload["prediction_for_date"]
    assert payload["run_id"] in report_path.as_posix()
    manifest_path = report_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == payload["run_id"]
    assert manifest["promoted"] is False
    for key in ("result_news_csv", "result_disclosure_csv"):
        context_frame = pd.read_csv(payload["artifacts"][key], encoding="utf-8-sig")
        assert {
            "record_type",
            "collection_status",
            "no_data_reason",
            "collection_error",
        }.issubset(context_frame.columns)
    pm_payload = json.loads(Path(payload["artifacts"]["pm_report_json"]).read_text(encoding="utf-8"))
    assert pm_payload["run_id"] == payload["run_id"]
    assert "walk_forward" in payload
    assert "baselines" in payload
    assert "config_input" in payload
    assert payload["news_impact_runtime"]["requested_mode"] in {"rule", "gemma", "none"}
    assert payload["news_impact_runtime"]["actual_mode"] in {"rule_based", "gemma", "none"}
    assert isinstance(payload["news_impact_runtime"]["fallback_used"], bool)
    assert "fallback_reason" in payload["news_impact_runtime"]
    assert payload["config_input"] == payload["config"]
    assert "tuned_signal" in payload
    assert set(payload["signal_weights_tuned"]) == set(payload["config"]["signal"])
    assert payload["config_input"]["signal"] != payload["signal_weights_tuned"]
    assert "walk_forward_diagnostics" in payload
    assert payload["walk_forward_diagnostics"]["final_fold_count"] >= 0
    assert "backtest" in payload
    assert "artifacts" in payload
    assert "external_feature_coverage" in payload
    assert payload["data_fetch_coverage"]["requested"] == 2
    assert "probability_calibration" in payload
    assert set(payload["probability_calibration"]) == {"fit", "tune", "eval"}
    assert "ece" in payload["probability_calibration"]["eval"]
    assert "brier" in payload["probability_calibration"]["eval"]
    assert payload["probability_calibration"]["eval"]["sample_count"] == payload["backtest_samples"]
    assert payload["oof_policy"]["duplicate_policy"] == "date_symbol_mean"
    assert payload["oof_policy"]["diagnostics"]["unique_row_count"] > 0
    assert payload["validation_split"]["status"] == "ok"
    assert "tuning_samples" in payload
    assert "backtest_samples" in payload
    diagnostics = payload["diagnostics"]
    assert "timings_seconds" in diagnostics
    assert "row_counts" in diagnostics
    assert "coverage_summary" in diagnostics
    assert "stage_status" in diagnostics
    assert "warnings" in diagnostics
    assert diagnostics["parallelism"]["cpu_count"] >= 1
    assert "walk_forward_n_jobs" in diagnostics["parallelism"]["resolved"]
    assert set(PIPELINE_STAGE_KEYS).issubset(diagnostics["stage_status"])
    assert "feature_missing_rates" in diagnostics
    assert "ma_120" in diagnostics["feature_missing_rates"]
    assert diagnostics["row_counts"]["raw_input"] > 0
    assert diagnostics["row_counts"]["features"] > 0
    assert diagnostics["row_counts"]["oof_predictions"] > 0
    assert diagnostics["row_counts"]["latest_predictions"] > 0
    assert diagnostics["coverage_summary"]["coverage_gate_status"] == payload["coverage_gate"]["status"]
    assert "avg_turnover" in payload["backtest"]
    assert "avg_selected_count" in payload["backtest"]
    assert Path(payload["artifacts"]["result_detail_csv"]).exists()
    assert Path(payload["artifacts"]["result_simple_csv"]).exists()
    assert Path(payload["artifacts"]["result_disclosure_csv"]).exists()
    assert Path(payload["artifacts"]["pm_report_json"]).exists()
    assert Path(payload["artifacts"]["model_artifact"]).exists()
    assert Path(payload["artifacts"]["model_metadata_json"]).exists()
    importance_path = Path(payload["artifacts"]["model_feature_importance_csv"])
    assert importance_path.exists()
    importance_df = pd.read_csv(importance_path, encoding="utf-8-sig")
    assert {"head", "feature", "importance"}.issubset(importance_df.columns)
    assert payload["model"]["feature_count"] == payload["feature_count"]
    assert payload["model"]["feature_hash"]
    assert payload["model"]["artifact_version"] == MODEL_ARTIFACT_VERSION
    assert "visualization_note" not in payload
    assert not any(
        "figure" in key.lower() or "plot" in key.lower() or str(value).lower().endswith(".png")
        for key, value in payload["artifacts"].items()
    )

    news_df = pd.read_csv(payload["artifacts"]["result_news_csv"])
    disclosure_df = pd.read_csv(payload["artifacts"]["result_disclosure_csv"])
    assert "뉴스 요약" in news_df.columns
    assert "공시 요약" in disclosure_df.columns
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
    assert "predicted_return_5d" not in detail_df.columns
    assert "predicted_return_20d" not in detail_df.columns
    assert "up_probability_5d" not in detail_df.columns
    assert "up_probability_20d" not in detail_df.columns
    assert "coverage_gate_status" in detail_df.columns
    assert "foreign_net_buy" in detail_df.columns
    assert "institution_net_buy" in detail_df.columns
    assert "내일 예상 종가" in detail_df.columns
    assert "상승확률(%)" in detail_df.columns
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
    assert "상승확률(%)" in simple_df.columns
    assert "예측 신뢰도" in simple_df.columns
    assert "예측 이유" not in simple_df.columns


def test_build_cli_parser_removes_graph_options():
    parser = build_cli_parser()
    args = parser.parse_args([])

    assert args.output == "result_detail.csv"
    assert args.report_json == "pipeline_report.json"
    assert not hasattr(args, "figure_dir")
    assert hasattr(args, "min_external_coverage_ratio")
    assert hasattr(args, "min_investor_coverage_ratio")
    assert hasattr(args, "portfolio_value")
    assert hasattr(args, "max_daily_participation")
    assert hasattr(args, "walk_forward_n_jobs")
    assert hasattr(args, "model_n_jobs")
    assert hasattr(args, "model_head_n_jobs")
    assert hasattr(args, "context_raw_event_n_jobs")
    assert hasattr(args, "issue_summary_n_jobs")
    assert not hasattr(args, "symbol_figure_limit")
    with pytest.raises(SystemExit):
        parser.parse_args(["--figure-dir", "figures"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--symbol-figure-limit", "30"])


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
    assert "predicted_return_5d" not in scored.columns
    assert "predicted_return_20d" not in scored.columns
    assert "up_probability_5d" not in scored.columns
    assert "up_probability_20d" not in scored.columns
    assert scored["external_coverage_ratio"].eq(0.8).all()
    assert scored["investor_coverage_ratio"].eq(0.7).all()


def test_run_pipeline_promotes_investor_context_to_separate_progress_step(tmp_path, monkeypatch, capsys):
    inp = Path("data/sample_ohlcv.csv")
    out = tmp_path / "predictions.csv"
    rep = tmp_path / "report.json"

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
        use_external=False,
        use_investor_context=True,
    )

    out_text = capsys.readouterr().out
    assert "[4/12] Adding investor context" in out_text
    assert "[5/12] Building price features" in out_text
    assert "[6/12] Adding external market features" in out_text


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

    result = _split_oof_for_tuning_and_eval(df, tune_ratio=0.7)

    assert result.status == "ok"
    assert not result.tune.empty
    assert not result.eval.empty
    assert result.tune["Date"].max() < result.eval["Date"].min()
    assert set(result.tune["Date"]).isdisjoint(set(result.eval["Date"]))


def test_split_oof_reports_insufficient_data_without_reusing_dates():
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=6),
            "Symbol": ["A"] * 6,
            "target_log_return": [0.0] * 6,
        }
    )

    result = _split_oof_for_tuning_and_eval(df, tune_ratio=0.7, min_tune_dates=5, min_eval_dates=3)

    assert result.status == "insufficient_data"
    assert set(result.tune["Date"]).isdisjoint(set(result.eval["Date"]))
    assert result.eval.empty


def test_pipeline_validation_does_not_backtest_when_eval_dates_are_insufficient(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=6)
    oof = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": ["A"] * 6,
            "Close": [100.0] * 6,
            "market_regime": ["normal"] * 6,
            "target_log_return": [-0.01, 0.01] * 3,
            "target_up": [0, 1] * 3,
            "log_return": [0.0] * 6,
            "predicted_return": [-0.005, 0.005] * 3,
            "up_probability": [0.4, 0.6] * 3,
            "quantile_low": [-0.02] * 6,
            "quantile_mid": [0.0] * 6,
            "quantile_high": [0.02] * 6,
        }
    )
    result = SimpleNamespace(
        folds=[SimpleNamespace(metrics={"mae": 0.01})],
        oof=oof,
        oof_diagnostics={"raw_row_count": 6, "unique_row_count": 6},
    )
    monkeypatch.setattr("src.pipeline.walk_forward_validate_result", lambda *_args, **_kwargs: result)

    validation = _run_pipeline_validation(
        feat=oof,
        feature_columns=[],
        cfg=AppConfig(),
        use_external=False,
        external_coverage={},
        investor_context_coverage={},
    )

    assert validation["validation_split"]["status"] == "insufficient_data"
    assert validation["eval_df"].empty
    assert validation["backtest_input"].empty
    assert validation["backtest_status"] == "insufficient_data"


def test_pipeline_validation_does_not_mutate_signal_config(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=12)
    oof = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": ["A"] * 12,
            "Close": [100.0] * 12,
            "market_regime": ["normal"] * 12,
            "target_log_return": [-0.01, 0.01] * 6,
            "target_up": [0, 1] * 6,
            "log_return": [0.0] * 12,
            "predicted_return": [-0.005, 0.005] * 6,
            "up_probability": [0.4, 0.6] * 6,
            "quantile_low": [-0.02] * 12,
            "quantile_mid": [0.0] * 12,
            "quantile_high": [0.02] * 12,
        }
    )
    result = SimpleNamespace(
        folds=[SimpleNamespace(metrics={"mae": 0.01}) for _ in range(3)],
        oof=oof,
        oof_diagnostics={"raw_row_count": 12, "unique_row_count": 12},
    )
    tuned = {
        "return_weight": 0.11,
        "up_prob_weight": 0.22,
        "uncertainty_penalty": 0.44,
    }
    monkeypatch.setattr("src.pipeline.walk_forward_validate_result", lambda *_args, **_kwargs: result)
    monkeypatch.setattr("src.pipeline.evaluate_baselines", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.pipeline.tune_signal_weights", lambda *_args, **_kwargs: tuned)
    cfg = AppConfig()
    original_signal = asdict(cfg.signal)

    validation = _run_pipeline_validation(
        feat=oof,
        feature_columns=[],
        cfg=cfg,
        use_external=False,
        external_coverage={},
        investor_context_coverage={},
    )

    assert asdict(cfg.signal) == original_signal
    tuned_signal = asdict(validation["tuned_signal_config"])
    for key, value in tuned.items():
        assert tuned_signal[key] == value
    assert tuned_signal["recommendation_buy_threshold_pct"] == original_signal["recommendation_buy_threshold_pct"]
    assert tuned_signal["recommendation_sell_threshold_pct"] == original_signal["recommendation_sell_threshold_pct"]


def test_pipeline_validation_retries_when_fold_count_is_too_low(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=12)
    oof = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": ["A"] * 12,
            "Close": [100.0] * 12,
            "market_regime": ["normal"] * 12,
            "target_log_return": [-0.01, 0.01] * 6,
            "target_up": [0, 1] * 6,
            "log_return": [0.0] * 12,
            "predicted_return": [-0.005, 0.005] * 6,
            "up_probability": [0.4, 0.6] * 6,
            "quantile_low": [-0.02] * 12,
            "quantile_mid": [0.0] * 12,
            "quantile_high": [0.02] * 12,
        }
    )
    calls = []

    def fake_walk_forward(*_args, **_kwargs):
        calls.append(1)
        fold_count = 1 if len(calls) == 1 else 3
        return SimpleNamespace(
            folds=[SimpleNamespace(metrics={"mae": 0.01}) for _ in range(fold_count)],
            oof=oof,
            oof_diagnostics={"raw_row_count": 12, "unique_row_count": 12},
        )

    monkeypatch.setattr("src.pipeline.walk_forward_validate_result", fake_walk_forward)
    monkeypatch.setattr("src.pipeline.evaluate_baselines", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "src.pipeline.tune_signal_weights",
        lambda *_args, **_kwargs: {
            "return_weight": 0.11,
            "up_prob_weight": 0.22,
            "uncertainty_penalty": 0.44,
        },
    )

    validation = _run_pipeline_validation(
        feat=oof,
        feature_columns=[],
        cfg=AppConfig(),
        use_external=False,
        external_coverage={},
        investor_context_coverage={},
    )

    assert len(calls) == 2
    assert validation["walk_forward_diagnostics"]["adaptive_retry_used"] is True
    assert validation["walk_forward_diagnostics"]["initial_fold_count"] == 1
    assert validation["walk_forward_diagnostics"]["final_fold_count"] == 3


def test_latest_prediction_accepts_tuned_signal_config(monkeypatch):
    from src.models.lgbm_heads import MultiHeadPrediction

    class IdentityCalibrator:
        def transform(self, values):
            return pd.Series(values)

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            return self

        def predict(self, latest):
            return MultiHeadPrediction(
                predicted_return=np.array([0.01, -0.01]),
                up_probability=np.array([0.5, 0.5]),
                quantile_low=np.array([-0.02, -0.02]),
                quantile_mid=np.array([0.0, 0.0]),
                quantile_high=np.array([0.02, 0.02]),
            )

        def metadata(self):
            return {}

    feat = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Symbol": ["A", "B"],
            "Close": [100.0, 100.0],
            "market_regime": ["normal", "normal"],
            "target_log_return": [0.0, 0.0],
            "target_up": [1, 0],
            "value_traded": [1_000_000.0, 1_000_000.0],
        }
    )
    scored_oof = pd.DataFrame(
        {
            "Symbol": ["A", "B"],
            "target_up": [1, 0],
            "predicted_return": [0.01, -0.01],
        }
    )
    monkeypatch.setattr("src.pipeline.MultiHeadStockModel", FakeModel)
    monkeypatch.setattr("src.pipeline.get_symbol_name_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.pipeline.append_issue_summary_columns", lambda pred_df, **_kwargs: pred_df)
    monkeypatch.setattr(
        "src.pipeline.append_generated_news_impact_context_with_runtime",
        lambda pred_df, *_args, **_kwargs: types.SimpleNamespace(
            frame=pred_df,
            to_metadata=lambda: {
                "requested_mode": "rule",
                "actual_mode": "none",
                "fallback_used": False,
                "fallback_reason": "no_context_rows",
            },
        ),
    )

    pred_df, *_ = _predict_pipeline_latest(
        feat=feat,
        feature_columns=[],
        cfg=AppConfig(),
        scored_oof=scored_oof,
        probability_calibrator=IdentityCalibrator(),
        prediction_context=PredictionFrameContext(
            external_coverage_ratio=1.0,
            investor_coverage_ratio=1.0,
            min_liquidity_threshold=0.0,
        ),
        coverage_gate_status="ok",
        context_raw_df=pd.DataFrame(),
        effective_openai_api_key=None,
        effective_openai_model=None,
        issue_summary_symbols=None,
        issue_summary_n_jobs=1,
        news_impact_report=None,
        signal_config=SignalConfig(
            return_weight=1.0,
            up_prob_weight=0.0,
            uncertainty_penalty=0.0,
        ),
    )

    ordered = pred_df.sort_values("Symbol")
    assert ordered["signal_score"].tolist() == pytest.approx(ordered["norm_return"].tolist())


def test_latest_prediction_uses_custom_recommendation_thresholds(monkeypatch):
    from src.models.lgbm_heads import MultiHeadPrediction

    class IdentityCalibrator:
        def transform(self, values):
            return pd.Series(values)

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            return self

        def predict(self, latest):
            return MultiHeadPrediction(
                predicted_return=np.array([np.log1p(0.025), np.log1p(-0.015)]),
                up_probability=np.array([0.8, 0.2]),
                quantile_low=np.array([0.0, -0.02]),
                quantile_mid=np.array([0.02, -0.01]),
                quantile_high=np.array([0.03, 0.0]),
            )

    feat = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Symbol": ["A", "B"],
            "Close": [100.0, 100.0],
            "market_regime": ["normal", "normal"],
            "target_log_return": [0.0, 0.0],
            "target_up": [1, 0],
            "value_traded": [5_000_000_000.0, 5_000_000_000.0],
        }
    )
    scored_oof = pd.DataFrame(
        {
            "Symbol": ["A", "B"],
            "target_up": [1, 0],
            "predicted_return": [0.01, -0.01],
        }
    )
    monkeypatch.setattr("src.pipeline.MultiHeadStockModel", FakeModel)
    monkeypatch.setattr("src.pipeline.get_symbol_name_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.pipeline.append_issue_summary_columns", lambda pred_df, **_kwargs: pred_df)
    monkeypatch.setattr(
        "src.pipeline.append_generated_news_impact_context_with_runtime",
        lambda pred_df, *_args, **_kwargs: SimpleNamespace(
            frame=pred_df,
            to_metadata=lambda: {
                "requested_mode": "rule",
                "actual_mode": "none",
                "fallback_used": False,
                "fallback_reason": "no_context_rows",
            },
        ),
    )

    pred_df, *_ = _predict_pipeline_latest(
        feat=feat,
        feature_columns=[],
        cfg=AppConfig(),
        scored_oof=scored_oof,
        probability_calibrator=IdentityCalibrator(),
        prediction_context=PredictionFrameContext(
            external_coverage_ratio=1.0,
            investor_coverage_ratio=1.0,
            min_liquidity_threshold=0.0,
        ),
        coverage_gate_status="ok",
        context_raw_df=pd.DataFrame(),
        effective_openai_api_key=None,
        effective_openai_model=None,
        issue_summary_symbols=None,
        issue_summary_n_jobs=1,
        news_impact_report=None,
        signal_config=SignalConfig(
            return_weight=1.0,
            up_prob_weight=0.0,
            uncertainty_penalty=0.0,
            recommendation_buy_threshold_pct=3.0,
            recommendation_sell_threshold_pct=-1.0,
        ),
    )

    ordered = pred_df.sort_values("Symbol")
    assert ordered["recommendation"].tolist() == ["관망", "매도"]


def test_latest_prediction_degrades_when_issue_summary_fails(monkeypatch):
    from src.models.lgbm_heads import MultiHeadPrediction

    class IdentityCalibrator:
        def transform(self, values):
            return pd.Series(values)

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            return self

        def predict(self, latest):
            return MultiHeadPrediction(
                predicted_return=np.array([0.01]),
                up_probability=np.array([0.5]),
                quantile_low=np.array([-0.02]),
                quantile_mid=np.array([0.0]),
                quantile_high=np.array([0.02]),
            )

    feat = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"]),
            "Symbol": ["A"],
            "Close": [100.0],
            "market_regime": ["normal"],
            "target_log_return": [0.0],
            "target_up": [1],
            "value_traded": [1_000_000.0],
        }
    )
    scored_oof = pd.DataFrame({"Symbol": ["A"], "target_up": [1], "predicted_return": [0.01]})
    monkeypatch.setattr("src.pipeline.MultiHeadStockModel", FakeModel)
    monkeypatch.setattr("src.pipeline.get_symbol_name_map", lambda *_args, **_kwargs: {})

    def fail_issue_summary(*_args, **_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr("src.pipeline.append_issue_summary_columns", fail_issue_summary)
    monkeypatch.setattr(
        "src.pipeline.append_generated_news_impact_context_with_runtime",
        lambda pred_df, *_args, **_kwargs: types.SimpleNamespace(
            frame=pred_df,
            to_metadata=lambda: {
                "requested_mode": "rule",
                "actual_mode": "none",
                "fallback_used": False,
                "fallback_reason": "no_context_rows",
            },
        ),
    )

    pred_df, *_ = _predict_pipeline_latest(
        feat=feat,
        feature_columns=[],
        cfg=AppConfig(),
        scored_oof=scored_oof,
        probability_calibrator=IdentityCalibrator(),
        prediction_context=PredictionFrameContext(
            external_coverage_ratio=1.0,
            investor_coverage_ratio=1.0,
            min_liquidity_threshold=0.0,
        ),
        coverage_gate_status="ok",
        context_raw_df=pd.DataFrame(),
        effective_openai_api_key=None,
        effective_openai_model=None,
        issue_summary_symbols=["A"],
        issue_summary_n_jobs=1,
        news_impact_report=None,
    )

    assert len(pred_df) == 1
    assert "predicted_return" in pred_df.columns
    assert any("issue summary unavailable: llm down" == warning for warning in pred_df.attrs["warnings"])


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


def test_external_feature_failure_degrades_to_price_features(monkeypatch):
    cfg = AppConfig()

    def fail_external(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.pipeline.add_external_market_features_with_coverage", fail_external)

    feat, coverage, feature_columns = _build_pipeline_feature_matrix(make_sample_df(days=90), cfg, use_external=True)

    assert not feat.empty
    assert feature_columns
    assert coverage["requested"] == len(cfg.external.market_symbols)
    assert coverage["successful"] == 0
    assert coverage["failed"] == len(cfg.external.market_symbols)
    assert coverage["status"] == "error"
    assert "network down" in coverage["error"]


def test_investor_context_failure_degrades_to_input_data(monkeypatch):
    data = make_sample_df(days=10)

    def fail_context(*_args, **_kwargs):
        raise RuntimeError("context down")

    monkeypatch.setattr("src.pipeline.add_investor_context_with_coverage", fail_context)

    out, coverage, raw_events = _prepare_pipeline_context(
        data=data,
        use_investor_context=True,
        enable_investor_disclosure=True,
        dart_api_key=None,
        dart_corp_map_csv=None,
        naver_client_id=None,
        naver_client_secret=None,
        context_raw_event_n_jobs=1,
    )

    pd.testing.assert_frame_equal(out, data)
    assert raw_events.empty
    assert coverage["enabled"] is True
    assert coverage["status"] == "error"
    assert "context down" in coverage["error"]


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


def test_fetch_real_ohlcv_raises_when_yfinance_returns_empty(monkeypatch):
    import src.data.fetch_real_data as fr

    def _empty_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(fr, "_safe_download_ohlcv", _empty_download)
    with pytest.raises(RuntimeError):
        fr.fetch_real_ohlcv(["005930.KS"], start="2024-01-01")
