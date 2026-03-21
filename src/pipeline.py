from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config.settings import AppConfig
from src.data.cleaners import clean_ohlcv
from src.data.fetch_real_data import append_real_ohlcv_csv, normalize_user_symbols, save_real_ohlcv_csv
from src.data.krx_universe import get_symbol_name_map
from src.data.loaders import load_ohlcv_csv
from src.data.investor_context import InvestorContextConfig, add_investor_context_with_coverage
from src.data.universe import filter_by_universe, load_universe_symbols
from src.features.external_features import add_external_market_features_with_coverage
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.inference.predict import build_prediction_frame
from src.models.lgbm_heads import MultiHeadPrediction, MultiHeadStockModel
from src.reports.visualize import (
    save_actual_vs_predicted_plot,
    save_actual_vs_predicted_price_plot,
    save_diagnostic_figures,
    save_symbol_level_comparison_figures,
    save_backtest_figures,
    save_signal_histogram,
    save_symbol_summary_artifacts,
)
from src.validation.backtest import run_long_only_topk_backtest
from src.validation.baselines import evaluate_baselines
from src.validation.signal_tuning import tune_signal_weights
from src.validation.walk_forward import walk_forward_oof_predictions, walk_forward_validate
from src.validation.metrics import probability_calibration_metrics



DEFAULT_REAL_SYMBOLS: list[str] = [
    "207940",
    "034020",
    "035420",
    "272210",
    "042660",
    "000660",
    "042700",
    "006400",
    "015760",
    "051910",
    "005380",
    "005930",
]

HIGH_CONVICTION_NET_BUY = 100_000_000_000
TOP_TURNOVER_UP_PROBABILITY = 0.65
STRONG_DUAL_BUY_UP_PROBABILITY = 0.7
HIGH_CONVICTION_COMBINED_UP_PROBABILITY = 0.78
NASDAQ_FUTURES_TAILWIND_UP_PROBABILITY = 0.55


def _fallback_symbols_from_input_or_default(input_csv: str) -> list[str]:
    """Return the built-in demo universe used when no explicit fetch universe is provided."""
    return normalize_user_symbols(DEFAULT_REAL_SYMBOLS)


def _project_result_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    out = root / "result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_output_path(output_csv: str, is_windows: bool | None = None) -> Path:
    """Force all file outputs under project-local ./result directory."""
    _ = (os.name == "nt") if is_windows is None else is_windows
    requested = Path(output_csv)
    result_dir = _project_result_dir()

    # keep explicit paths already inside result dir; otherwise redirect by filename
    try:
        if requested.is_absolute() and requested.resolve().is_relative_to(result_dir.resolve()):
            output_path = requested
        else:
            output_path = result_dir / requested.name
    except Exception:
        output_path = result_dir / requested.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_output_dir(output_dir: str) -> Path:
    requested = Path(output_dir)
    result_dir = _project_result_dir()
    try:
        if requested.is_absolute() and requested.resolve().is_relative_to(result_dir.resolve()):
            out_dir = requested
        else:
            out_dir = result_dir / requested.name
    except Exception:
        out_dir = result_dir / requested.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _feature_columns(df: pd.DataFrame) -> list[str]:
    base = {
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
        "value_traded",
        "turnover_rank_daily",
        "is_top_turnover_10",
        "market_type_kospi",
        "market_type_kosdaq",
        "market_type_konex",
        "venue_krx",
        "venue_nxt",
        "session_regular",
        "session_premarket",
        "session_aftermarket",
        "session_offhours",
        "days_since_listing",
        "is_newly_listed",
        "is_newly_listed_60d",
        "individual_net_buy",
        "foreign_net_buy",
        "institution_net_buy",
        "foreign_ownership_ratio",
        "program_trading_flow",
        "foreign_buy_signal",
        "institution_buy_signal",
        "smart_money_buy_signal",
        "close_to_52w_high",
        "near_52w_high_flag",
        "breakout_52w_flag",
        "investor_event_score",
        "limit_hit_up_flag",
        "limit_hit_down_flag",
        "limit_event_flag",
        "pbr",
        "per",
        "roe",
        "dividend_yield",
        "buyback_flag",
        "share_cancellation_flag",
        "shareholder_return_score",
        "short_sell_event_score",
    }
    return [
        c
        for c in df.columns
        if c.startswith(("ret_", "ma_", "close_to_ma_", "vol_", "ks", "kq", "gspc", "ixic", "nq_f", "sox", "vix", "krw", "tnx"))
        or c in base
    ]




def _adaptive_training_cfg(cfg, feat: pd.DataFrame):
    tuned = cfg.training
    uniq = len(feat["Date"].unique())
    tuned.min_train_size = min(tuned.min_train_size, max(60, int(uniq * 0.6)))
    tuned.test_size = min(tuned.test_size, max(20, int(uniq * 0.2)))
    tuned.step_size = min(tuned.step_size, max(20, tuned.test_size // 2))
    return tuned


def _display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
    return width


def _pad_display(text: str, width: int, align: str = "left") -> str:
    s = str(text)
    pad = max(0, width - _display_width(s))
    if align == "right":
        return " " * pad + s
    return s + " " * pad


def _recommendation_from_signal(signal_score: float | int | None, predicted_return: float | int | None) -> str:
    if pd.isna(predicted_return):
        return "관망"

    ret = float(predicted_return)

    if ret > 1.0:
        return "매수"
    if ret <= -1.0:
        return "매도"
    return "관망"


def _confidence_label(confidence_score: float | int | None) -> str:
    if pd.isna(confidence_score):
        return "신뢰도 보통"
    c = float(confidence_score)
    if c >= 0.67:
        return "신뢰도 높음"
    if c >= 0.34:
        return "신뢰도 보통"
    return "신뢰도 낮음"


def _combined_confidence_score(row: pd.Series) -> float:
    confidence = float(row.get("confidence_score", 0.5) or 0.5)
    history_acc = float(row.get("history_direction_accuracy", 0.5) or 0.5)
    return max(0.0, min(1.0, 0.5 * confidence + 0.5 * history_acc))


def _risk_flag(row: pd.Series) -> str:
    flags = []
    if float(row.get("uncertainty_score", 0) or 0) >= 0.75:
        flags.append("HIGH_UNCERTAINTY")
    if float(row.get("up_probability", 0) or 0) < 0.5:
        flags.append("LOW_UP_PROB")
    if float(row.get("history_direction_accuracy", 0.5) or 0.5) < 0.45:
        flags.append("LOW_HISTORY_ACC")
    return "|".join(flags) if flags else "NORMAL"


def _position_size_hint(confidence_score: float | int | None, risk_flag: str) -> str:
    c = float(confidence_score) if not pd.isna(confidence_score) else 0.5
    if "HIGH_UNCERTAINTY" in risk_flag:
        return "소액"
    if c >= 0.75:
        return "중간"
    if c >= 0.5:
        return "소액"
    return "관망"


def _format_percentage_text(value, digits: int = 1, unit_interval: bool = False) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    percent_value = float(numeric) * 100.0 if unit_interval else float(numeric)
    return f"{percent_value:.{digits}f}%"


def _format_korean_amount(value: float | int | None) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric) / 100_000_000:,.0f}억"


def _has_top_turnover_support(row: pd.Series) -> bool:
    turnover_rank = pd.to_numeric(pd.Series([row.get("turnover_rank_daily")]), errors="coerce").iloc[0]
    return not pd.isna(turnover_rank) and float(turnover_rank) <= 15.0


def _has_strong_dual_buy_support(row: pd.Series) -> bool:
    foreign_net_buy = float(row.get("foreign_net_buy", 0) or 0)
    institution_net_buy = float(row.get("institution_net_buy", 0) or 0)
    return foreign_net_buy >= HIGH_CONVICTION_NET_BUY and institution_net_buy >= HIGH_CONVICTION_NET_BUY


def _is_high_conviction_flow(row: pd.Series) -> bool:
    return _has_top_turnover_support(row) and _has_strong_dual_buy_support(row)


def _has_nasdaq_futures_tailwind(row: pd.Series) -> bool:
    nq_ret = pd.to_numeric(pd.Series([row.get("nq_f_ret_1d")]), errors="coerce").iloc[0]
    return not pd.isna(nq_ret) and float(nq_ret) > 0


def _apply_probability_overrides(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "up_probability" not in pred_df.columns:
        return pred_df

    out = pred_df.copy()
    top_turnover_mask = out.apply(_has_top_turnover_support, axis=1)
    dual_buy_mask = out.apply(_has_strong_dual_buy_support, axis=1)
    combined_mask = top_turnover_mask & dual_buy_mask
    nasdaq_tailwind_mask = out.apply(_has_nasdaq_futures_tailwind, axis=1)

    if top_turnover_mask.any():
        out.loc[top_turnover_mask, "up_probability"] = out.loc[top_turnover_mask, "up_probability"].clip(
            lower=TOP_TURNOVER_UP_PROBABILITY
        )
    if dual_buy_mask.any():
        out.loc[dual_buy_mask, "up_probability"] = out.loc[dual_buy_mask, "up_probability"].clip(
            lower=STRONG_DUAL_BUY_UP_PROBABILITY
        )
    if combined_mask.any():
        out.loc[combined_mask, "up_probability"] = out.loc[combined_mask, "up_probability"].clip(
            lower=HIGH_CONVICTION_COMBINED_UP_PROBABILITY
        )
    if nasdaq_tailwind_mask.any():
        out.loc[nasdaq_tailwind_mask, "up_probability"] = out.loc[nasdaq_tailwind_mask, "up_probability"].clip(
            lower=NASDAQ_FUTURES_TAILWIND_UP_PROBABILITY
        )
    return out


def _prediction_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    up_probability = float(row.get("up_probability", 0.5) or 0.5)
    foreign_net_buy = float(row.get("foreign_net_buy", 0) or 0)
    institution_net_buy = float(row.get("institution_net_buy", 0) or 0)
    breakout_52w_flag = float(row.get("breakout_52w_flag", 0) or 0)
    near_52w_high_flag = float(row.get("near_52w_high_flag", 0) or 0)
    history_acc = float(row.get("history_direction_accuracy", 0.5) or 0.5)
    uncertainty_score = float(row.get("uncertainty_score", 0.5) or 0.5)

    if _is_high_conviction_flow(row):
        reasons.append("수급: 거래대금 상위 15위 안에 들고 외국인·기관이 각각 1,000억 이상 순매수했습니다")
    elif _has_top_turnover_support(row):
        reasons.append("수급: 거래대금 상위 15위로 시장 관심이 높은 종목입니다")
    elif _has_strong_dual_buy_support(row):
        reasons.append("수급: 외국인·기관이 각각 1,000억 이상 순매수했습니다")
    elif foreign_net_buy > 0 and institution_net_buy > 0:
        reasons.append(
            f"수급: 외국인 {_format_korean_amount(foreign_net_buy)}, 기관 {_format_korean_amount(institution_net_buy)} 순매수입니다"
        )

    if _has_nasdaq_futures_tailwind(row):
        reasons.append("해외 흐름: 나스닥100 선물이 올라 한국 증시에 우호적인 환경입니다")

    if up_probability >= 0.7:
        reasons.append(f"확률: 상승 가능성이 {up_probability * 100:.1f}%로 높은 편입니다")
    elif up_probability >= 0.55:
        reasons.append(f"확률: 상승 가능성이 {up_probability * 100:.1f}%로 우세합니다")
    else:
        reasons.append(f"확률: 상승 가능성이 {up_probability * 100:.1f}%로 아직 뚜렷하지 않습니다")

    if breakout_52w_flag > 0:
        reasons.append("추세: 52주 고점을 돌파한 흐름입니다")
    elif near_52w_high_flag > 0:
        reasons.append("추세: 52주 고점 부근에서 버티는 흐름입니다")

    if history_acc >= 0.6:
        reasons.append(f"신뢰도: 과거 방향 적중률이 {history_acc * 100:.1f}%였습니다")
    elif uncertainty_score >= 0.7:
        reasons.append("주의: 변동성이 큰 편이라 보수적으로 보는 것이 좋습니다")
    elif uncertainty_score <= 0.35:
        reasons.append("안정성: 예측 변동성이 비교적 낮은 편입니다")

    if not reasons:
        reasons.append("종합: 수급과 추세가 모두 중립권이라 모델 종합 점수를 중심으로 판단했습니다")
    return " / ".join(reasons[:4])


def _build_result_simple(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["종목코드"] = out["Symbol"].astype(str).str.replace(r"\..*$", "", regex=True)
    out["종목명"] = out["symbol_name"].astype(str)
    out["권고"] = out.apply(
        lambda row: _recommendation_from_signal(row.get("signal_score"), row.get("predicted_return")),
        axis=1,
    )
    out["예측 신뢰도"] = out.apply(_combined_confidence_score, axis=1)
    out["예측 이유"] = out.apply(_prediction_reason, axis=1)
    if "up_probability" in out.columns:
        out["상승확률(%)"] = out["up_probability"].map(lambda v: _format_percentage_text(v, digits=1, unit_interval=True))
    else:
        out["상승확률(%)"] = "-"

    simple = out[
        ["종목코드", "종목명", "권고", "predicted_close", "predicted_return", "상승확률(%)", "예측 신뢰도", "예측 이유"]
    ].rename(
        columns={
            "predicted_close": "내일 예상 종가",
            "predicted_return": "내일 예상 수익률(%)",
        }
    )
    simple["내일 예상 종가"] = pd.to_numeric(simple["내일 예상 종가"], errors="coerce").map(
        lambda v: "-" if pd.isna(v) else f"{float(v):,.0f}원"
    )
    simple["내일 예상 수익률(%)"] = out["predicted_return"].map(lambda v: _format_percentage_text(v, digits=3))
    simple["예측 신뢰도"] = out["예측 신뢰도"].map(lambda v: _format_percentage_text(v, digits=1, unit_interval=True))
    simple["_sort_confidence"] = pd.to_numeric(out["예측 신뢰도"], errors="coerce").values
    simple["_sort_return"] = pd.to_numeric(out["predicted_return"], errors="coerce").values
    simple = simple.sort_values(["_sort_confidence", "_sort_return"], ascending=[False, False]).reset_index(drop=True)
    return simple.drop(columns=["_sort_confidence", "_sort_return"])


def _backtest_summary_fields(backtest: dict) -> dict[str, float]:
    keys = ["days", "cum_return", "avg_daily_return", "sharpe", "max_drawdown", "avg_turnover", "avg_selected_count"]
    out = {}
    for k in keys:
        v = backtest.get(k, 0.0)
        out[f"backtest_{k}"] = float(v) if isinstance(v, (int, float)) else 0.0
    return out


def _print_backtest_console_summary(backtest: dict):
    print("\n=== Backtest Summary ===")
    print(f"Days: {int(backtest.get('days', 0))}")
    print(f"CumReturn: {float(backtest.get('cum_return', 0.0)):.3f}")
    print(f"AvgDailyReturn: {float(backtest.get('avg_daily_return', 0.0)):.4f}")
    print(f"Sharpe: {float(backtest.get('sharpe', 0.0)):.3f}")
    print(f"MaxDrawdown: {float(backtest.get('max_drawdown', 0.0)):.3f}")
    print(f"AvgTurnover: {float(backtest.get('avg_turnover', 0.0)):.3f}")
    print(f"AvgSelectedCount: {float(backtest.get('avg_selected_count', 0.0)):.2f}")


def _print_prediction_console_summary(pred_df: pd.DataFrame):
    if pred_df.empty:
        print("\n=== Prediction ===")
        print("(no rows)")
        return

    out = pred_df.copy()
    if "history_direction_accuracy" not in out.columns:
        out["history_direction_accuracy"] = 0.5
    out = out.sort_values(["history_direction_accuracy", "signal_score"], ascending=[False, False]).head(10).copy()
    out = _build_result_simple(out)
    rows = []
    for _, r in out.iterrows():
        rows.append(
            {
                "종목코드": str(r.get("종목코드", "")),
                "종목명": str(r.get("종목명", "")),
                "권고": str(r.get("권고", "")),
                "내일 예상 종가": str(r.get("내일 예상 종가", "-")),
                "내일 예상 수익률(%)": str(r.get("내일 예상 수익률(%)", "-")),
                "상승확률(%)": str(r.get("상승확률(%)", "-")),
                "예측 신뢰도": str(r.get("예측 신뢰도", "-")),
            }
        )

    headers = ["종목코드", "종목명", "권고", "내일 예상 종가", "내일 예상 수익률(%)", "상승확률(%)", "예측 신뢰도"]
    col_widths = {h: max(_display_width(h), *(_display_width(row[h]) for row in rows)) for h in headers}

    print("\n=== Prediction ===")
    print("  ".join(_pad_display(h, col_widths[h], "left") for h in headers))
    for row in rows:
        print(
            "  ".join(
                [
                    _pad_display(row["종목코드"], col_widths["종목코드"], "left"),
                    _pad_display(row["종목명"], col_widths["종목명"], "left"),
                    _pad_display(row["권고"], col_widths["권고"], "left"),
                    _pad_display(row["내일 예상 종가"], col_widths["내일 예상 종가"], "right"),
                    _pad_display(row["내일 예상 수익률(%)"], col_widths["내일 예상 수익률(%)"], "right"),
                    _pad_display(row["상승확률(%)"], col_widths["상승확률(%)"], "right"),
                    _pad_display(row["예측 신뢰도"], col_widths["예측 신뢰도"], "right"),
                ]
            )
        )

def _split_oof_for_tuning_and_eval(scored_oof: pd.DataFrame, tune_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(scored_oof["Date"]).dropna().unique())
    if len(dates) < 10:
        return scored_oof.copy(), scored_oof.copy()

    split_idx = max(1, min(len(dates) - 1, int(len(dates) * tune_ratio)))
    tune_dates = set(dates[:split_idx])
    eval_dates = set(dates[split_idx:])

    tune_df = scored_oof[scored_oof["Date"].isin(tune_dates)].copy()
    eval_df = scored_oof[scored_oof["Date"].isin(eval_dates)].copy()
    if tune_df.empty or eval_df.empty:
        return scored_oof.copy(), scored_oof.copy()
    return tune_df, eval_df


def _prediction_from_oof_df(oof: pd.DataFrame) -> MultiHeadPrediction:
    return MultiHeadPrediction(
        predicted_return=oof["predicted_return"].values,
        up_probability=oof["up_probability"].values,
        quantile_low=oof["quantile_low"].values,
        quantile_mid=oof["quantile_mid"].values,
        quantile_high=oof["quantile_high"].values,
    )

def _print_progress(step: int, total: int, message: str):
    print(f"[{step}/{total}] {message}")


def _round_floats(obj, digits: int = 3):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    return obj


def _compute_oof_diagnostics(scored_oof: pd.DataFrame) -> dict:
    if scored_oof.empty:
        return {}

    req = {"target_log_return", "rel_strength", "norm_return", "predicted_log_return", "uncertainty_score", "uncertainty_width"}
    if not req.issubset(set(scored_oof.columns)):
        return {}

    df = scored_oof[list(req)].copy().dropna()
    if df.empty:
        return {}

    actual_up = (df["target_log_return"] > 0).astype(int)

    rel_dir_acc = float(((df["rel_strength"] > 0).astype(int) == actual_up).mean())
    norm_dir_acc = float(((df["norm_return"] > 0).astype(int) == actual_up).mean())
    pred_dir_acc = float(((df["predicted_log_return"] > 0).astype(int) == actual_up).mean())

    abs_error = (df["predicted_log_return"] - df["target_log_return"]).abs()

    return {
        "direction_accuracy": {
            "predicted_log_return": pred_dir_acc,
            "rel_strength": rel_dir_acc,
            "norm_return": norm_dir_acc,
        },
        "uncertainty_diagnostics": {
            "corr_uncertainty_vs_abs_error": float(df["uncertainty_width"].corr(abs_error)),
            "corr_uncertainty_score_vs_abs_error": float(df["uncertainty_score"].corr(abs_error)),
            "uncertainty_score_zero_ratio": float((df["uncertainty_score"] == 0).mean()),
            "uncertainty_score_mean": float(df["uncertainty_score"].mean()),
        },
    }




def _ensure_universe_size(symbols: list[str], expected_size: int) -> list[str]:
    """Backward-compatible helper retained for older tests/import paths."""
    uniq = list(dict.fromkeys(str(s) for s in symbols))
    if len(uniq) >= expected_size:
        return uniq[:expected_size]
    pads = [f"NO_DATA_{i:03d}" for i in range(1, expected_size - len(uniq) + 1)]
    return uniq + pads

def _expand_predictions_to_universe(pred_df: pd.DataFrame, universe_symbols: list[str] | None) -> pd.DataFrame:
    if not universe_symbols:
        return pred_df

    universe = set(str(s) for s in universe_symbols)
    return pred_df[pred_df["Symbol"].astype(str).isin(universe)].copy()





def _calibrate_up_probability(oof_df: pd.DataFrame, up_probs: pd.Series | pd.Index | list | tuple | pd.Series) -> pd.Series:
    if oof_df.empty or "up_probability" not in oof_df.columns or "target_log_return" not in oof_df.columns:
        return pd.Series(up_probs, dtype=float)

    cal = oof_df[["up_probability", "target_log_return"]].copy().dropna()
    if cal.empty or cal["up_probability"].nunique() < 3:
        return pd.Series(up_probs, dtype=float)

    y = (cal["target_log_return"] > 0).astype(int)
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal["up_probability"].astype(float).values, y.values)
        return pd.Series(iso.predict(pd.Series(up_probs, dtype=float).values), dtype=float).clip(0.0, 1.0)
    except Exception:
        return pd.Series(up_probs, dtype=float)


def _normalize_text_columns_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    object_columns = out.select_dtypes(include=["object"]).columns
    for column in object_columns:
        out[column] = out[column].map(
            lambda value: unicodedata.normalize("NFC", value) if isinstance(value, str) else value
        )
    return out


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    normalized = _normalize_text_columns_for_csv(df)
    try:
        normalized.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
        normalized.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[경고] 파일이 열려있어 기본 경로에 저장하지 못했습니다. 대체 경로로 저장: {fallback}")
        return fallback


def _build_combined_symbol_results(pred_df: pd.DataFrame, summary_csv: str | None, out_path: Path) -> str | None:
    if pred_df.empty or not summary_csv:
        return None
    try:
        summary = pd.read_csv(summary_csv)
    except Exception:
        return None

    if summary.empty:
        return None

    if "Symbol" not in summary.columns:
        return None

    extra_cols = [c for c in summary.columns if c not in pred_df.columns]
    combined = pred_df.merge(summary[["Symbol", *extra_cols]], on="Symbol", how="left")
    saved = _safe_to_csv(combined, out_path)
    return str(saved)




def _calibrate_up_probability(oof_df: pd.DataFrame, up_probs: pd.Series | pd.Index | list | tuple | pd.Series) -> pd.Series:
    if oof_df.empty or "up_probability" not in oof_df.columns or "target_log_return" not in oof_df.columns:
        return pd.Series(up_probs, dtype=float)

    cal = oof_df[["up_probability", "target_log_return"]].copy().dropna()
    if cal.empty or cal["up_probability"].nunique() < 3:
        return pd.Series(up_probs, dtype=float)

    y = (cal["target_log_return"] > 0).astype(int)
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal["up_probability"].astype(float).values, y.values)
        return pd.Series(iso.predict(pd.Series(up_probs, dtype=float).values), dtype=float).clip(0.0, 1.0)
    except Exception:
        return pd.Series(up_probs, dtype=float)


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[경고] 파일이 열려있어 기본 경로에 저장하지 못했습니다. 대체 경로로 저장: {fallback}")
        return fallback




def _calibrate_up_probability(oof_df: pd.DataFrame, up_probs: pd.Series | pd.Index | list | tuple | pd.Series) -> pd.Series:
    if oof_df.empty or "up_probability" not in oof_df.columns or "target_log_return" not in oof_df.columns:
        return pd.Series(up_probs, dtype=float)

    cal = oof_df[["up_probability", "target_log_return"]].copy().dropna()
    if cal.empty or cal["up_probability"].nunique() < 3:
        return pd.Series(up_probs, dtype=float)

    y = (cal["target_log_return"] > 0).astype(int)
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal["up_probability"].astype(float).values, y.values)
        return pd.Series(iso.predict(pd.Series(up_probs, dtype=float).values), dtype=float).clip(0.0, 1.0)
    except Exception:
        return pd.Series(up_probs, dtype=float)


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[경고] 파일이 열려있어 기본 경로에 저장하지 못했습니다. 대체 경로로 저장: {fallback}")
        return fallback


def run_pipeline(
    input_csv: str,
    output_csv: str,
    universe_csv: str | None = None,
    report_json: str | None = None,
    figure_dir: str = "reports/figures",
    use_external: bool = True,
    use_investor_context: bool = False,
    dart_api_key: str | None = None,
    dart_corp_map_csv: str | None = None,
):
    total_steps = 13
    _print_progress(1, total_steps, "Loading app configuration")
    cfg = AppConfig()

    _print_progress(2, total_steps, f"Loading input data: {input_csv}")

    raw = load_ohlcv_csv(input_csv)
    cleaned = clean_ohlcv(raw)

    _print_progress(3, total_steps, "Applying data cleaning and universe filter")
    requested_universe_symbols = None
    if universe_csv:
        universe = load_universe_symbols(universe_csv)
        requested_universe_symbols = list(universe)
        data = filter_by_universe(cleaned, universe)
    else:
        requested_universe_symbols = sorted(cleaned["Symbol"].astype(str).unique().tolist())
        data = cleaned.copy()

    _print_progress(4, total_steps, "Adding investor context")
    investor_context_coverage = {
        "enabled": False,
        "flow": {"requested": 0, "successful": 0, "failed": 0},
        "disclosure": {"requested": 0, "successful": 0, "failed": 0},
        "news": {"requested": 0, "successful": 0, "failed": 0},
    }
    if use_investor_context:
        data, investor_context_coverage = add_investor_context_with_coverage(
            data,
            InvestorContextConfig(
                enabled=True,
                enable_disclosure=False,
                enable_news=False,
                dart_api_key=dart_api_key,
                dart_corp_map_csv=dart_corp_map_csv,
            ),
        )

    _print_progress(5, total_steps, "Building price features")
    feat = build_features(data, cfg.feature)
    _print_progress(6, total_steps, "Adding external market features")
    external_coverage = {"requested": 0, "successful": 0, "failed": 0, "fallback_used": 0, "details": []}
    if cfg.external.enabled and use_external:
        feat, external_coverage = add_external_market_features_with_coverage(feat, cfg.external.market_symbols)
    feat = annotate_market_regime(feat)
    feat = feat.dropna(subset=["target_log_return"]).copy()
    feature_columns = _feature_columns(feat)

    _print_progress(7, total_steps, "Running walk-forward validation")
    folds = walk_forward_validate(feat, feature_columns, cfg.training)
    effective_cfg = cfg.training
    if not folds:
        effective_cfg = _adaptive_training_cfg(cfg, feat)
        folds = walk_forward_validate(feat, feature_columns, effective_cfg)

    wf_summary = pd.DataFrame([f.metrics for f in folds]).mean().to_dict() if folds else {}

    _print_progress(8, total_steps, "Evaluating baselines")
    baseline_summary = evaluate_baselines(feat)

    _print_progress(9, total_steps, "Generating OOF predictions")
    oof = walk_forward_oof_predictions(feat, feature_columns, effective_cfg)
    if oof.empty:
        raise RuntimeError("OOF predictions are empty. Increase data length or adjust training window.")

    oof_pred = _prediction_from_oof_df(oof)
    oof_pred.up_probability = _calibrate_up_probability(oof, oof_pred.up_probability).values
    scored_oof = build_prediction_frame(oof, oof_pred, cfg.signal)
    scored_oof["target_log_return"] = oof["target_log_return"].values
    if "vol_ratio_20" in oof.columns:
        scored_oof["vol_ratio_20"] = oof["vol_ratio_20"].values

    tune_df, eval_df = _split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7)

    _print_progress(10, total_steps, "Tuning signal weights (train split)")
    tuned = tune_signal_weights(tune_df)
    cfg.signal.return_weight = tuned["return_weight"]
    cfg.signal.up_prob_weight = tuned["up_prob_weight"]
    cfg.signal.rel_strength_weight = tuned["rel_strength_weight"]
    cfg.signal.uncertainty_penalty = tuned["uncertainty_penalty"]

    scored_oof = build_prediction_frame(oof, oof_pred, cfg.signal)
    scored_oof["target_log_return"] = oof["target_log_return"].values
    if "vol_ratio_20" in oof.columns:
        scored_oof["vol_ratio_20"] = oof["vol_ratio_20"].values

    _print_progress(11, total_steps, "Running backtest on holdout split and creating figures")
    backtest_input = eval_df if not eval_df.empty else scored_oof
    backtest = run_long_only_topk_backtest(backtest_input, cfg.backtest)
    backtest_series = pd.DataFrame(backtest.get("series", []))
    figure_dir_path = resolve_output_dir(figure_dir)
    fig_paths = save_backtest_figures(backtest_series, str(figure_dir_path))
    signal_hist = save_signal_histogram(scored_oof, str(figure_dir_path))
    actual_vs_pred = save_actual_vs_predicted_plot(scored_oof, str(figure_dir_path))
    actual_vs_pred_price = save_actual_vs_predicted_price_plot(scored_oof, str(figure_dir_path))
    diagnostic_figs = save_diagnostic_figures(scored_oof, str(figure_dir_path))
    symbol_level_figs = save_symbol_level_comparison_figures(scored_oof, str(figure_dir_path))

    _print_progress(12, total_steps, "Training final model and creating latest predictions")
    train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(random_state=cfg.training.random_state)
    model.fit(train_df, feature_columns, cfg.training.quantiles)

    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
    latest_pred = model.predict(latest)
    latest_pred.up_probability = _calibrate_up_probability(scored_oof, latest_pred.up_probability).values
    pred_df = build_prediction_frame(latest, latest_pred, cfg.signal)
    pred_df = _apply_probability_overrides(pred_df)
    symbol_name_map = get_symbol_name_map(pred_df["Symbol"].dropna().astype(str).tolist())
    pred_df["symbol_name"] = pred_df["Symbol"].astype(str).map(symbol_name_map).fillna(pred_df["Symbol"].astype(str))
    pred_df["confidence_score"] = (1 - pred_df["uncertainty_score"].fillna(1)).clip(lower=0, upper=1)
    pred_df["signal_label"] = pred_df["confidence_score"].map(_confidence_label)

    sym_acc = pd.DataFrame(columns=["Symbol", "history_direction_accuracy"])
    if {"Symbol", "target_log_return", "predicted_log_return"}.issubset(set(scored_oof.columns)):
        tmp_acc = scored_oof[["Symbol", "target_log_return", "predicted_log_return"]].copy()
        tmp_acc["history_direction_accuracy"] = (
            (tmp_acc["target_log_return"] > 0).astype(int) == (tmp_acc["predicted_log_return"] > 0).astype(int)
        ).astype(float)
        sym_acc = tmp_acc.groupby("Symbol", as_index=False)["history_direction_accuracy"].mean()
    pred_df = pred_df.merge(sym_acc, on="Symbol", how="left")
    pred_df["history_direction_accuracy"] = pred_df["history_direction_accuracy"].fillna(0.5)
    pred_df["risk_flag"] = pred_df.apply(_risk_flag, axis=1)
    pred_df["position_size_hint"] = pred_df.apply(
        lambda r: _position_size_hint(r.get("confidence_score"), r.get("risk_flag", "")), axis=1
    )

    bt_summary_cols = _backtest_summary_fields(backtest)
    for k, v in bt_summary_cols.items():
        pred_df[k] = v

    symbol_summary_artifacts = save_symbol_summary_artifacts(pred_df, scored_oof, str(figure_dir_path))
    oof_diagnostics = _compute_oof_diagnostics(scored_oof)

    _print_progress(13, total_steps, "Saving artifacts")
    detail_df = latest.merge(pred_df.drop(columns=["Close"], errors="ignore"), on=["Date", "Symbol"], how="left")
    detail_numeric_cols = detail_df.select_dtypes(include=["number"]).columns
    detail_df.loc[:, detail_numeric_cols] = detail_df.loc[:, detail_numeric_cols].round(3)
    detail_df["내일 예상 종가"] = detail_df["predicted_close"].map(lambda v: "-" if pd.isna(v) else f"{float(v):,.0f}원")
    detail_df["상승확률(%)"] = detail_df["up_probability"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    detail_df["predicted_return_display"] = detail_df["predicted_return"].map(lambda v: _format_percentage_text(v, digits=3))
    detail_df["up_probability_display"] = detail_df["up_probability"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    detail_df["confidence_score_display"] = detail_df["confidence_score"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    detail_df["history_direction_accuracy_display"] = detail_df["history_direction_accuracy"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    simple_df = _build_result_simple(detail_df)

    detail_path = resolve_output_path("result_detail.csv")
    detail_path = _safe_to_csv(detail_df, detail_path)

    simple_path = resolve_output_path("result_simple.csv")
    simple_path = _safe_to_csv(simple_df, simple_path)

    report = {
        "universe_name": cfg.universe.name,
        "universe_size_used": int(data["Symbol"].nunique()),
        "feature_count": len(feature_columns),
        "walk_forward": wf_summary,
        "baselines": baseline_summary,
        "tuned_signal": tuned,
        "tuning_samples": int(len(tune_df)),
        "backtest_samples": int(len(backtest_input)),
        "backtest": {k: v for k, v in backtest.items() if k != "series"},
        "external_feature_coverage": external_coverage,
        "investor_context_coverage": investor_context_coverage,
        "oof_diagnostics": oof_diagnostics,
        "probability_calibration": probability_calibration_metrics(
            (scored_oof["target_log_return"] > 0).astype(int).values,
            scored_oof["up_probability"].astype(float).values,
        ),
        "prediction_coverage": {
            "requested_universe_size": int(len(set(requested_universe_symbols))) if requested_universe_symbols else None,
            "predictions_row_count": int(len(pred_df)),
            "available_prediction_count": int(pred_df["predicted_return"].notna().sum()) if "predicted_return" in pred_df.columns else 0,
            "missing_prediction_count": int(pred_df["predicted_return"].isna().sum()) if "predicted_return" in pred_df.columns else 0,
        },
        "visualization_note": "시각화는 전체 집계 + 종목별 + 종목별 최근1개월 비교 그래프(OOF 기준)를 제공합니다.",
        "artifacts": {
            "result_detail_csv": str(detail_path),
            "result_simple_csv": str(simple_path),
            "figure_dir": str(figure_dir_path),
            **fig_paths,
            "signal_hist": signal_hist,
            "actual_vs_predicted": actual_vs_pred,
            "actual_vs_predicted_price": actual_vs_pred_price,
            **diagnostic_figs,
            **symbol_level_figs,
            **symbol_summary_artifacts,
        },
    }

    if report_json:
        report_path = resolve_output_path(report_json)
        report_path.write_text(json.dumps(_round_floats(report, 3), indent=2, ensure_ascii=False))

    _print_prediction_console_summary(pred_df)


def main():
    parser = argparse.ArgumentParser(description="Stock next-day prediction pipeline")
    parser.add_argument("--input", required=False, default="data/real_ohlcv.csv", help="OHLCV CSV path")
    parser.add_argument(
        "--output",
        default=r"C:\Users\카운\Desktop\result\predictions_direct.csv",
        help="Legacy option (CSV outputs are always saved as result_detail.csv and result_simple.csv under result/)",
    )
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with Symbol column")
    parser.add_argument("--report-json", default=r"C:\Users\카운\Desktop\result\pipeline_report.json", help="Pipeline summary JSON")
    parser.add_argument("--figure-dir", default=r"C:\Users\카운\Desktop\result\figures", help="Directory for generated charts")
    parser.add_argument("--fetch-real", action="store_true", help="Fetch real OHLCV from yfinance before running")
    parser.add_argument("--disable-external", action="store_true", help="Disable external market feature download")
    parser.add_argument("--fetch-investor-context", action="store_true", help="Fetch investor flow context features (foreign/institution flows)")
    parser.add_argument("--dart-api-key", default=None, help="Deprecated legacy option kept for compatibility")
    parser.add_argument("--dart-corp-map-csv", default=None, help="Deprecated legacy option kept for compatibility")
    parser.add_argument(
        "--real-symbols",
        nargs="*",
        default=None,
        help="Symbols used when --fetch-real is enabled (no auto KRX universe)",
    )
    parser.add_argument("--real-start", default="2018-01-01", help="Start date for real data fetch")
    parser.add_argument(
        "--add-symbols",
        nargs="*",
        default=None,
        help="Append user-entered stock codes/symbols into --input CSV (e.g., 005930 000660.KS)",
    )
    args = parser.parse_args()

    input_csv = args.input
    if args.add_symbols:
        symbols_to_add = normalize_user_symbols(args.add_symbols)
        if symbols_to_add:
            append_real_ohlcv_csv(input_csv, symbols=symbols_to_add, start=args.real_start)
            print(f"Added symbols to {input_csv}: {len(symbols_to_add)}")
    if args.fetch_real:
        symbols = args.real_symbols
        if not symbols and args.universe_csv:
            try:
                symbols = load_universe_symbols(args.universe_csv)
                print(f"Loaded symbols from universe CSV: {len(symbols)}")
            except Exception as exc:
                print(f"[경고] universe CSV 로드 실패: {exc}")

        if not symbols:
            symbols = _fallback_symbols_from_input_or_default(input_csv)

        save_real_ohlcv_csv(input_csv, symbols=symbols, start=args.real_start)

    run_pipeline(
        args.input,
        args.output,
        args.universe_csv,
        args.report_json,
        args.figure_dir,
        use_external=not args.disable_external,
        use_investor_context=args.fetch_investor_context,
        dart_api_key=args.dart_api_key,
        dart_corp_map_csv=args.dart_corp_map_csv,
    )


if __name__ == "__main__":
    main()
