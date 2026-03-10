from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=r"Glyph .* missing from font", category=UserWarning)


def save_backtest_figures(backtest_series: pd.DataFrame, out_dir: str) -> dict:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    out = {}
    if backtest_series.empty:
        return out

    bt = backtest_series.copy()
    bt["Date"] = pd.to_datetime(bt["Date"])

    fig1 = p / "equity_curve.png"
    plt.figure(figsize=(10, 4))
    plt.plot(bt["Date"], bt["equity"])
    plt.title("Backtest Equity Curve")
    plt.tight_layout()
    plt.savefig(fig1)
    plt.close()

    fig2 = p / "drawdown_curve.png"
    plt.figure(figsize=(10, 4))
    plt.plot(bt["Date"], bt["drawdown"])
    plt.title("Backtest Drawdown")
    plt.tight_layout()
    plt.savefig(fig2)
    plt.close()

    out["equity_curve"] = str(fig1)
    out["drawdown_curve"] = str(fig2)
    return out


def save_signal_histogram(pred_df: pd.DataFrame, out_dir: str) -> str | None:
    if pred_df.empty or "signal_score" not in pred_df.columns:
        return None
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig = p / "signal_score_hist.png"

    plt.figure(figsize=(8, 4))
    pred_df["signal_score"].hist(bins=30)
    plt.title("Signal Score Distribution")
    plt.tight_layout()
    plt.savefig(fig)
    plt.close()
    return str(fig)




def _pick_korean_font() -> str | None:
    preferred = [
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return None


def _configure_korean_font() -> bool:
    chosen = _pick_korean_font()
    if chosen:
        mpl.rcParams["font.family"] = [chosen, "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
        return True

    mpl.rcParams["font.family"] = ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return False


def save_actual_vs_predicted_plot(oof_df: pd.DataFrame, out_dir: str) -> str | None:
    required = {"Date", "predicted_log_return", "target_log_return"}
    if oof_df.empty or not required.issubset(set(oof_df.columns)):
        return None

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig_path = p / "actual_vs_predicted_return.png"

    use_korean = _configure_korean_font()
    by_day = oof_df.copy()
    by_day["Date"] = pd.to_datetime(by_day["Date"])
    daily = (
        by_day.groupby("Date", as_index=False)[["predicted_log_return", "target_log_return"]]
        .mean()
        .sort_values("Date")
    )

    plt.figure(figsize=(11, 4.5))
    actual_label = "실제 수익률(%)" if use_korean else "Actual return(%)"
    pred_label = "예측 수익률(%)" if use_korean else "Predicted return(%)"
    title = "실제 vs 예측 수익률(일자 평균)" if use_korean else "Actual vs Predicted Return (daily mean)"
    xlabel = "날짜" if use_korean else "Date"
    ylabel = "수익률(%)" if use_korean else "Return(%)"

    plt.plot(daily["Date"], np.expm1(daily["target_log_return"]) * 100.0, label=actual_label, linewidth=1.7)
    plt.plot(daily["Date"], np.expm1(daily["predicted_log_return"]) * 100.0, label=pred_label, linewidth=1.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return str(fig_path)




def save_actual_vs_predicted_price_plot(oof_df: pd.DataFrame, out_dir: str) -> str | None:
    required = {"Date", "Close", "predicted_log_return", "target_log_return"}
    if oof_df.empty or not required.issubset(set(oof_df.columns)):
        return None

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig_path = p / "actual_vs_predicted_price.png"

    use_korean = _configure_korean_font()
    work = oof_df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["actual_next_close"] = work["Close"] * np.exp(work["target_log_return"])
    work["predicted_next_close"] = work["Close"] * np.exp(work["predicted_log_return"])
    daily = (
        work.groupby("Date", as_index=False)[["actual_next_close", "predicted_next_close"]]
        .mean()
        .sort_values("Date")
    )

    plt.figure(figsize=(11, 4.5))
    plt.plot(
        daily["Date"],
        daily["actual_next_close"],
        label="실제 다음 종가(평균)" if use_korean else "Actual next close (mean)",
        linewidth=1.7,
    )
    plt.plot(
        daily["Date"],
        daily["predicted_next_close"],
        label="예측 다음 종가(평균)" if use_korean else "Predicted next close (mean)",
        linewidth=1.7,
    )
    plt.title("실제 vs 예측 가격 비교" if use_korean else "Actual vs Predicted Price")
    plt.xlabel("날짜" if use_korean else "Date")
    plt.ylabel("가격" if use_korean else "Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return str(fig_path)






def _annotate_all_points(x, y, fmt: str = "{:.3f}"):
    for xi, yi in zip(x, y):
        if pd.isna(yi):
            continue
        plt.annotate(fmt.format(float(yi)), (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

def _safe_symbol_filename(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(symbol))


def save_symbol_level_comparison_figures(oof_df: pd.DataFrame, out_dir: str, max_symbols: int | None = None) -> dict:
    required = {"Date", "Symbol", "Close", "predicted_log_return", "target_log_return"}
    if oof_df.empty or not required.issubset(set(oof_df.columns)):
        return {}

    p = Path(out_dir)
    symbol_dir = p / "symbol_level"
    recent_dir = symbol_dir / "recent_month"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    recent_dir.mkdir(parents=True, exist_ok=True)
    use_korean = _configure_korean_font()

    work = oof_df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["actual_next_close"] = work["Close"] * np.exp(work["target_log_return"])
    work["predicted_next_close"] = work["Close"] * np.exp(work["predicted_log_return"])
    work["actual_return_pct"] = np.expm1(work["target_log_return"]) * 100.0
    work["predicted_return_pct"] = np.expm1(work["predicted_log_return"]) * 100.0

    symbols = sorted(work["Symbol"].dropna().astype(str).unique().tolist())
    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    generated = 0
    recent_generated = 0
    for symbol in symbols:
        sdf = work[work["Symbol"].astype(str) == symbol].sort_values("Date")
        if sdf.empty:
            continue

        safe = _safe_symbol_filename(symbol)

        price_fig = symbol_dir / f"{safe}_actual_vs_predicted_price.png"
        plt.figure(figsize=(10, 4))
        plt.plot(sdf["Date"], sdf["actual_next_close"], label="실제 다음 종가" if use_korean else "Actual next close")
        plt.plot(sdf["Date"], sdf["predicted_next_close"], label="예측 다음 종가" if use_korean else "Predicted next close")
        plt.title(f"{symbol} - 실제/예측 가격" if use_korean else f"{symbol} - Actual vs Predicted Price")
        plt.xlabel("날짜" if use_korean else "Date")
        plt.ylabel("가격" if use_korean else "Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(price_fig)
        plt.close()

        ret_fig = symbol_dir / f"{safe}_actual_vs_predicted_return.png"
        plt.figure(figsize=(10, 4))
        plt.plot(sdf["Date"], sdf["actual_return_pct"], label="실제 수익률(%)" if use_korean else "Actual return(%)")
        plt.plot(sdf["Date"], sdf["predicted_return_pct"], label="예측 수익률(%)" if use_korean else "Predicted return(%)")
        plt.title(f"{symbol} - 실제/예측 수익률" if use_korean else f"{symbol} - Actual vs Predicted Return")
        plt.xlabel("날짜" if use_korean else "Date")
        plt.ylabel("수익률(%)" if use_korean else "Return(%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ret_fig)
        plt.close()

        # recent month charts (day-of-month x-axis + annotate all points)
        max_date = sdf["Date"].max()
        recent = sdf[sdf["Date"] >= (max_date - pd.Timedelta(days=31))].copy()
        if not recent.empty:
            recent["day"] = recent["Date"].dt.day.astype(int)

            r_price = recent_dir / f"{safe}_recent_month_price.png"
            plt.figure(figsize=(10, 4))
            plt.plot(recent["day"], recent["actual_next_close"], marker="o", label="실제 다음 종가" if use_korean else "Actual next close")
            plt.plot(recent["day"], recent["predicted_next_close"], marker="o", label="예측 다음 종가" if use_korean else "Predicted next close")
            _annotate_all_points(recent["day"], recent["actual_next_close"], "{:.3f}")
            _annotate_all_points(recent["day"], recent["predicted_next_close"], "{:.3f}")
            plt.xticks(recent["day"].tolist(), [str(int(x)) for x in recent["day"].tolist()])
            plt.title(f"{symbol} - 최근1개월 실제/예측 가격" if use_korean else f"{symbol} - Recent Month Price")
            plt.xlabel("일" if use_korean else "Day")
            plt.ylabel("가격" if use_korean else "Price")
            plt.legend()
            plt.tight_layout()
            plt.savefig(r_price)
            plt.close()

            r_ret = recent_dir / f"{safe}_recent_month_return.png"
            plt.figure(figsize=(10, 4))
            plt.plot(recent["day"], recent["actual_return_pct"], marker="o", label="실제 수익률(%)" if use_korean else "Actual return(%)")
            plt.plot(recent["day"], recent["predicted_return_pct"], marker="o", label="예측 수익률(%)" if use_korean else "Predicted return(%)")
            _annotate_all_points(recent["day"], recent["actual_return_pct"], "{:.3f}")
            _annotate_all_points(recent["day"], recent["predicted_return_pct"], "{:.3f}")
            plt.xticks(recent["day"].tolist(), [str(int(x)) for x in recent["day"].tolist()])
            plt.title(f"{symbol} - 최근1개월 실제/예측 수익률" if use_korean else f"{symbol} - Recent Month Return")
            plt.xlabel("일" if use_korean else "Day")
            plt.ylabel("수익률(%)" if use_korean else "Return(%)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(r_ret)
            plt.close()
            recent_generated += 2

        generated += 2

    return {
        "symbol_level_figure_dir": str(symbol_dir),
        "symbol_level_recent_month_dir": str(recent_dir),
        "symbol_level_figure_count": generated,
        "symbol_level_recent_month_figure_count": recent_generated,
        "symbol_level_symbol_count": len(symbols),
    }

def save_diagnostic_figures(oof_df: pd.DataFrame, out_dir: str) -> dict:
    if oof_df.empty:
        return {}
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    use_korean = _configure_korean_font()
    out = {}

    if {"up_probability", "target_log_return"}.issubset(set(oof_df.columns)):
        cal_path = p / "up_probability_calibration.png"
        cal = oof_df[["up_probability", "target_log_return"]].copy().dropna()
        if not cal.empty:
            cal["actual_up"] = (cal["target_log_return"] > 0).astype(int)
            n_bins = min(10, max(3, int(np.sqrt(len(cal)))))
            cal["prob_bin"] = pd.qcut(cal["up_probability"], q=n_bins, duplicates="drop")
            grp = cal.groupby("prob_bin", observed=False).agg(pred_prob=("up_probability", "mean"), actual_up_rate=("actual_up", "mean")).dropna()
            grp = grp.sort_values("pred_prob")
            if not grp.empty:
                plt.figure(figsize=(5.5, 5.5))
                plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
                plt.plot(grp["pred_prob"], grp["actual_up_rate"], marker="o")
                plt.title("상승확률 캘리브레이션" if use_korean else "Up-probability calibration")
                plt.xlabel("예측 상승확률" if use_korean else "Predicted up probability")
                plt.ylabel("실제 상승 비율" if use_korean else "Actual up rate")
                plt.tight_layout()
                plt.savefig(cal_path)
                plt.close()
                out["up_probability_calibration"] = str(cal_path)

    if {"uncertainty_width", "predicted_log_return", "target_log_return"}.issubset(set(oof_df.columns)):
        unc_path = p / "uncertainty_vs_error.png"
        unc = oof_df[["uncertainty_width", "predicted_log_return", "target_log_return"]].copy().dropna()
        if not unc.empty:
            unc["abs_error"] = (unc["predicted_log_return"] - unc["target_log_return"]).abs()
            plt.figure(figsize=(7, 4.5))
            plt.scatter(unc["uncertainty_width"], unc["abs_error"], alpha=0.35, s=10)
            plt.title("불확실성 폭 vs 예측오차" if use_korean else "Uncertainty width vs absolute error")
            plt.xlabel("불확실성 폭" if use_korean else "Uncertainty width")
            plt.ylabel("절대 오차" if use_korean else "Absolute error")
            plt.tight_layout()
            plt.savefig(unc_path)
            plt.close()
            out["uncertainty_vs_error"] = str(unc_path)

    return out
def build_symbol_summary_table(pred_df: pd.DataFrame, oof_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    accuracy = pd.DataFrame(columns=["Symbol", "direction_accuracy"])
    if not oof_df.empty and {"Symbol", "target_log_return", "predicted_log_return"}.issubset(set(oof_df.columns)):
        tmp = oof_df[["Symbol", "target_log_return", "predicted_log_return"]].copy()
        tmp["actual_up"] = (tmp["target_log_return"] > 0).astype(int)
        tmp["pred_up"] = (tmp["predicted_log_return"] > 0).astype(int)
        tmp["direction_hit"] = (tmp["actual_up"] == tmp["pred_up"]).astype(float)
        accuracy = tmp.groupby("Symbol", as_index=False)["direction_hit"].mean().rename(columns={"direction_hit": "direction_accuracy"})

    table = pred_df.copy()
    table["종목코드"] = table["Symbol"].astype(str).str.replace(r"\..*$", "", regex=True)
    table["종목명"] = table.get("symbol_name", table["Symbol"]).astype(str)
    table["상승확률"] = table["up_probability"]
    table["하락확률"] = 1.0 - table["up_probability"]
    table["상승/하락(±)"] = table["상승확률"].map(lambda x: f"+{x:.3f}") + " / " + table["하락확률"].map(lambda x: f"-{x:.3f}")
    table["시그널라벨"] = table["signal_label"].astype(str).replace({"nan": "신뢰도 보통"}).fillna("신뢰도 보통")

    table = table.merge(accuracy, on="Symbol", how="left")
    table["direction_accuracy"] = table["direction_accuracy"].fillna(0.5)

    summary = pd.DataFrame(
        {
            "Symbol": table["Symbol"],
            "종목코드": table["종목코드"],
            "종목명": table["종목명"],
            "예상수익률(%)": table["predicted_return"],
            "시그널 라벨": table["시그널라벨"],
            "내일 예상 종가": table["predicted_close"],
            "오늘 종가": table["Close"],
            "상승/하락 확률(±)": table["상승/하락(±)"],
            "상승/하락 방향 정확도": table["direction_accuracy"],
            "불확실성 범위": table["uncertainty_band"],
            "시그널 점수": table["signal_score"],
        }
    )
    num_cols = summary.select_dtypes(include=["number"]).columns
    summary.loc[:, num_cols] = summary.loc[:, num_cols].round(3)
    return summary.sort_values("시그널 점수", ascending=False)


def save_symbol_summary_artifacts(pred_df: pd.DataFrame, oof_df: pd.DataFrame, out_dir: str) -> dict:
    summary = build_symbol_summary_table(pred_df, oof_df)
    if summary.empty:
        return {}

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    use_korean = _configure_korean_font()

    top = summary.head(20).copy()
    png_path = p / "symbol_summary_table_top20.png"
    fig, ax = plt.subplots(figsize=(18, 0.55 * (len(top) + 2)))
    ax.axis("off")
    display_df = top.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
    if not use_korean:
        display_df["시그널 라벨"] = display_df["시그널 라벨"].replace({
            "신뢰도 높음": "High Confidence",
            "신뢰도 보통": "Medium Confidence",
            "신뢰도 낮음": "Low Confidence",
        })
        display_df = display_df.rename(columns={
            "Symbol": "Symbol",
            "종목코드": "Code",
            "종목명": "Name",
            "예상수익률(%)": "ExpectedReturn(%)",
            "시그널 라벨": "ConfidenceLabel",
            "내일 예상 종가": "PredClose(T+1)",
            "오늘 종가": "Close(T)",
            "상승/하락 확률(±)": "UpDownProb(±)",
            "상승/하락 방향 정확도": "DirectionAcc",
            "불확실성 범위": "UncertaintyBand",
            "시그널 점수": "SignalScore",
        })
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    plt.title("종목별 예측 요약표 (상위 20개)" if use_korean else "Per-symbol prediction summary (Top 20)")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    return {"symbol_summary_png": str(png_path)}
