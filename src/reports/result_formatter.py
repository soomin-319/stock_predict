from __future__ import annotations

import unicodedata

import pandas as pd


def display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
    return width


def pad_display(text: str, width: int, align: str = "left") -> str:
    s = str(text)
    pad = max(0, width - display_width(s))
    if align == "right":
        return " " * pad + s
    return s + " " * pad


def format_percentage_text(value, digits: int = 1, unit_interval: bool = False) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    percent_value = float(numeric) * 100.0 if unit_interval else float(numeric)
    return f"{percent_value:.{digits}f}%"


def build_result_simple(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["종목코드"] = out["Symbol"].astype(str).str.replace(r"\..*$", "", regex=True)
    out["종목명"] = out["symbol_name"].astype(str)
    out["권고"] = out["recommendation"].astype(str)
    display_confidence = (
        pd.to_numeric(out.get("confidence_score"), errors="coerce").fillna(0.5) * 0.5
        + pd.to_numeric(out.get("history_direction_accuracy"), errors="coerce").fillna(0.5) * 0.5
    )
    out["예측 신뢰도"] = display_confidence.map(lambda v: format_percentage_text(v, digits=1, unit_interval=True))
    out["예측 이유"] = out["prediction_reason"].astype(str)
    up_prob_series = pd.to_numeric(out["up_probability"], errors="coerce") if "up_probability" in out.columns else pd.Series(0.5, index=out.index)
    out["상승확률(%)"] = up_prob_series.map(lambda v: format_percentage_text(v, digits=1, unit_interval=True))

    simple = out[
        [
            "종목코드",
            "종목명",
            "권고",
            "risk_flag",
            "predicted_close",
            "predicted_return",
            *([c for c in ["predicted_return_5d", "predicted_return_20d"] if c in out.columns]),
            "상승확률(%)",
            *([c for c in ["up_probability_5d", "up_probability_20d"] if c in out.columns]),
            "예측 신뢰도",
            "예측 이유",
            *([c for c in ["공시 요약", "뉴스 요약"] if c in out.columns]),
        ]
    ].rename(
        columns={
            "predicted_close": "내일 예상 종가",
            "predicted_return": "내일 예상 수익률(%)",
            "predicted_return_5d": "5일 예상 수익률(%)",
            "predicted_return_20d": "20일 예상 수익률(%)",
            "up_probability_5d": "5일 상승확률(%)",
            "up_probability_20d": "20일 상승확률(%)",
        }
    )
    simple["내일 예상 종가"] = pd.to_numeric(simple["내일 예상 종가"], errors="coerce").map(
        lambda v: "-" if pd.isna(v) else f"{float(v):,.0f}원"
    )
    simple["내일 예상 수익률(%)"] = out["predicted_return"].map(lambda v: format_percentage_text(v, digits=3))
    if "5일 예상 수익률(%)" in simple.columns:
        simple["5일 예상 수익률(%)"] = out["predicted_return_5d"].map(lambda v: format_percentage_text(v, digits=3))
    if "20일 예상 수익률(%)" in simple.columns:
        simple["20일 예상 수익률(%)"] = out["predicted_return_20d"].map(lambda v: format_percentage_text(v, digits=3))
    if "5일 상승확률(%)" in simple.columns:
        simple["5일 상승확률(%)"] = pd.to_numeric(out["up_probability_5d"], errors="coerce").map(
            lambda v: format_percentage_text(v, digits=1, unit_interval=True)
        )
    if "20일 상승확률(%)" in simple.columns:
        simple["20일 상승확률(%)"] = pd.to_numeric(out["up_probability_20d"], errors="coerce").map(
            lambda v: format_percentage_text(v, digits=1, unit_interval=True)
        )
    simple["_sort_confidence"] = pd.to_numeric(out["confidence_score"], errors="coerce").values
    simple["_sort_return"] = pd.to_numeric(out["predicted_return"], errors="coerce").values
    simple = simple.sort_values(["_sort_confidence", "_sort_return"], ascending=[False, False]).reset_index(drop=True)
    return simple.drop(columns=["_sort_confidence", "_sort_return"])


def print_prediction_console_summary(pred_df: pd.DataFrame):
    if pred_df.empty:
        print("\n=== Prediction ===")
        print("(no rows)")
        return

    out = pred_df.copy()
    if "history_direction_accuracy" not in out.columns:
        out["history_direction_accuracy"] = 0.5
    out = out.sort_values(["history_direction_accuracy", "signal_score"], ascending=[False, False]).head(10).copy()
    out = build_result_simple(out)
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
    col_widths = {h: max(display_width(h), *(_display_width(row[h]) for row in rows)) for h in headers}

    print("\n=== Prediction ===")
    print("  ".join(pad_display(h, col_widths[h], "left") for h in headers))
    for row in rows:
        print(
            "  ".join(
                [
                    pad_display(row["종목코드"], col_widths["종목코드"], "left"),
                    pad_display(row["종목명"], col_widths["종목명"], "left"),
                    pad_display(row["권고"], col_widths["권고"], "left"),
                    pad_display(row["내일 예상 종가"], col_widths["내일 예상 종가"], "right"),
                    pad_display(row["내일 예상 수익률(%)"], col_widths["내일 예상 수익률(%)"], "right"),
                    pad_display(row["상승확률(%)"], col_widths["상승확률(%)"], "right"),
                    pad_display(row["예측 신뢰도"], col_widths["예측 신뢰도"], "right"),
                ]
            )
        )


def _display_width(text: str) -> int:
    return display_width(text)


__all__ = [
    "build_result_simple",
    "display_width",
    "format_percentage_text",
    "pad_display",
    "print_prediction_console_summary",
]
