from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

GRADE_STRONG = "\uac15\ub825 \ud6c4\ubcf4"
GRADE_CANDIDATE = "\ud6c4\ubcf4"
GRADE_WATCH = "\uad00\ucc30"
GRADE_EXCLUDED = "\ucd94\ucc9c \uc81c\uc678"


@dataclass(frozen=True)
class CloseBettingRecommendation:
    rank: int
    symbol: str
    name: str
    grade: str
    final_score: int
    first_buy_ratio: float
    reasons: tuple[str, ...]


def add_trade_value_rank(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "trade_value" not in result.columns:
        result["trade_value"] = pd.NA
    result["trade_value"] = pd.to_numeric(result["trade_value"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
    result["trade_value"] = result["trade_value"].fillna(result["close"] * result["volume"])
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    latest = result["date"].max()
    latest_rows = result.loc[result["date"] == latest, ["symbol", "trade_value"]].copy()
    latest_rows["trade_value_rank"] = latest_rows["trade_value"].rank(method="first", ascending=False).astype("Int64")
    rank_by_symbol = latest_rows.set_index("symbol")["trade_value_rank"]
    result["trade_value_rank"] = result["symbol"].map(rank_by_symbol).astype("Int64")
    return result


def add_technical_indicators(
    df: pd.DataFrame,
    ma_periods: tuple[int, ...] = (5, 20, 60),
    high_periods: tuple[int, int] = (20, 252),
    near_high_threshold: float = 0.98,
    min_history_days: int = 252,
    volume_spike_ratio: float = 1.5,
) -> pd.DataFrame:
    result = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = result.groupby("symbol", group_keys=False)

    result["history_days"] = grouped.cumcount() + 1
    result["is_history_sufficient"] = result["history_days"] >= min_history_days

    for period in ma_periods:
        result[f"ma{period}"] = grouped["close"].transform(lambda s, p=period: s.rolling(p, min_periods=1).mean())

    short_period, long_period = high_periods
    result["close_20d_high"] = grouped["close"].transform(lambda s: s.rolling(short_period, min_periods=1).max())
    result["close_52w_high"] = grouped["close"].transform(lambda s: s.rolling(long_period, min_periods=1).max())
    result["is_20d_high"] = result["close"] >= result["close_20d_high"]
    result["is_52w_high"] = result["close"] >= result["close_52w_high"]
    result["is_near_20d_high"] = result["close"] >= result["close_20d_high"] * near_high_threshold
    result["is_near_52w_high"] = result["close"] >= result["close_52w_high"] * near_high_threshold

    avg_volume_20 = grouped["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    result["volume_change_rate"] = result["volume"] / avg_volume_20.replace(0, pd.NA)
    result["volume_change_rate"] = result["volume_change_rate"].fillna(0)
    result["is_volume_spike"] = result["volume_change_rate"] >= volume_spike_ratio

    result["is_bullish"] = result["close"] > result["open"]
    body_rate = (result["open"] - result["close"]) / result["open"]
    result["is_long_bearish"] = (result["close"] < result["open"]) & (body_rate >= 0.03)
    result["is_close_near_high"] = result["close"] >= (result["high"] - (result["high"] - result["low"]) * 0.25)
    return result


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.groupby("symbol")["date"].transform("max")
    return df[df["date"] == dates].copy()


def grade_for_score(score: int | float) -> str:
    if score >= 100:
        return GRADE_STRONG
    if score >= 80:
        return GRADE_CANDIDATE
    if score >= 60:
        return GRADE_WATCH
    return GRADE_EXCLUDED


def _rank_score(rank: object) -> int:
    if pd.isna(rank):
        return 0
    rank_int = int(rank)
    if 1 <= rank_int <= 5:
        return 30
    if 6 <= rank_int <= 10:
        return 20
    if 11 <= rank_int <= 20:
        return 10
    return 0


def _score_row(row: pd.Series, theme_score_default: int, top_trade_value_count: int) -> pd.Series:
    breakdown: dict[str, int] = {}
    reasons: list[str] = []

    rank_points = _rank_score(row.get("trade_value_rank"))
    if rank_points:
        breakdown["trade_value_rank"] = rank_points
        reasons.append(f"\uac70\ub798\ub300\uae08 {int(row['trade_value_rank'])}\uc704")

    if bool(row.get("is_52w_high", False)):
        breakdown["52w_high"] = 100
        reasons.append("52\uc8fc \uc885\uac00 \uae30\uc900 \uc2e0\uace0\uac00")
    elif bool(row.get("is_near_52w_high", False)):
        breakdown["near_52w_high"] = 70
        reasons.append("52\uc8fc \uc885\uac00 \uae30\uc900 \uadfc\uc811 \uc2e0\uace0\uac00")

    if bool(row.get("is_20d_high", False)):
        breakdown["20d_high"] = 40
        reasons.append("20\uc77c \uc885\uac00 \uae30\uc900 \uc2e0\uace0\uac00")
    elif bool(row.get("is_near_20d_high", False)):
        breakdown["near_20d_high"] = 25
        reasons.append("20\uc77c \uc885\uac00 \uae30\uc900 \uadfc\uc811 \uc2e0\uace0\uac00")

    if pd.notna(row.get("ma5")) and row.get("close", 0) > row.get("ma5"):
        breakdown["above_ma5"] = 10
        reasons.append("5\uc77c\uc120 \uc704")
    if pd.notna(row.get("ma20")) and row.get("close", 0) > row.get("ma20"):
        breakdown["above_ma20"] = 10
        reasons.append("20\uc77c\uc120 \uc704")
    if bool(row.get("is_volume_spike", False)):
        breakdown["volume_spike"] = 20
        reasons.append("\uac70\ub798\ub7c9 \uae09\uc99d")
    if bool(row.get("is_long_bearish", False)):
        breakdown["long_bearish"] = -30
        reasons.append("\uc7a5\ub300\uc74c\ubd09 \ud328\ub110\ud2f0")

    technical_score = sum(breakdown.values())
    theme_score = 0 if theme_score_default is None else int(theme_score_default)
    final_score = technical_score + theme_score
    rank = row.get("trade_value_rank")
    is_top = pd.notna(rank) and int(rank) <= top_trade_value_count
    is_forced = bool(row.get("is_history_sufficient", False)) and is_top and (
        bool(row.get("is_52w_high", False)) or bool(row.get("is_near_52w_high", False))
    )
    if is_forced and final_score < 100:
        final_score = 100

    grade = grade_for_score(final_score)
    if not bool(row.get("is_history_sufficient", False)):
        grade = GRADE_EXCLUDED

    return pd.Series(
        {
            "score_breakdown": breakdown,
            "reasons": reasons,
            "technical_score": technical_score,
            "theme_score": theme_score,
            "final_score": final_score,
            "recommendation_grade": grade,
            "is_forced_candidate": is_forced,
        }
    )


def score_candidates(df: pd.DataFrame, theme_score_default: int = 0, top_trade_value_count: int = 20) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        return result
    scores = result.apply(lambda row: _score_row(row, theme_score_default, top_trade_value_count), axis=1)
    return pd.concat([result, scores], axis=1)


def select_close_betting_candidates(
    scored: pd.DataFrame,
    top_n: int | None = 3,
    first_buy_ratio: float = 0.6,
    min_final_score: int | None = None,
) -> pd.DataFrame:
    if scored.empty or "recommendation_grade" not in scored.columns:
        return scored.head(0).copy()
    candidates = scored[scored["recommendation_grade"] != GRADE_EXCLUDED].copy()
    if min_final_score is not None:
        candidates = candidates[pd.to_numeric(candidates["final_score"], errors="coerce") >= min_final_score].copy()
    if {"is_52w_high", "is_near_52w_high"}.issubset(candidates.columns):
        candidates = candidates[candidates["is_52w_high"].astype(bool) | candidates["is_near_52w_high"].astype(bool)].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(["final_score", "trade_value_rank", "volume_change_rate"], ascending=[False, True, False])
    if top_n is not None:
        candidates = candidates.head(top_n)
    candidates = candidates.reset_index(drop=True)
    candidates["recommendation_rank"] = candidates.index + 1
    candidates["first_buy_ratio"] = first_buy_ratio
    candidates["key_reasons"] = candidates["reasons"].apply(lambda reasons: list(reasons)[:3])
    return candidates


def recommendations_from_candidates(candidates: pd.DataFrame) -> list[CloseBettingRecommendation]:
    out: list[CloseBettingRecommendation] = []
    if candidates.empty:
        return out
    for row in candidates.sort_values("recommendation_rank").itertuples(index=False):
        reasons = getattr(row, "key_reasons", None) or getattr(row, "reasons", None) or []
        out.append(
            CloseBettingRecommendation(
                rank=int(getattr(row, "recommendation_rank")),
                symbol=str(getattr(row, "symbol")).split(".")[0].zfill(6),
                name=str(getattr(row, "name")),
                grade=str(getattr(row, "recommendation_grade")),
                final_score=int(getattr(row, "final_score")),
                first_buy_ratio=float(getattr(row, "first_buy_ratio", 0.6)),
                reasons=tuple(str(reason) for reason in reasons),
            )
        )
    return out


def format_recommendation_message(
    recommendations: Iterable[CloseBettingRecommendation],
    as_of: date | None = None,
) -> str:
    recs = list(recommendations)
    report_date = (as_of or date.today()).isoformat()
    lines = ["[\uc2e4\uc2dc\uac04 \ucd94\ucc9c]", f"\uae30\uc900\uc77c: {report_date}", ""]
    if not recs:
        lines.append("\ud604\uc7ac \uc870\uac74\uc744 \ub9cc\uc871\ud558\ub294 \ucd94\ucc9c \ud6c4\ubcf4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
        return "\n".join(lines).strip()

    for item in recs:
        ratio = item.first_buy_ratio * 100
        lines.append(f"{item.rank}\uc704 {item.name}({item.symbol}) - {item.grade}")
        lines.append(f"\uc810\uc218: {item.final_score} / 1\ucc28 \ub9e4\uc218\ube44\uc911: {ratio:.0f}%")
        if item.reasons:
            lines.append("\uadfc\uac70: " + ", ".join(item.reasons[:3]))
        lines.append("")
    return "\n".join(lines).strip()
