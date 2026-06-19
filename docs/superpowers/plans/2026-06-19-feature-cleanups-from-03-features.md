# Feature Cleanups from 03 Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Subagents are forbidden by AGENTS.md.

**Goal:** Implement the P2 cleanup proposals from `docs/03_features.md`: vectorize leader confirmation and make market-regime volatility labeling point-in-time.

**Architecture:** Keep public APIs unchanged. Replace row/date Python loops with pandas groupby transforms in `investment_signals.py`; replace full-frame volatility percentile with expanding per-symbol percentile in `regime_features.py`.

**Tech Stack:** Python 3.10+, pandas, pytest.

---

## File Structure

- Modify: `src/features/investment_signals.py`
  - Responsibility: create investment-support flags, including leader returns and confirmation flags.
- Modify: `src/features/regime_features.py`
  - Responsibility: create display-only market-regime labels.
- Modify: `tests/test_investment_signal_features.py`
  - Responsibility: verify vectorized leader confirmation semantics, input-order preservation, and no Date fallback.
- Create: `tests/test_regime_features.py`
  - Responsibility: verify point-in-time volatility state and no future full-frame quantile dependence.
- Modify: `docs/03_features.md`
  - Responsibility: mark proposals as implemented and document new regime volatility basis.

---

### Task 1: Vectorize leader confirmation

**Files:**
- Modify: `tests/test_investment_signal_features.py`
- Modify: `src/features/investment_signals.py`

- [ ] **Step 1: Write failing tests**

Append these tests to `tests/test_investment_signal_features.py`:

```python
def test_leader_confirmation_populates_group_values_without_reordering_rows():
    cfg = InvestmentCriteriaConfig(leader_top_n=3, leader_min_co_movers=2, leader_min_return=0.02)
    df = pd.DataFrame(
        [
            {"Date": "2026-03-29", "Symbol": "D", "turnover_rank_daily": 4, "daily_return": -0.01},
            {"Date": "2026-03-28", "Symbol": "B", "turnover_rank_daily": 2, "daily_return": 0.03},
            {"Date": "2026-03-28", "Symbol": "A", "turnover_rank_daily": 1, "daily_return": 0.08},
            {"Date": "2026-03-28", "Symbol": "C", "turnover_rank_daily": 3, "daily_return": 0.01},
            {"Date": "2026-03-29", "Symbol": "E", "turnover_rank_daily": 1, "daily_return": 0.01},
            {"Date": "2026-03-29", "Symbol": "F", "turnover_rank_daily": 2, "daily_return": 0.04},
        ]
    )

    out = add_investment_signal_features(df, cfg)

    assert out["Symbol"].tolist() == df["Symbol"].tolist()
    day1 = out[out["Date"] == "2026-03-28"]
    assert day1["leader_1_return"].tolist() == [0.08, 0.08, 0.08]
    assert day1["leader_2_return"].tolist() == [0.03, 0.03, 0.03]
    assert day1["leader_3_return"].tolist() == [0.01, 0.01, 0.01]
    assert day1["leader_confirmation_flag"].tolist() == [1, 1, 1]
    day2 = out[out["Date"] == "2026-03-29"]
    assert day2["leader_confirmation_flag"].tolist() == [0, 0, 0]
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_investment_signal_features.py::test_leader_confirmation_populates_group_values_without_reordering_rows -q`
Expected before implementation: pass or fail depending current semantics; keep as regression coverage.

- [ ] **Step 3: Implement vectorized `_leader_confirmation`**

In `src/features/investment_signals.py`, replace the loop body with stable sort plus per-date transforms:

```python
def _leader_confirmation(df: pd.DataFrame, cfg: InvestmentCriteriaConfig) -> pd.DataFrame:
    out = df.copy()
    leader_1 = pd.Series(0.0, index=out.index, dtype=float)
    leader_2 = pd.Series(0.0, index=out.index, dtype=float)
    leader_3 = pd.Series(0.0, index=out.index, dtype=float)
    leader_confirm = pd.Series(0, index=out.index, dtype=int)
    if "Date" not in out.columns:
        out["leader_1_return"] = leader_1
        out["leader_2_return"] = leader_2
        out["leader_3_return"] = leader_3
        out["leader_confirmation_flag"] = leader_confirm
        return out

    ret = _to_numeric(out, "daily_return", default=0.0)
    rank = _to_numeric(out, "turnover_rank_daily", default=999.0)
    leader_top_n = max(2, int(cfg.leader_top_n))
    min_co_movers = max(1, int(cfg.leader_min_co_movers))
    min_ret = float(cfg.leader_min_return)

    work = pd.DataFrame({"Date": out["Date"], "ret": ret, "rank": rank}, index=out.index)
    work = work.sort_values(["Date", "rank", "ret"], ascending=[True, True, False], kind="mergesort")
    top = work.groupby("Date", sort=False).head(leader_top_n).copy()
    top["leader_pos"] = top.groupby("Date", sort=False).cumcount() + 1

    leader_returns = top.pivot_table(index="Date", columns="leader_pos", values="ret", aggfunc="first")
    leader_returns = leader_returns.rename(columns={1: "leader_1_return", 2: "leader_2_return", 3: "leader_3_return"})
    for col in ["leader_1_return", "leader_2_return", "leader_3_return"]:
        if col not in leader_returns.columns:
            leader_returns[col] = 0.0
    leader_returns = leader_returns[["leader_1_return", "leader_2_return", "leader_3_return"]].fillna(0.0)

    co_movers = top.assign(is_co_mover=(top["ret"] > min_ret).astype(int)).groupby("Date", sort=False)["is_co_mover"].sum()
    leader_returns["leader_confirmation_flag"] = (
        (leader_returns["leader_1_return"] > min_ret) & (co_movers >= min_co_movers)
    ).astype(int)

    mapped = work[["Date"]].join(leader_returns, on="Date").reindex(out.index)
    out["leader_1_return"] = mapped["leader_1_return"].fillna(0.0).astype(float)
    out["leader_2_return"] = mapped["leader_2_return"].fillna(0.0).astype(float)
    out["leader_3_return"] = mapped["leader_3_return"].fillna(0.0).astype(float)
    out["leader_confirmation_flag"] = mapped["leader_confirmation_flag"].fillna(0).astype(int)
    return out
```

- [ ] **Step 4: Run investment signal tests**

Run: `pytest tests/test_investment_signal_features.py -q`
Expected: all tests pass.

---

### Task 2: Make market-regime volatility point-in-time

**Files:**
- Create: `tests/test_regime_features.py`
- Modify: `src/features/regime_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_regime_features.py`:

```python
import pandas as pd

from src.features.regime_features import annotate_market_regime


def test_market_regime_high_vol_uses_past_and_current_values_only_per_symbol():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "Symbol": ["A"] * 5,
            "close_to_ma_20": [0.02] * 5,
            "vol_20": [0.10, 0.11, 0.12, 0.13, 9.00],
        }
    )

    out = annotate_market_regime(df)

    assert out["market_regime"].iloc[3] == "uptrend_high_vol"
    assert out["market_regime"].iloc[4] == "uptrend_high_vol"


def test_market_regime_expanding_vol_threshold_is_symbol_scoped():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "Symbol": ["A", "A", "B", "B"],
            "close_to_ma_20": [0.02, 0.02, -0.02, -0.02],
            "vol_20": [0.10, 0.11, 10.0, 10.1],
        }
    )

    out = annotate_market_regime(df)

    assert out.loc[1, "market_regime"] == "uptrend_high_vol"
    assert out.loc[3, "market_regime"] == "downtrend_high_vol"
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_regime_features.py -q`
Expected before implementation: first test fails because full-frame quantile sees future 9.00.

- [ ] **Step 3: Implement expanding threshold**

Replace `src/features/regime_features.py` with:

```python
from __future__ import annotations

import pandas as pd


def _expanding_volatility_threshold(out: pd.DataFrame, vol: pd.Series) -> pd.Series:
    if "Symbol" not in out.columns or "Date" not in out.columns:
        return vol.expanding(min_periods=1).quantile(0.75).reindex(out.index)

    order = out[["Symbol", "Date"]].copy()
    order["_original_index"] = out.index
    order["_vol_20"] = vol
    order = order.sort_values(["Symbol", "Date", "_original_index"], kind="mergesort")
    thresholds = (
        order.groupby("Symbol", sort=False)["_vol_20"]
        .expanding(min_periods=1)
        .quantile(0.75)
        .reset_index(level=0, drop=True)
    )
    return thresholds.reindex(out.index)


def annotate_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    trend = out["close_to_ma_20"].fillna(0)
    vol = out["vol_20"].fillna(0)

    trend_state = pd.Series("sideways", index=out.index)
    trend_state[trend > 0.01] = "uptrend"
    trend_state[trend < -0.01] = "downtrend"

    vol_threshold = _expanding_volatility_threshold(out, vol)
    vol_state = pd.Series("low_vol", index=out.index)
    vol_state[vol > vol_threshold] = "high_vol"

    out["market_regime"] = trend_state + "_" + vol_state
    return out
```

- [ ] **Step 4: Run regime tests**

Run: `pytest tests/test_regime_features.py -q`
Expected: pass.

---

### Task 3: Update docs and verify smoke

**Files:**
- Modify: `docs/03_features.md`

- [ ] **Step 1: Update docs**

Change `docs/03_features.md` market-regime paragraph to state volatility uses an expanding per-symbol 75th percentile based only on past/current rows when `Symbol` and `Date` exist.

Change the “개선 및 수정 제안” section to “반영 완료” and summarize both implemented changes.

- [ ] **Step 2: Run impacted tests**

Run: `pytest tests/test_investment_signal_features.py tests/test_regime_features.py tests/test_pipeline_smoke.py -q`
Expected: pass.

- [ ] **Step 3: Run sample pipeline**

Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`
Expected: exits 0 and writes report JSON under allowed output handling.

- [ ] **Step 4: Run full tests if time permits**

Run: `pytest -q`
Expected: pass.

- [ ] **Step 5: Commit, push, PR**

Run:

```bash
git add src/features/investment_signals.py src/features/regime_features.py tests/test_investment_signal_features.py tests/test_regime_features.py docs/03_features.md docs/superpowers/plans/2026-06-19-feature-cleanups-from-03-features.md
git commit -m "Vectorize feature cleanup proposals"
git push -u origin HEAD
gh pr create --fill --draft
```

Expected: commit pushed and draft PR created.
