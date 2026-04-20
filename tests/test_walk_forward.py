from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import TrainingConfig
from src.validation.walk_forward import _iter_folds


@pytest.fixture
def trading_dates() -> list[pd.Timestamp]:
    # 600 consecutive business days so min_train_size (756-> overridden) fits
    return list(pd.bdate_range("2020-01-01", periods=600))


def _make_df(dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({"Date": dates, "Symbol": "TEST", "value": range(len(dates))})


def _small_cfg(**overrides) -> TrainingConfig:
    cfg = TrainingConfig(
        min_train_size=252,
        test_size=60,
        step_size=60,
        purge_gap_days=0,
        embargo_days=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_iter_folds_yields_non_overlapping_validation_windows(trading_dates):
    df = _make_df(trading_dates)
    folds = list(_iter_folds(df, _small_cfg()))
    assert len(folds) > 0

    prev_valid_end = None
    for _train_end, valid_start, valid_end, _train_df, _valid_df in folds:
        assert valid_start <= valid_end
        if prev_valid_end is not None:
            assert valid_start > prev_valid_end
        prev_valid_end = valid_end


def test_iter_folds_with_purge_gap_drops_training_tail(trading_dates):
    df = _make_df(trading_dates)
    cfg = _small_cfg(purge_gap_days=20)
    for train_end, valid_start, _valid_end, train_df, valid_df in _iter_folds(df, cfg):
        # All training dates must precede the valid window by at least purge_gap_days trading days.
        assert train_df["Date"].max() == train_end
        gap_days = (valid_start - train_end).days
        assert gap_days >= 20, f"purge gap not enforced: {gap_days} days between {train_end} and {valid_start}"
        # And training rows must not contaminate validation rows.
        assert train_df["Date"].max() < valid_df["Date"].min()


def test_iter_folds_embargo_shifts_validation_forward(trading_dates):
    df = _make_df(trading_dates)
    base_folds = list(_iter_folds(df, _small_cfg(embargo_days=0)))
    embargo_folds = list(_iter_folds(df, _small_cfg(embargo_days=5)))

    # Embargo should shift each valid_start by (approximately) the embargo length.
    for (_, base_start, _, _, _), (_, emb_start, _, _, _) in zip(base_folds, embargo_folds):
        assert emb_start > base_start


def test_iter_folds_skips_when_purge_gap_larger_than_train_size(trading_dates):
    df = _make_df(trading_dates)
    cfg = _small_cfg(min_train_size=50, purge_gap_days=100)
    # purge requires train_cutoff_idx >= 0 -> start >= 100 + 1. min_train_size=50 means first
    # attempted start=50 is rejected; iterator must tolerate this and continue to valid starts.
    folds = list(_iter_folds(df, cfg))
    for train_end, valid_start, *_ in folds:
        assert (valid_start - train_end).days >= 100


def test_iter_folds_no_training_date_leaks_into_validation(trading_dates):
    df = _make_df(trading_dates)
    for train_end, valid_start, valid_end, train_df, valid_df in _iter_folds(df, _small_cfg(purge_gap_days=5)):
        train_dates = set(train_df["Date"])
        valid_dates = set(valid_df["Date"])
        assert train_dates.isdisjoint(valid_dates)
        assert all(d <= train_end for d in train_dates)
        assert all(valid_start <= d <= valid_end for d in valid_dates)
