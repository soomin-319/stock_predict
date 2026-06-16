from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import TrainingConfig
import src.validation.walk_forward as wf
from src.validation.walk_forward import FoldResult, _execute_folds, _iter_folds, _run_fold, aggregate_oof_predictions


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


def test_iter_folds_can_limit_training_to_recent_lookback_window(trading_dates):
    df = _make_df(trading_dates)
    cfg = _small_cfg(walk_forward_lookback_days=60)

    train_end, _valid_start, _valid_end, train_df, _valid_df = next(_iter_folds(df, cfg))

    assert train_df["Date"].max() == train_end
    assert train_df["Date"].nunique() == 60
    assert train_df["Date"].min() == sorted(df[df["Date"] <= train_end]["Date"].unique())[-60]


def test_execute_folds_caps_nested_model_parallelism(monkeypatch):
    seen_cfg = []
    seen_workers = []

    class _FakeExecutor:
        def __init__(self, max_workers):
            seen_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, folds):
            return [fn(fold) for fold in folds]

    def _fake_run_fold(fold, feature_columns, cfg):
        seen_cfg.append((cfg.model_n_jobs, cfg.model_head_n_jobs))
        result = FoldResult(
            train_end=pd.Timestamp("2020-01-01"),
            valid_start=pd.Timestamp("2020-01-02"),
            valid_end=pd.Timestamp("2020-01-03"),
            metrics={"rmse": 0.0},
        )
        return result, pd.DataFrame({"predicted_return": [0.1], "up_probability": [0.6]})

    folds = [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"), pd.DataFrame(), pd.DataFrame()),
        (pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06"), pd.DataFrame(), pd.DataFrame()),
    ]
    cfg = _small_cfg(walk_forward_n_jobs=2, model_n_jobs=-1, model_head_n_jobs=4)

    monkeypatch.setattr(wf, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(wf, "_run_fold", _fake_run_fold)

    executed = _execute_folds(folds, ["f1"], cfg)

    assert seen_workers == [2]
    assert seen_cfg == [(1, 1), (1, 1)]
    assert len(executed) == 2


def test_execute_folds_keeps_oof_shape_consistent_between_sequential_and_parallel(monkeypatch):
    class _FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, folds):
            return [fn(fold) for fold in folds]

    def _fake_run_fold(fold, feature_columns, cfg):
        _fold_id, _train_end, valid_start, valid_end, _train_df, valid_df = fold
        result = FoldResult(
            train_end=pd.Timestamp("2020-01-01"),
            valid_start=valid_start,
            valid_end=valid_end,
            metrics={"rmse": 0.0},
        )
        oof = valid_df[["Date", "Symbol"]].copy()
        oof["predicted_return"] = range(len(oof))
        oof["up_probability"] = 0.5
        return result, oof

    folds = [
        (
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-03"),
            pd.DataFrame(),
            pd.DataFrame({"Date": pd.date_range("2020-01-02", periods=2), "Symbol": ["A", "B"]}),
        ),
        (
            pd.Timestamp("2020-01-04"),
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-01-06"),
            pd.DataFrame(),
            pd.DataFrame({"Date": pd.date_range("2020-01-05", periods=3), "Symbol": ["A", "B", "C"]}),
        ),
    ]
    monkeypatch.setattr(wf, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(wf, "_run_fold", _fake_run_fold)

    sequential = _execute_folds(folds, ["f1"], _small_cfg(walk_forward_n_jobs=1))
    parallel = _execute_folds(folds, ["f1"], _small_cfg(walk_forward_n_jobs=2))

    seq_oof = pd.concat([oof for _, oof in sequential], ignore_index=True)
    par_oof = pd.concat([oof for _, oof in parallel], ignore_index=True)
    assert len(seq_oof) == len(par_oof)
    assert list(seq_oof.columns) == list(par_oof.columns)
    assert {"predicted_return", "up_probability"}.issubset(par_oof.columns)


def test_aggregate_oof_predictions_averages_duplicate_predictions():
    raw = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 2,
            "Symbol": ["A", "A"],
            "target_log_return": [0.02, 0.02],
            "target_up": [1, 1],
            "predicted_return": [0.01, 0.03],
            "up_probability": [0.6, 0.8],
            "quantile_low": [-0.01, 0.00],
            "quantile_mid": [0.01, 0.03],
            "quantile_high": [0.04, 0.06],
            "fold_id": [0, 1],
            "train_start": [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-02-01")],
            "train_end": [pd.Timestamp("2023-12-20"), pd.Timestamp("2023-12-27")],
            "valid_start": [pd.Timestamp("2024-01-02")] * 2,
            "valid_end": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")],
        }
    )

    aggregated, diagnostics = aggregate_oof_predictions(raw)

    assert len(aggregated) == 1
    assert aggregated.loc[0, "predicted_return"] == pytest.approx(0.02)
    assert aggregated.loc[0, "up_probability"] == pytest.approx(0.7)
    assert aggregated.loc[0, "oof_prediction_count"] == 2
    assert aggregated.loc[0, "fold_ids"] == [0, 1]
    assert diagnostics["duplicate_row_count"] == 1
    assert diagnostics["duplicate_ratio"] == pytest.approx(0.5)


def test_aggregate_oof_predictions_rejects_conflicting_targets():
    raw = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 2,
            "Symbol": ["A", "A"],
            "target_log_return": [0.01, 0.02],
            "target_up": [1, 1],
            "predicted_return": [0.01, 0.02],
            "up_probability": [0.6, 0.7],
        }
    )

    with pytest.raises(ValueError, match="Conflicting OOF target values"):
        aggregate_oof_predictions(raw)


def test_aggregate_oof_predictions_records_conflicting_stable_values():
    raw = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 2,
            "Symbol": ["A", "A"],
            "target_log_return": [0.02, 0.02],
            "target_up": [1, 1],
            "predicted_return": [0.01, 0.03],
            "up_probability": [0.6, 0.8],
            "sector": ["IT", "Finance"],
        }
    )

    aggregated, diagnostics = aggregate_oof_predictions(raw)

    assert len(aggregated) == 1
    assert aggregated.loc[0, "sector"] == "IT"
    assert diagnostics["stable_conflict_count"] == 1
    assert diagnostics["stable_conflict_columns"] == {"sector": 1}


def test_run_fold_adds_fold_provenance_to_oof(monkeypatch):
    class _FakeModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            return None

        def predict(self, valid_df):
            count = len(valid_df)
            return SimpleNamespace(
                predicted_return=[0.01] * count,
                up_probability=[0.6] * count,
                quantile_low=[-0.01] * count,
                quantile_mid=[0.01] * count,
                quantile_high=[0.03] * count,
            )

    from types import SimpleNamespace

    monkeypatch.setattr(wf, "MultiHeadStockModel", _FakeModel)
    train_df = pd.DataFrame({"Date": pd.bdate_range("2023-01-01", periods=3), "feature": [1.0, 2.0, 3.0]})
    valid_df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=2),
            "Symbol": ["A", "A"],
            "Close": [100.0, 101.0],
            "market_regime": ["normal", "normal"],
            "target_log_return": [0.01, -0.01],
            "target_up": [1, 0],
            "feature": [4.0, 5.0],
        }
    )
    fold = (7, train_df["Date"].max(), valid_df["Date"].min(), valid_df["Date"].max(), train_df, valid_df)

    result, oof = _run_fold(fold, ["feature"], _small_cfg())

    assert result.fold_id == 7
    assert result.train_start == train_df["Date"].min()
    assert {"fold_id", "train_start", "train_end", "valid_start", "valid_end"}.issubset(oof.columns)
    assert oof["fold_id"].eq(7).all()
