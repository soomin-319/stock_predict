import numpy as np
import pandas as pd

from src.config.settings import AppConfig
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.models.lgbm_heads import MultiHeadStockModel


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
        or c in {"daily_return", "gap_return", "intraday_return", "range_pct", "vol_ratio_20", "rsi_14"}
    ]

    train = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(random_state=cfg.training.random_state)
    model.fit(train, feature_columns, cfg.training.quantiles)

    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(3).fillna(0)
    pred = model.predict(latest)

    assert len(pred.predicted_return) == len(latest)
    assert len(pred.up_probability) == len(latest)
    assert (pred.quantile_high >= pred.quantile_low).all()
