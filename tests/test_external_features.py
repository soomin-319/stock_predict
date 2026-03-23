from __future__ import annotations

import pandas as pd

from src.features import external_features



def test_add_external_market_features_handles_multi_column_reset_index_shape(monkeypatch):
    base = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "Symbol": ["AAA", "AAA"]})

    weird = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Adj Close": [100.0, 101.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    weird.index.name = "Date"

    monkeypatch.setattr(external_features, "_safe_download", lambda symbol, start, end: weird.copy())

    out, coverage = external_features.add_external_market_features_with_coverage(base, ["^GSPC"])

    assert coverage["successful"] == 1
    assert "gspc_close" in out.columns
    assert out["gspc_close"].notna().all()
