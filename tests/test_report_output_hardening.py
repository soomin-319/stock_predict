from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reports import output


def test_drop_empty_detail_columns_preserves_optional_schema_by_default():
    detail_df = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "predicted_return": 1.2,
                "foreign_net_buy": pd.NA,
                "news_sentiment": "",
                "target_up": pd.NA,
            }
        ]
    )

    cleaned = output.drop_empty_detail_columns(detail_df)

    assert cleaned.columns.tolist() == detail_df.columns.tolist()
    assert "foreign_net_buy" in cleaned.columns
    assert "news_sentiment" in cleaned.columns
    assert "target_up" in cleaned.columns


def test_drop_empty_detail_columns_can_prune_when_requested():
    detail_df = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "predicted_return": 1.2,
                "foreign_net_buy": pd.NA,
                "news_sentiment": "",
                "target_up": pd.NA,
            }
        ]
    )

    cleaned = output.drop_empty_detail_columns(detail_df, prune_empty_optional=True)

    assert cleaned.columns.tolist() == ["Symbol", "predicted_return"]


def test_build_combined_symbol_results_reads_summary_with_utf8_sig(monkeypatch, tmp_path: Path):
    calls: list[dict] = []
    summary = pd.DataFrame([{"Symbol": "005930.KS", "extra": "x"}])

    def fake_read_csv(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return summary

    monkeypatch.setattr(output.pd, "read_csv", fake_read_csv)

    saved = output.build_combined_symbol_results(
        pd.DataFrame([{"Symbol": "005930.KS", "predicted_return": 1.2}]),
        "summary.csv",
        tmp_path / "combined.csv",
    )

    assert saved == str(tmp_path / "combined.csv")
    assert calls == [{"path": "summary.csv", "encoding": "utf-8-sig"}]
