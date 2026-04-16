from pathlib import Path

from src.data import krx_universe



def test_get_symbol_name_map_uses_csv_mapping(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "krx_symbol_name_map.csv"
    csv_path.write_text(
        "Ticker,Symbol,Name,Market\n005930,005930.KS,삼성전자,KOSPI\n000660,000660.KS,SK하이닉스,KOSPI\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(krx_universe, "KRX_SYMBOL_NAME_CSV", csv_path)
    krx_universe._load_krx_symbol_name_df.cache_clear()

    assert krx_universe.get_symbol_name_map(["005930.KS", "000660.KS"]) == {
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스",
    }



def test_get_symbol_name_map_returns_symbol_for_missing_csv_entries(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "krx_symbol_name_map.csv"
    csv_path.write_text(
        "Ticker,Symbol,Name,Market\n005930,005930.KS,삼성전자,KOSPI\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(krx_universe, "KRX_SYMBOL_NAME_CSV", csv_path)
    krx_universe._load_krx_symbol_name_df.cache_clear()

    assert krx_universe.get_symbol_name_map(["005930.KS", "000660.KS"]) == {
        "005930.KS": "삼성전자",
        "000660.KS": "000660.KS",
    }



def test_find_symbol_candidates_by_name_returns_exact_then_similar_matches_from_csv(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "krx_symbol_name_map.csv"
    csv_path.write_text(
        "Ticker,Symbol,Name,Market\n005930,005930.KS,삼성전자,KOSPI\n005935,005935.KS,삼성전자우,KOSPI\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(krx_universe, "KRX_SYMBOL_NAME_CSV", csv_path)
    krx_universe._load_krx_symbol_name_df.cache_clear()

    candidates = krx_universe.find_symbol_candidates_by_name("삼성전자")

    assert candidates[0]["ticker"] == "005930"
    assert candidates[0]["symbol"] == "005930.KS"
    assert candidates[1]["ticker"] == "005935"
