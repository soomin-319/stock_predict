import pandas as pd
import pytest

from src.data.universe import load_universe_symbols


def test_load_universe_symbols_requires_explicit_path():
    with pytest.raises(TypeError):
        load_universe_symbols()


def test_load_universe_symbols_reads_symbol_column(tmp_path):
    path = tmp_path / "universe.csv"
    pd.DataFrame({"Symbol": ["005930.KS", "000660.KS", "005930.KS"]}).to_csv(path, index=False)

    assert load_universe_symbols(str(path)) == {"005930.KS", "000660.KS"}
