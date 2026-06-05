from pathlib import Path

import pandas as pd
import pytest

from src.data import universe as universe_module
from src.data.universe import load_default_universe_symbols, load_universe_symbols, load_universe_symbols_list



def test_load_universe_symbols_requires_explicit_path():
    with pytest.raises(TypeError):
        load_universe_symbols()



def test_load_universe_symbols_reads_symbol_column(tmp_path):
    path = tmp_path / "universe.csv"
    pd.DataFrame({"Symbol": ["005930.KS", "000660.KS", "005930.KS"]}).to_csv(path, index=False)

    assert load_universe_symbols(str(path)) == {"005930.KS", "000660.KS"}
    assert load_universe_symbols_list(str(path)) == ["005930.KS", "000660.KS"]



def test_load_default_universe_symbols_uses_repo_csv():
    symbols = load_default_universe_symbols()

    assert len(symbols) == 200
    assert symbols[0] == "018880.KS"
    assert symbols[49] == "007340.KS"
    assert symbols[-1] == "0126Z0.KS"
    assert all(symbol.endswith(".KS") for symbol in symbols)
    assert universe_module.DEFAULT_UNIVERSE_CSV == Path("data/kospi200_symbol_name_map.csv").resolve()
