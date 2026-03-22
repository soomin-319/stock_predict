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

    assert len(symbols) == 100
    assert symbols[0] == "005930.KS"
    assert symbols[49] == "012450.KS"
    assert symbols[50] == "247540.KQ"
    assert symbols[-1] == "900140.KQ"
    assert universe_module.DEFAULT_UNIVERSE_CSV == Path("data/default_universe_kospi50_kosdaq50.csv").resolve()
