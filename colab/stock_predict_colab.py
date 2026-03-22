from __future__ import annotations

"""
Colab-friendly single-file runner for the stock_predict project.

Usage in Google Colab:

1. Upload or clone the repository so this file exists.
2. Install requirements:
   !pip install -r requirements.txt
3. Run:
   from colab.stock_predict_colab import run_colab_pipeline
   run_colab_pipeline(
       input_csv="data/sample_ohlcv.csv",
       use_external=False,
       report_json="pipeline_report_colab.json",
       figure_dir="figures_colab",
   )

Outputs are always saved under ./result as:
- result_detail.csv
- result_simple.csv
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch_real_data import save_real_ohlcv_csv
from src.pipeline import _fallback_symbols_from_input_or_default, run_pipeline


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _looks_like_krx_symbol(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    if not s:
        return False
    if s.endswith((".KS", ".KQ")):
        return s.split(".")[0].isdigit()
    return s.isdigit() and len(s) == 6


def _should_bootstrap_default_symbols(input_csv: str | Path) -> bool:
    path = _resolve_project_path(input_csv)
    if not path.exists():
        return True

    try:
        df = pd.read_csv(path, usecols=["Symbol"])
    except Exception:
        return False

    if df.empty or "Symbol" not in df.columns:
        return True

    symbols = [str(symbol) for symbol in df["Symbol"].dropna().astype(str).unique().tolist()]
    if not symbols:
        return True
    return not all(_looks_like_krx_symbol(symbol) for symbol in symbols)


def run_colab_pipeline(
    input_csv: str = "data/sample_ohlcv.csv",
    universe_csv: str | None = None,
    report_json: str | None = "pipeline_report_colab.json",
    figure_dir: str = "figures_colab",
    use_external: bool = False,
    use_investor_context: bool = False,
    dart_api_key: str | None = None,
    dart_corp_map_csv: str | None = None,
    bootstrap_default_symbols: bool = True,
    real_start: str = "2018-01-01",
) -> dict[str, str]:
    pipeline_input = input_csv
    if bootstrap_default_symbols and _should_bootstrap_default_symbols(input_csv):
        default_symbols = _fallback_symbols_from_input_or_default(input_csv)
        bootstrap_target = PROJECT_ROOT / "data" / "real_ohlcv.csv"
        print(
            f"[Colab] demo/placeholder 입력을 감지해 기본 KRX 22종목 데이터를 먼저 수집합니다: {len(default_symbols)} symbols"
        )
        save_real_ohlcv_csv(bootstrap_target, symbols=default_symbols, start=real_start)
        pipeline_input = str(Path("data") / "real_ohlcv.csv")

    run_pipeline(
        input_csv=pipeline_input,
        output_csv="result_detail.csv",
        universe_csv=universe_csv,
        report_json=report_json,
        figure_dir=figure_dir,
        use_external=use_external,
        use_investor_context=use_investor_context,
        dart_api_key=dart_api_key,
        dart_corp_map_csv=dart_corp_map_csv,
    )

    result_dir = PROJECT_ROOT / "result"
    return {
        "result_detail_csv": str(result_dir / "result_detail.csv"),
        "result_simple_csv": str(result_dir / "result_simple.csv"),
        "report_json": str(result_dir / Path(report_json).name) if report_json else "",
        "figure_dir": str(result_dir / Path(figure_dir).name),
    }


if __name__ == "__main__":
    outputs = run_colab_pipeline()
    for key, value in outputs.items():
        if value:
            print(f"{key}: {value}")
