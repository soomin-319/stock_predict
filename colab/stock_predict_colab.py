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

from src.pipeline import run_pipeline


def _format_price_display(value) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return f"{float(numeric):,.0f}원"

    text = str(value or "").strip()
    if not text:
        return "-"
    if text.endswith("원"):
        return text

    cleaned = text.replace(",", "").replace("원", "")
    numeric = pd.to_numeric(pd.Series([cleaned]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return text
    return f"{float(numeric):,.0f}원"


def _prepare_colab_preview(simple: pd.DataFrame) -> pd.DataFrame:
    preview = simple.copy()
    if "종목코드" in preview.columns:
        preview["종목코드"] = preview["종목코드"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if "내일 예상 종가" in preview.columns:
        preview["내일 예상 종가"] = preview["내일 예상 종가"].map(_format_price_display)
    return preview


def _print_colab_preview(result_simple_path: Path):
    if not result_simple_path.exists():
        return
    try:
        simple = pd.read_csv(result_simple_path, encoding="utf-8-sig", dtype={"종목코드": str})
    except Exception:
        return
    if simple.empty:
        return

    preview = _prepare_colab_preview(simple)
    preview_columns = [
        column
        for column in ["종목코드", "종목명", "권고", "내일 예상 종가", "내일 예상 수익률(%)", "상승확률(%)", "예측 이유"]
        if column in preview.columns
    ]
    print("\n=== Colab Prediction Preview ===")
    head = preview[preview_columns].head(10)
    try:
        from IPython.display import display

        display(head)
    except Exception:
        print(head.to_string(index=False))
    print("\nresult_simple.csv는 UTF-8 BOM(utf-8-sig)으로 저장되었습니다.")


def run_colab_pipeline(
    input_csv: str = "data/sample_ohlcv.csv",
    universe_csv: str | None = None,
    report_json: str | None = "pipeline_report_colab.json",
    figure_dir: str = "figures_colab",
    use_external: bool = False,
    use_investor_context: bool = False,
    dart_api_key: str | None = None,
    dart_corp_map_csv: str | None = None,
) -> dict[str, str]:
    run_pipeline(
        input_csv=input_csv,
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
    _print_colab_preview(result_dir / "result_simple.csv")
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
