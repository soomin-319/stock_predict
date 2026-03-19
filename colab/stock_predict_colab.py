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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


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
