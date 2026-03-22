import pandas as pd

from colab.stock_predict_colab import _format_price_display, _prepare_colab_preview


def test_format_price_display_handles_numeric_and_text_inputs():
    assert _format_price_display(71200) == "71,200원"
    assert _format_price_display("71200") == "71,200원"
    assert _format_price_display("71,200원") == "71,200원"


def test_prepare_colab_preview_normalizes_stock_code_and_predicted_close():
    preview = _prepare_colab_preview(
        pd.DataFrame(
            [
                {
                    "종목코드": "5930",
                    "종목명": "삼성전자",
                    "내일 예상 종가": "71200",
                }
            ]
        )
    )

    assert preview.loc[0, "종목코드"] == "005930"
    assert preview.loc[0, "내일 예상 종가"] == "71,200원"
