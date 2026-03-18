from __future__ import annotations

import warnings


def import_pykrx_stock():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from pykrx import stock

        return stock
    except Exception:
        return None
