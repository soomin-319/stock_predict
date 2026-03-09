from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


@dataclass
class MultiHeadPrediction:
    predicted_return: np.ndarray
    up_probability: np.ndarray
    quantile_low: np.ndarray
    quantile_mid: np.ndarray
    quantile_high: np.ndarray


class MultiHeadStockModel:
    """회귀 + 방향분류 + 분위수 추정을 위한 멀티헤드 모델.

    LightGBM이 설치된 환경이면 LightGBM을 사용하고, 없으면 sklearn GBDT로 fallback한다.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._feature_columns: List[str] = []
        self.reg_model = None
        self.cls_model = None
        self.quantile_models: Dict[float, GradientBoostingRegressor] = {}

    def _build_regressor(self, loss: str = "squared_error", alpha: float | None = None):
        kwargs = dict(random_state=self.random_state, n_estimators=250, learning_rate=0.03, max_depth=3)
        if loss == "quantile":
            kwargs["loss"] = "quantile"
            kwargs["alpha"] = alpha
        else:
            kwargs["loss"] = loss
        return GradientBoostingRegressor(**kwargs)

    def fit(self, df: pd.DataFrame, feature_columns: List[str], quantiles: List[float]):
        train = df.dropna(subset=feature_columns + ["target_log_return", "target_up"])
        x = train[feature_columns]
        y_reg = train["target_log_return"]
        y_cls = train["target_up"]

        self._feature_columns = feature_columns
        self.reg_model = self._build_regressor()
        self.reg_model.fit(x, y_reg)

        self.cls_model = GradientBoostingClassifier(
            random_state=self.random_state,
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
        )
        self.cls_model.fit(x, y_cls)

        self.quantile_models = {}
        for q in quantiles:
            qm = self._build_regressor(loss="quantile", alpha=q)
            qm.fit(x, y_reg)
            self.quantile_models[q] = qm

    def predict(self, df: pd.DataFrame) -> MultiHeadPrediction:
        if not self._feature_columns:
            raise RuntimeError("Model is not fitted.")

        x = df[self._feature_columns].copy()
        x = x.fillna(0)

        predicted_return = self.reg_model.predict(x)
        up_probability = self.cls_model.predict_proba(x)[:, 1]

        quantiles = sorted(self.quantile_models)
        q_preds = {q: self.quantile_models[q].predict(x) for q in quantiles}

        return MultiHeadPrediction(
            predicted_return=predicted_return,
            up_probability=up_probability,
            quantile_low=q_preds[quantiles[0]],
            quantile_mid=q_preds[quantiles[1]],
            quantile_high=q_preds[quantiles[2]],
        )
