from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    lgb = None
    LIGHTGBM_AVAILABLE = False


@dataclass
class MultiHeadPrediction:
    predicted_return: np.ndarray
    up_probability: np.ndarray
    quantile_low: np.ndarray
    quantile_mid: np.ndarray
    quantile_high: np.ndarray
    horizon_predicted_return: dict[int, np.ndarray] = field(default_factory=dict)
    horizon_up_probability: dict[int, np.ndarray] = field(default_factory=dict)


class MultiHeadStockModel:
    """회귀 + 방향분류 + 분위수 추정을 위한 멀티헤드 모델.

    LightGBM이 설치된 환경이면 LightGBM을 사용하고, 없으면 sklearn GBDT로 fallback한다.
    """

    def __init__(self, random_state: int = 42, n_jobs: int | None = None, use_gpu: bool = False):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self._feature_columns: List[str] = []
        self.reg_model = None
        self.cls_model = None
        self.quantile_models: Dict[float, Any] = {}
        self.horizon_reg_models: Dict[int, Any] = {}
        self.horizon_cls_models: Dict[int, Any] = {}
        self.backend: str = "lightgbm" if LIGHTGBM_AVAILABLE else "sklearn"

    def _lightgbm_params(self) -> dict[str, Any]:
        params: dict[str, Any] = dict(
            random_state=self.random_state,
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
        if self.n_jobs is not None:
            params["n_jobs"] = self.n_jobs
        if self.use_gpu:
            params["device"] = "gpu"
        return params

    def _build_regressor(self, loss: str = "squared_error", alpha: float | None = None):
        if LIGHTGBM_AVAILABLE:
            params = self._lightgbm_params()
            if loss == "quantile":
                params["objective"] = "quantile"
                params["alpha"] = alpha
            else:
                params["objective"] = "regression"
            return lgb.LGBMRegressor(**params)

        kwargs = dict(random_state=self.random_state, n_estimators=250, learning_rate=0.03, max_depth=3)
        if loss == "quantile":
            kwargs["loss"] = "quantile"
            kwargs["alpha"] = alpha
        else:
            kwargs["loss"] = loss
        return GradientBoostingRegressor(**kwargs)

    def _build_classifier(self):
        if LIGHTGBM_AVAILABLE:
            params = self._lightgbm_params()
            params["objective"] = "binary"
            return lgb.LGBMClassifier(**params)

        return GradientBoostingClassifier(
            random_state=self.random_state,
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
        )

    def fit(self, df: pd.DataFrame, feature_columns: List[str], quantiles: List[float]):
        train = df.dropna(subset=feature_columns + ["target_log_return", "target_up"])
        x = train[feature_columns]
        y_reg = train["target_log_return"]
        y_cls = train["target_up"]

        self._feature_columns = feature_columns
        self.reg_model = self._build_regressor()
        self.reg_model.fit(x, y_reg)

        self.cls_model = self._build_classifier()
        self.cls_model.fit(x, y_cls)

        self.quantile_models = {}
        for q in quantiles:
            qm = self._build_regressor(loss="quantile", alpha=q)
            qm.fit(x, y_reg)
            self.quantile_models[q] = qm

        self.horizon_reg_models = {}
        self.horizon_cls_models = {}
        for horizon in (5, 20):
            reg_target = f"target_log_return_{horizon}d"
            cls_target = f"target_up_{horizon}d"
            if reg_target not in train.columns or cls_target not in train.columns:
                continue
            horizon_train = train.dropna(subset=feature_columns + [reg_target, cls_target])
            if horizon_train.empty:
                continue
            horizon_x = horizon_train[feature_columns]
            reg_model = self._build_regressor()
            reg_model.fit(horizon_x, horizon_train[reg_target])
            cls_model = self._build_classifier()
            cls_model.fit(horizon_x, horizon_train[cls_target])
            self.horizon_reg_models[horizon] = reg_model
            self.horizon_cls_models[horizon] = cls_model

    def predict(self, df: pd.DataFrame) -> MultiHeadPrediction:
        if not self._feature_columns:
            raise RuntimeError("Model is not fitted.")

        x = df[self._feature_columns].copy().fillna(0)

        predicted_return = self.reg_model.predict(x)
        up_probability = self.cls_model.predict_proba(x)[:, 1]

        quantiles = sorted(self.quantile_models)
        q_preds = {q: self.quantile_models[q].predict(x) for q in quantiles}
        horizon_predicted_return = {
            horizon: model.predict(x) for horizon, model in self.horizon_reg_models.items()
        }
        horizon_up_probability = {
            horizon: model.predict_proba(x)[:, 1] for horizon, model in self.horizon_cls_models.items()
        }

        return MultiHeadPrediction(
            predicted_return=predicted_return,
            up_probability=up_probability,
            quantile_low=q_preds[quantiles[0]],
            quantile_mid=q_preds[quantiles[1]],
            quantile_high=q_preds[quantiles[2]],
            horizon_predicted_return=horizon_predicted_return,
            horizon_up_probability=horizon_up_probability,
        )
