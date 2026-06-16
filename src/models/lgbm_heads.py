from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

MODEL_ARTIFACT_VERSION = 2
NEUTRAL_FEATURE_DEFAULTS: dict[str, float] = {
    "rsi_14": 50.0,
    "stoch_k": 50.0,
    "stoch_d": 50.0,
    "cci_20": 0.0,
}


def _validate_quantiles(quantiles: List[float]) -> list[float]:
    values = list(quantiles)
    if len(values) < 3:
        raise ValueError(f"quantiles must contain at least 3 values, got {values}")
    if any(not isinstance(q, (int, float)) or isinstance(q, bool) or not 0 < q < 1 for q in values):
        raise ValueError(f"quantiles must be numeric values strictly between 0 and 1, got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"quantiles must not contain duplicates, got {values}")
    if values != sorted(values):
        raise ValueError(f"quantiles must be strictly increasing, got {values}")
    return values


def _fit_one(task):
    """joblib Parallel 호환을 위한 모듈 수준 헬퍼."""
    model, x, y = task
    model.fit(x, y)
    return model

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    lgb = None
    LIGHTGBM_AVAILABLE = False


def _require_sklearn_ensemble():
    try:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required when LightGBM is unavailable. "
            "Install scikit-learn/lightgbm before training."
        ) from exc
    return GradientBoostingClassifier, GradientBoostingRegressor


def _require_joblib():
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RuntimeError("joblib is required for parallel model heads and model persistence.") from exc
    return joblib


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

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int | None = None,
        use_gpu: bool = False,
        head_n_jobs: int | None = 1,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.head_n_jobs = 1 if head_n_jobs is None else int(head_n_jobs or 1)
        self._feature_columns: List[str] = []
        self._feature_imputer_values: Dict[str, float] = {}
        self.reg_model = None
        self.cls_model = None
        self.quantile_models: Dict[float, Any] = {}
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

        _, GradientBoostingRegressor = _require_sklearn_ensemble()
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

        GradientBoostingClassifier, _ = _require_sklearn_ensemble()
        return GradientBoostingClassifier(
            random_state=self.random_state,
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
        )

    def _compute_feature_imputer_values(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for column in feature_columns:
            if column in NEUTRAL_FEATURE_DEFAULTS:
                values[column] = NEUTRAL_FEATURE_DEFAULTS[column]
                continue
            series = pd.to_numeric(df[column], errors="coerce")
            median = series.median()
            if pd.isna(median):
                values[column] = 0.0
            else:
                values[column] = float(median)
        return values

    def _impute_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self._feature_columns].copy()
        for column in self._feature_columns:
            x[column] = pd.to_numeric(x[column], errors="coerce").fillna(self._feature_imputer_values.get(column, 0.0))
        return x

    def fit(self, df: pd.DataFrame, feature_columns: List[str], quantiles: List[float]):
        quantiles = _validate_quantiles(quantiles)
        if not feature_columns:
            raise ValueError("feature_columns must contain at least one feature.")
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Training data is missing {len(missing)} feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        required_targets = ["target_log_return", "target_up"]
        missing_targets = [c for c in required_targets if c not in df.columns]
        if missing_targets:
            raise ValueError(f"Training data is missing target columns: {missing_targets}")

        target_ready = df.dropna(subset=required_targets).copy()
        usable_mask = target_ready[feature_columns].notna().any(axis=1)
        train = target_ready[usable_mask].copy()
        if train.empty:
            raise ValueError("No usable training rows after dropping missing targets and all-missing feature rows.")
        y_cls = train["target_up"]
        if y_cls.nunique(dropna=True) < 2:
            raise ValueError("target_up must contain at least two classes for classification training.")

        self._feature_columns = list(feature_columns)
        self._feature_imputer_values = self._compute_feature_imputer_values(target_ready, self._feature_columns)
        x = self._impute_features(train)
        y_reg = train["target_log_return"]

        # 모든 독립 모델 학습 태스크를 수집한다.
        tasks: list[tuple] = [
            (self._build_regressor(), x, y_reg),
            (self._build_classifier(), x, y_cls),
            *[(self._build_regressor(loss="quantile", alpha=q), x, y_reg) for q in quantiles],
        ]

        # ③: LightGBM은 C++ 구간에서 GIL을 해제하므로 스레드 병렬로 실질적인 속도 향상이 가능하다.
        if self.head_n_jobs == 1:
            fitted = [_fit_one(task) for task in tasks]
        else:
            joblib = _require_joblib()
            fitted = joblib.Parallel(n_jobs=self.head_n_jobs, prefer="threads")(
                joblib.delayed(_fit_one)(task) for task in tasks
            )

        self.reg_model = fitted[0]
        self.cls_model = fitted[1]
        self.quantile_models = {q: fitted[2 + i] for i, q in enumerate(quantiles)}

    def predict(self, df: pd.DataFrame) -> MultiHeadPrediction:
        if not self._feature_columns:
            raise RuntimeError("Model is not fitted.")

        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} feature columns used during training: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        x = self._impute_features(df)

        predicted_return = self.reg_model.predict(x)
        up_probability = self.cls_model.predict_proba(x)[:, 1]

        quantiles = sorted(self.quantile_models)
        q_preds = {q: self.quantile_models[q].predict(x) for q in quantiles}
        if len(quantiles) < 3:
            raise RuntimeError(
                f"MultiHeadPrediction requires at least 3 quantile heads, got {len(quantiles)}: {quantiles}"
            )
        selected = np.vstack(
            [
                q_preds[quantiles[0]],
                q_preds[quantiles[len(quantiles) // 2]],
                q_preds[quantiles[-1]],
            ]
        ).T
        selected = np.sort(selected, axis=1)
        return MultiHeadPrediction(
            predicted_return=predicted_return,
            up_probability=up_probability,
            quantile_low=selected[:, 0],
            quantile_mid=selected[:, 1],
            quantile_high=selected[:, 2],
        )

    def _feature_columns_hash(self) -> str:
        """Stable hash of training feature columns for artifact compatibility checks."""
        blob = "|".join(self._feature_columns).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "backend": self.backend,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "head_n_jobs": self.head_n_jobs,
            "use_gpu": self.use_gpu,
            "feature_count": len(self._feature_columns),
            "feature_hash": self._feature_columns_hash() if self._feature_columns else None,
            "imputer_feature_count": len(self._feature_imputer_values),
            "quantiles": sorted(self.quantile_models.keys()),
        }

    def feature_importance_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        def _append(head: str, model: Any) -> None:
            values = getattr(model, "feature_importances_", None)
            if values is None:
                values = getattr(model, "coef_", None)
            if values is None:
                return
            arr = np.asarray(values, dtype=float).reshape(-1)
            if len(arr) != len(self._feature_columns):
                return
            for feature, importance in zip(self._feature_columns, arr):
                rows.append({"head": head, "feature": feature, "importance": float(importance)})

        _append("regression", self.reg_model)
        _append("classification", self.cls_model)
        for quantile, model in sorted(self.quantile_models.items()):
            _append(f"quantile_{quantile:g}", model)
        return pd.DataFrame(rows, columns=["head", "feature", "importance"])

    def save(self, path: str | Path) -> Path:
        """Persist a fitted model to disk with metadata for reproducibility.

        Stores a joblib bundle at ``path`` plus a sidecar ``<path>.meta.json`` so
        the training context (seed, backend, feature hash) is discoverable
        without loading the pickle.
        """
        if not self._feature_columns:
            raise RuntimeError("Cannot save an unfitted model.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "backend": self.backend,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "head_n_jobs": self.head_n_jobs,
            "use_gpu": self.use_gpu,
            "feature_columns": list(self._feature_columns),
            "feature_imputer_values": dict(self._feature_imputer_values),
            "feature_hash": self._feature_columns_hash(),
            "reg_model": self.reg_model,
            "cls_model": self.cls_model,
            "quantile_models": self.quantile_models,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib = _require_joblib()
        joblib.dump(payload, out_path)

        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta = {**self.metadata(), "saved_at": payload["saved_at"], "path": str(out_path)}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "MultiHeadStockModel":
        joblib = _require_joblib()
        payload = joblib.load(Path(path))
        version = payload.get("artifact_version")
        if version != MODEL_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported model artifact version {version!r}; expected {MODEL_ARTIFACT_VERSION}."
            )
        model = cls(
            random_state=payload.get("random_state", 42),
            n_jobs=payload.get("n_jobs"),
            use_gpu=payload.get("use_gpu", False),
            head_n_jobs=payload.get("head_n_jobs", 1),
        )
        model.backend = payload.get("backend", model.backend)
        model._feature_columns = list(payload["feature_columns"])
        model._feature_imputer_values = {
            str(key): float(value) for key, value in payload.get("feature_imputer_values", {}).items()
        }
        if not model._feature_imputer_values:
            model._feature_imputer_values = {column: 0.0 for column in model._feature_columns}
        model.reg_model = payload["reg_model"]
        model.cls_model = payload["cls_model"]
        model.quantile_models = payload.get("quantile_models", {})

        stored_hash = payload.get("feature_hash")
        if stored_hash and stored_hash != model._feature_columns_hash():
            raise ValueError("Feature-hash mismatch: saved feature list inconsistent with artifact.")
        return model
