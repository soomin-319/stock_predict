#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_targetdate_safe_final_v15.py

v15 구조 보강 요약
1) live 샘플(today->tomorrow)과 train/eval 샘플 완전 분리
2) calibration scale 안정화(검증구간 rolling + 중앙값)
3) magnitude weighting의 배치 의존 제거(global_mean_abs 상수 사용)
4) 동적 임계값 실제 동작(predicted return volatility 기반)
5) 추가 지표: 방향성 hit ratio, 수수료 반영 전략 수익률
"""

from __future__ import annotations

import os
import time
import random
import json
import logging
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("pipeline_targetdate_safe_final_v15")


def configure_gpu_for_local() -> None:
    """로컬 환경에서 TensorFlow가 GPU를 안정적으로 사용하도록 설정."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPU detected. Running on CPU.")
            return

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.warning("GPU enabled (%d found), mixed_float16 policy applied.", len(gpus))
        except Exception as e:
            logger.warning("GPU enabled (%d found), mixed precision not applied: %s", len(gpus), e)
    except Exception as e:
        logger.warning("GPU configuration failed, fallback to default device config: %s", e)


configure_gpu_for_local()

WINDOW_SIZE = 30
TEST_RATIO = 0.2
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.1
L2_WEIGHT = 1e-5
DIRECTION_WEIGHT = 0.2
MAG_WEIGHT = 2.0
CALIB_CLIP = 8.0

CORE_FEATURES = [
    'rv_20', 'ATR14', 'mom_5', 'mom_10', 'RSI14', 'bb_position', 'vol_surge',
]

SYMBOL_META_CSV = 'symbol_meta.csv'
RESULT_CSV = 'result.csv'
PER_MODEL_CSV = 'per_model.csv'
DEBUG_PRED_CSV = 'debug_predictions.csv'
ZIP_PATH = 'results.zip'
SCALER_DIR = 'scalers'
CACHE_DIR = 'external_cache'
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def magnitude_weighted_directional_huber_loss(
    global_mean_abs: float,
    delta: float = 0.5,
    direction_weight: float = DIRECTION_WEIGHT,
    mag_weight: float = MAG_WEIGHT,
):
    """global_mean_abs를 고정 상수로 사용해 배치 의존성 제거."""
    gm = tf.constant(max(global_mean_abs, 1e-8), dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        abs_err = tf.abs(y_true - y_pred)
        huber = tf.where(abs_err <= delta, 0.5 * tf.square(abs_err), delta * abs_err - 0.5 * delta**2)
        direction_penalty = tf.nn.relu(-y_true * y_pred)
        w = 1.0 + mag_weight * tf.abs(y_true) / gm
        return tf.reduce_mean(w * (huber + direction_weight * direction_penalty))

    loss_fn.__name__ = 'mag_weighted_directional_huber_global'
    return loss_fn


def fdr_read(symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    for i in range(3):
        try:
            return fdr.DataReader(symbol, start, end)
        except Exception as e:
            logger.warning("FDR fetch failed for %s (attempt %d): %s", symbol, i + 1, e)
            time.sleep(1 + i)
    raise RuntimeError(f"Failed to fetch {symbol}")


def compute_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_indicators_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['RSI14'] = compute_rsi_wilder(df['Close'], 14)

    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()

    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['rv_20'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
    df.drop(columns=['log_ret'], inplace=True, errors='ignore')

    df['ret_t'] = df['Close'].pct_change(fill_method=None)
    df['ret_t_1'] = df['Close'].pct_change(fill_method=None).shift(1)
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['range'] = (df['High'] - df['Low']) / df['Close']

    df['mom_5'] = df['Close'].pct_change(5, fill_method=None)
    df['mom_10'] = df['Close'].pct_change(10, fill_method=None)

    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['bb_position'] = ((df['Close'] - ma20) / (2 * std20 + 1e-9)).clip(-1, 1)

    if 'Volume' in df.columns:
        vma = df['Volume'].rolling(20).mean().replace(0, np.nan)
        df['vol_surge'] = (df['Volume'] / vma).clip(upper=10).fillna(1.0)
    else:
        df['vol_surge'] = 1.0
    return df


def _cache_path(symbol: str) -> str:
    safe = symbol.replace('/', '_').replace('\\', '_').replace(':', '_')
    return os.path.join(CACHE_DIR, f"{safe}.parquet")


def load_cached_series(symbol: str, index: pd.DatetimeIndex) -> Optional[pd.Series]:
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        s = df['close'] if 'close' in df.columns else df.iloc[:, 0]
        return s.reindex(index).ffill(limit=3)
    except Exception:
        return None


def save_series_cache(symbol: str, series: pd.Series):
    try:
        series.to_frame(name='close').to_parquet(_cache_path(symbol))
    except Exception:
        pass


def load_symbol_meta(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        mdf = pd.read_csv(path)
        if 'symbol' not in mdf.columns:
            return {}
        name_col = next((c for c in ['longName', 'long_name', 'name'] if c in mdf.columns), None)
        if name_col is None:
            return {}
        out = {}
        for _, r in mdf.iterrows():
            if pd.isna(r.get(name_col)) or pd.isna(r.get('symbol')):
                continue
            out[str(r[name_col])] = str(r['symbol'])
        return out
    except Exception:
        return {}


def _fetch_single_external(name: str, symbol: str, start: str, end: str, idx: pd.DatetimeIndex):
    cached = load_cached_series(symbol, idx)
    if cached is not None and cached.notna().sum() > 0:
        return name, cached
    try:
        df = fdr_read(symbol, start, end)
    except Exception:
        return name, pd.Series(index=idx, dtype=float)
    series = None
    for col in ['Close', 'Adj Close']:
        if col in df.columns:
            series = df[col]
            break
    if series is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            series = df[num_cols[0]]
    if series is None:
        return name, pd.Series(index=idx, dtype=float)
    s = series.reindex(idx).ffill(limit=3)
    save_series_cache(symbol, s)
    return name, s


def fetch_external_series_parallel(candidates: Dict[str, str], start: str, end: str, idx: pd.DatetimeIndex, max_workers: int = 8) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(index=idx)
    results: Dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_single_external, n, s, start, end, idx): n for n, s in candidates.items()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                n, ser = fut.result()
                results[n] = ser
            except Exception:
                results[name] = pd.Series(index=idx, dtype=float)
    out = pd.concat(results.values(), axis=1)
    out.columns = list(results.keys())
    out.index = idx
    return out


def rank_externals_by_correlation(df_target: pd.DataFrame, externals_df: pd.DataFrame, top_n: int = 5, min_obs: int = 30) -> List[str]:
    if externals_df.empty:
        return []
    y = df_target['Close'].pct_change(fill_method=None).shift(-1)
    vals = {}
    for col in externals_df.columns:
        x = externals_df[col].pct_change(fill_method=None)
        c = pd.concat([x, y], axis=1).dropna()
        if len(c) < min_obs:
            continue
        stdx = float(c.iloc[:, 0].std())
        stdy = float(c.iloc[:, 1].std())
        if stdx < 1e-12 or stdy < 1e-12:
            continue
        vals[col] = abs(float(c.iloc[:, 0].corr(c.iloc[:, 1])))
    return pd.Series(vals).sort_values(ascending=False).head(top_n).index.tolist() if vals else []


def prepare_sequences_with_live_split(
    df: pd.DataFrame,
    external_map: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    target_col: str = 'ret_t1',
    nan_threshold: float = 0.35,
) -> Dict[str, Any]:
    """train/eval과 live(today->tomorrow)를 분리하여 반환."""
    df = df.copy()
    df[target_col] = df['Close'].pct_change(fill_method=None).shift(-1)

    for f in CORE_FEATURES:
        if f not in df.columns:
            df[f] = np.nan

    short_feats = [c for c in ['ret_t', 'ret_t_1', 'gap', 'range'] if c in df.columns]
    ext_ret_cols = []
    for name in external_map.columns:
        col = f'ext_{name}'
        df[col] = external_map[name]
        rc = f'{col}_ret'
        df[rc] = df[col].pct_change(fill_method=None)
        lcol = f'{rc}_lag0'
        df[lcol] = df[rc]
        ext_ret_cols.append(lcol)

    all_features = CORE_FEATURES + short_feats + ext_ret_cols
    fill_cols = all_features + ['Close']
    df[fill_cols] = df[fill_cols].interpolate(method='time').ffill(limit=3).bfill(limit=3)

    nan_ratios = {f: df[f].isna().mean() for f in all_features}
    features_filtered = [f for f in all_features if nan_ratios.get(f, 1.0) <= nan_threshold]

    # 평가 가능한 샘플(정답 존재)만
    df_eval = df[features_filtered + ['Close', target_col, 'rv_20']].dropna().copy()
    if df_eval.empty or len(df_eval) < window_size + 10:
        raise ValueError("Not enough rows after preprocessing.")

    X_eval, y_eval, dates_eval, inp_eval, tgt_eval, vol_eval = [], [], [], [], [], []
    eval_idx = df_eval.index
    full_idx = df.index
    Xv = df_eval[features_filtered].values
    Yv = df_eval[target_col].values

    for i in range(window_size, len(df_eval)):
        s, e = i - window_size, i
        inp_idx = eval_idx[e - 1]
        try:
            pos = full_idx.get_loc(inp_idx)
        except KeyError:
            pos = full_idx.get_indexer([inp_idx], method='nearest')[0]
        if pos + 1 >= len(full_idx):
            continue
        t_date = full_idx[pos + 1]
        if pd.isna(df.loc[t_date, 'Close']):
            continue

        X_eval.append(Xv[s:e])
        y_eval.append(float(Yv[e - 1]))
        dates_eval.append(pd.to_datetime(t_date).normalize())
        inp_eval.append(float(df.loc[inp_idx, 'Close']))
        tgt_eval.append(float(df.loc[t_date, 'Close']))

        rv_annual = float(df.loc[inp_idx, 'rv_20']) if np.isfinite(df.loc[inp_idx, 'rv_20']) else 0.16
        vol_eval.append(max(rv_annual / np.sqrt(252), 0.001))

    # live sample 분리
    live_sample = None
    today_df = df[features_filtered + ['Close', 'rv_20']].dropna()
    if len(today_df) >= window_size:
        last_X = today_df[features_filtered].values[-window_size:]
        last_date = today_df.index[-1]
        live_sample = {
            'X': np.array(last_X, dtype=float),
            'input_close': float(today_df['Close'].iloc[-1]),
            'target_date': pd.bdate_range(last_date, periods=2)[-1].normalize(),
            'vol_factor': max(float(today_df['rv_20'].iloc[-1]) / np.sqrt(252), 0.001),
        }

    return {
        'X': np.array(X_eval),
        'y': np.array(y_eval),
        'target_dates': dates_eval,
        'target_closes': np.array(tgt_eval),
        'input_closes': np.array(inp_eval),
        'vol_norm_factors': np.array(vol_eval),
        'features_used': features_filtered,
        'live_sample': live_sample,
    }


def build_lstm_model(
    input_shape: Tuple[int, int],
    global_mean_abs: float,
    lr: float = LR,
    dropout_rate: float = DROPOUT,
    l2_weight: float = L2_WEIGHT,
    direction_weight: float = DIRECTION_WEIGHT,
    mag_weight: float = MAG_WEIGHT,
) -> tf.keras.Model:
    w, f = input_shape
    model = Sequential([
        Input(shape=(w, f)),
        LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(l2_weight)),
        LayerNormalization(),
        Dropout(dropout_rate),
        LSTM(16, kernel_regularizer=regularizers.l2(l2_weight)),
        LayerNormalization(),
        Dropout(dropout_rate),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_weight)),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=magnitude_weighted_directional_huber_loss(
            global_mean_abs=global_mean_abs,
            direction_weight=direction_weight,
            mag_weight=mag_weight,
        )
    )
    return model


def robust_calibration_scale(
    y_true_raw: np.ndarray,
    pred_raw: np.ndarray,
    clip_min: float = 0.2,
    clip_max: float = CALIB_CLIP,
    window: int = 30,
) -> float:
    """rolling std ratio의 중앙값으로 보정 계수 안정화."""
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    pred_raw = np.asarray(pred_raw, dtype=float)
    valid = np.isfinite(y_true_raw) & np.isfinite(pred_raw)
    y_true_raw = y_true_raw[valid]
    pred_raw = pred_raw[valid]
    if len(y_true_raw) < 10:
        return 1.0

    scales = []
    w = max(10, min(window, len(y_true_raw)))
    for i in range(w, len(y_true_raw) + 1):
        a = y_true_raw[i - w:i]
        p = pred_raw[i - w:i]
        ps = np.std(p)
        if ps < 1e-8:
            continue
        scales.append(np.std(a) / ps)

    if not scales:
        ps = np.std(pred_raw)
        return float(np.clip((np.std(y_true_raw) / ps) if ps > 1e-8 else 1.0, clip_min, clip_max))

    return float(np.clip(np.median(scales), clip_min, clip_max))


def directional_hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return None
    t = np.sign(y_true[valid])
    p = np.sign(y_pred[valid])
    return float((t == p).mean())


def simple_strategy_return(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0, fee_bps: float = 5.0) -> Optional[float]:
    """long/short one-day strategy with transaction cost."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return None
    yt, yp = y_true[valid], y_pred[valid]
    pos = np.where(yp > threshold, 1.0, np.where(yp < -threshold, -1.0, 0.0))
    gross = pos * yt
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    fee = (fee_bps / 10000.0) * turnover
    net = gross - fee
    return float(np.prod(1.0 + net) - 1.0)


def run(
    ticker: str,
    recent_years: int = 5,
    num_seeds: int = 3,
    seed_start: int = 0,
    symbol_meta_path: str = SYMBOL_META_CSV,
    top_n_externals: int = 5,
    test_ratio: float = TEST_RATIO,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    dropout_rate: float = DROPOUT,
    direction_weight: float = DIRECTION_WEIGHT,
    mag_weight: float = MAG_WEIGHT,
    window_size: int = WINDOW_SIZE,
    plot_path: str = 'ensemble_prediction_plot.png',
    result_csv: str = RESULT_CSV,
    zip_path: str = ZIP_PATH,
) -> Dict[str, Any]:
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * recent_years)).strftime('%Y-%m-%d')

    df_target = add_indicators_core(fdr_read(ticker, start, end).sort_index())

    external_meta = load_symbol_meta(symbol_meta_path)
    ext_all = fetch_external_series_parallel(external_meta, start, end, df_target.index, max_workers=8) if external_meta else pd.DataFrame(index=df_target.index)
    if not ext_all.empty:
        valid_cols = ext_all.columns[ext_all.isna().mean() <= 0.1]
        ext_all = ext_all[valid_cols]
    top_names = rank_externals_by_correlation(df_target, ext_all, top_n=top_n_externals)
    ext_sel = ext_all[top_names] if top_names else pd.DataFrame(index=df_target.index)

    summaries, preds_by_date, per_model_rows = [], {}, []
    live_predictions = []

    for i in range(num_seeds):
        seed = seed_start + i
        set_global_seed(seed)
        tf.keras.backend.clear_session()

        seq = prepare_sequences_with_live_split(df_target, ext_sel, window_size=window_size)
        X_all = seq['X']
        y_raw = seq['y']
        vol_nf = seq['vol_norm_factors']
        dates = seq['target_dates']
        inp_closes = seq['input_closes']
        tgt_closes = seq['target_closes']

        y_vn = np.clip(y_raw / vol_nf, -10.0, 10.0)
        split = max(int(len(X_all) * (1 - test_ratio)), 1)

        X_train, X_test = X_all[:split], X_all[split:]
        y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
        y_train_vn = y_vn[:split]
        vol_test = vol_nf[split:]
        dates_test = dates[split:]

        _, w, f = X_train.shape
        x_scaler = StandardScaler().fit(X_train.reshape(-1, f))
        X_train_sc = x_scaler.transform(X_train.reshape(-1, f)).reshape(X_train.shape)
        X_test_sc = x_scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape) if len(X_test) else np.zeros((0, w, f))

        y_scaler = StandardScaler().fit(y_train_vn.reshape(-1, 1))
        y_train_s = y_scaler.transform(y_train_vn.reshape(-1, 1)).flatten()

        val_idx = max(int(len(X_train_sc) * 0.8), 1)
        X_tr, y_tr = X_train_sc[:val_idx], y_train_s[:val_idx]
        X_vl, y_vl = X_train_sc[val_idx:], y_train_s[val_idx:]

        global_mean_abs = float(np.mean(np.abs(y_tr))) if len(y_tr) else 1.0
        model = build_lstm_model(
            input_shape=(w, f),
            global_mean_abs=global_mean_abs,
            lr=lr,
            dropout_rate=dropout_rate,
            direction_weight=direction_weight,
            mag_weight=mag_weight,
        )

        model.fit(
            tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(1000, seed=seed).batch(batch_size),
            validation_data=tf.data.Dataset.from_tensor_slices((X_vl, y_vl)).batch(batch_size),
            epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, min_delta=1e-6, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0),
            ],
            verbose=0,
        )

        calib_scale = 1.0
        if len(X_vl):
            pred_vl_s = model.predict(X_vl, batch_size=batch_size, verbose=0).flatten()
            pred_vl_vn = y_scaler.inverse_transform(pred_vl_s.reshape(-1, 1)).flatten()
            vol_vl = vol_nf[val_idx:split]
            pred_vl_raw = pred_vl_vn * vol_vl
            calib_scale = robust_calibration_scale(y_train_raw[val_idx:], pred_vl_raw)

        pred_test_raw = np.array([])
        if len(X_test_sc):
            pred_ts_s = model.predict(X_test_sc, batch_size=batch_size, verbose=0).flatten()
            pred_ts_vn = y_scaler.inverse_transform(pred_ts_s.reshape(-1, 1)).flatten()
            pred_test_raw = pred_ts_vn * vol_test * calib_scale

        pred_test_close = np.array(inp_closes[split:]) * (1 + pred_test_raw) if len(pred_test_raw) else np.array([])
        test_mse_ret = float(mean_squared_error(y_test_raw, pred_test_raw)) if len(pred_test_raw) else None
        test_mae_ret = float(mean_absolute_error(y_test_raw, pred_test_raw)) if len(pred_test_raw) else None
        hit_ratio = directional_hit_ratio(y_test_raw, pred_test_raw) if len(pred_test_raw) else None
        strategy_ret = simple_strategy_return(y_test_raw, pred_test_raw, threshold=0.0, fee_bps=5.0) if len(pred_test_raw) else None

        per_model_rows.append({
            'seed': seed,
            'calib_scale': calib_scale,
            'test_mse_ret': test_mse_ret,
            'test_mae_ret': test_mae_ret,
            'hit_ratio': hit_ratio,
            'strategy_return_net': strategy_ret,
            'n_test': len(pred_test_raw),
        })

        for d, p in zip(dates_test, pred_test_close):
            dt = pd.to_datetime(d).normalize()
            preds_by_date.setdefault(dt, []).append(float(p))

        if seq['live_sample'] is not None:
            live = seq['live_sample']
            x_live = x_scaler.transform(live['X']).reshape(1, w, f)
            p_live_s = model.predict(x_live, verbose=0).flatten()[0]
            p_live_vn = y_scaler.inverse_transform([[p_live_s]])[0, 0]
            p_live_raw = p_live_vn * live['vol_factor'] * calib_scale
            p_live_close = live['input_close'] * (1 + p_live_raw)
            live_predictions.append({
                'seed': seed,
                'target_date': pd.to_datetime(live['target_date']).normalize(),
                'pred_close': float(p_live_close),
                'pred_ret': float(p_live_raw),
            })

        summaries.append({
            'seed': seed,
            'calib_scale': calib_scale,
            'test_mse_ret': test_mse_ret,
            'test_mae_ret': test_mae_ret,
            'hit_ratio': hit_ratio,
            'strategy_return_net': strategy_ret,
        })

    ensemble_df = None
    if preds_by_date:
        ensemble_df = pd.DataFrame([
            {'date': k, 'pred_mean': float(np.mean(v)), 'pred_std': float(np.std(v))}
            for k, v in preds_by_date.items() if v
        ]).sort_values('date').set_index('date')

    # live 예측 집계(평가셋과 분리됨)
    predicted_price = None
    mean_pred = None
    if live_predictions:
        lp = pd.DataFrame(live_predictions)
        predicted_price = float(lp['pred_close'].median())
        mean_pred = float(lp['pred_ret'].median())

    # 동적 임계값 실제 적용
    threshold = 0.005
    if ensemble_df is not None and not ensemble_df.empty:
        recent_ret_std = float(ensemble_df['pred_mean'].pct_change(fill_method=None).dropna().tail(60).std())
        if np.isfinite(recent_ret_std):
            threshold = float(np.clip(0.5 * recent_ret_std, 0.002, 0.02))

    if mean_pred is None:
        recommendation = "데이터 부족"
        reason = "live 예측값 부족"
    elif mean_pred > threshold:
        recommendation = "매수 권고"
        reason = f"예측 수익률 {mean_pred*100:.2f}% > 임계값 {threshold*100:.2f}%"
    elif mean_pred < -threshold:
        recommendation = "매도 권고"
        reason = f"예측 수익률 {mean_pred*100:.2f}% < -임계값 {-threshold*100:.2f}%"
    else:
        recommendation = "관망"
        reason = f"예측 수익률 {mean_pred*100:.2f}%가 임계값 ±{threshold*100:.2f}% 이내"

    pd.DataFrame(summaries).to_csv(result_csv, index=False)
    pd.DataFrame(per_model_rows).to_csv(PER_MODEL_CSV, index=False)

    if ensemble_df is not None and not ensemble_df.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(ensemble_df.index, ensemble_df['pred_mean'], label='Predicted Close (mean)')
        plt.title(f'{ticker} Predicted Close (v15)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    with zipfile.ZipFile(zip_path, 'w') as zf:
        for fn in [result_csv, PER_MODEL_CSV, plot_path]:
            if os.path.exists(fn):
                zf.write(fn)

    out = {
        '종목': ticker,
        '권고': recommendation,
        '내일 예측 수익률': f"{mean_pred*100:+.2f}%" if mean_pred is not None else 'N/A',
        '내일 예측 주가': f"{int(round(predicted_price)):,}원" if predicted_price is not None else 'N/A',
        '사유': reason,
        '동적 임계값': round(threshold, 4),
        '평균 보정 스케일': round(float(np.mean([r['calib_scale'] for r in per_model_rows])), 4) if per_model_rows else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == '__main__':
    run(ticker='005930', recent_years=5, num_seeds=3)
